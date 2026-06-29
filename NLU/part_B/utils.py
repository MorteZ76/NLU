import os
import json
import urllib.request
from collections import Counter

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

PAD_TOKEN = 0


# =========================================================================
# DOWNLOAD HELPERS
# =========================================================================

def download_atis_and_conll(dest_dir="dataset/ATIS"):
    os.makedirs(dest_dir, exist_ok=True)
    urls = {
        "train.json": "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json",
        "test.json":  "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json",
    }
    for filename, url in urls.items():
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            print(f"[Dataset] Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)

    if not os.path.exists("conll.py"):
        print("[Dataset] Downloading conll.py evaluation script...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py",
            "conll.py"
        )

def load_data(path):
    with open(path) as f:
        return json.load(f)


# =========================================================================
# LABEL VOCABULARY
# =========================================================================

class Lang:
    """Holds slot and intent label-to-id mappings (no word vocab needed for BERT)."""
    def __init__(self, intents, slots):
        self.slot2id   = self._lab2id(slots,   pad=True)
        self.intent2id = self._lab2id(intents, pad=False)
        self.id2slot   = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in sorted(elements):        # sorted for determinism across runs
            if elem not in vocab:
                vocab[elem] = len(vocab)
        return vocab


# =========================================================================
# BERT DATASET  —  handles sub-tokenization alignment
# =========================================================================

class BERTIntentsAndSlots(data.Dataset):
    """
    Tokenizes utterances with BertTokenizerFast and aligns slot labels
    to sub-token positions.

    Alignment rule (from https://arxiv.org/abs/1902.10909):
        - First sub-token of a word  → real slot label id
        - All other positions        → -100  (ignored by CrossEntropyLoss)
          This covers [CLS], [SEP], padding, and continuation sub-tokens.

    Each item stores the original words and raw slot strings so the eval
    loop can reconstruct word-level predictions without needing the tokenizer.
    """
    def __init__(self, raw_dataset, tokenizer, lang):
        self.dataset   = raw_dataset
        self.tokenizer = tokenizer
        self.lang      = lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item   = self.dataset[idx]
        words  = item['utterance'].split()
        slots  = item['slots'].split()
        intent = item['intent']

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        word_ids = encoding.word_ids()   # None for special tokens, int for real tokens

        slot_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], or padding
                slot_ids.append(-100)
            elif word_idx != prev_word_idx:
                # First sub-token of a new word → assign real label
                slot_ids.append(self.lang.slot2id.get(slots[word_idx], PAD_TOKEN))
            else:
                # Continuation sub-token → ignore
                slot_ids.append(-100)
            prev_word_idx = word_idx

        return {
            'input_ids':      encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'slot_ids':       slot_ids,
            'intent_id':      self.lang.intent2id[intent],
            # kept for eval reconstruction
            'words':          words,
            'raw_slots':      slots,
        }


# =========================================================================
# COLLATE — dynamic padding within each batch
# =========================================================================

def collate_fn(batch, device):
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids_batch      = []
    attention_mask_batch = []
    slot_ids_batch       = []
    intent_ids_batch     = []

    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids_batch.append(     item['input_ids']      + [0]    * pad_len)
        attention_mask_batch.append(item['attention_mask'] + [0]    * pad_len)
        slot_ids_batch.append(      item['slot_ids']       + [-100] * pad_len)
        intent_ids_batch.append(item['intent_id'])

    return {
        'input_ids':      torch.tensor(input_ids_batch,      dtype=torch.long).to(device),
        'attention_mask': torch.tensor(attention_mask_batch, dtype=torch.long).to(device),
        'slot_ids':       torch.tensor(slot_ids_batch,       dtype=torch.long).to(device),
        'intent_ids':     torch.tensor(intent_ids_batch,     dtype=torch.long).to(device),
        'raw_batch':      batch,   # list of dicts; used in eval for word-level reconstruction
    }


# =========================================================================
# TOP-LEVEL DATA PREPARATION
# =========================================================================

def prepare_bert_data(bert_model_name='bert-base-uncased'):
    """
    Downloads ATIS, builds Lang, tokenizes with BertTokenizerFast, and
    returns (train_dataset, dev_dataset, test_dataset, lang, tokenizer).
    """
    download_atis_and_conll()

    tmp_train_raw = load_data(os.path.join('dataset', 'ATIS', 'train.json'))
    test_raw      = load_data(os.path.join('dataset', 'ATIS', 'test.json'))

    # Stratified 90/10 train/dev split (same logic as Part A)
    intents  = [x['intent'] for x in tmp_train_raw]
    count_y  = Counter(intents)
    inputs, labels, mini_train = [], [], []

    for idx, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[idx])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[idx])

    X_train, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=0.10, random_state=42,
        shuffle=True, stratify=labels
    )
    X_train.extend(mini_train)
    train_raw, dev_raw = X_train, X_dev

    # Build label vocab from the full corpus so test labels are never unknown
    corpus  = train_raw + dev_raw + test_raw
    slots   = set(sum([line['slots'].split()  for line in corpus], []))
    intents = set(line['intent'] for line in corpus)
    lang    = Lang(intents, slots)

    tokenizer     = BertTokenizerFast.from_pretrained(bert_model_name)
    train_dataset = BERTIntentsAndSlots(train_raw, tokenizer, lang)
    dev_dataset   = BERTIntentsAndSlots(dev_raw,   tokenizer, lang)
    test_dataset  = BERTIntentsAndSlots(test_raw,  tokenizer, lang)

    print(f"[Dataset] Train: {len(train_raw)} | Dev: {len(dev_raw)} | Test: {len(test_raw)}")
    print(f"[Dataset] Intents: {len(lang.intent2id)} | Slots: {len(lang.slot2id)}")

    return train_dataset, dev_dataset, test_dataset, lang, tokenizer