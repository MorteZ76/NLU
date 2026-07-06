import os
import json
import urllib.request
import random
from collections import Counter
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

PAD_TOKEN = 0


# =========================================================================
# DOWNLOAD HELPERS
# =========================================================================

def download_atis_and_conll(dest_dir="dataset/ATIS"):
    """
    Downloads the ATIS train/test dataset and the conll.py evaluation script if missing.
    """
    os.makedirs(dest_dir, exist_ok=True)
    urls = {
        "train.json": "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json",
        "test.json": "https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json"
    }
    
    for filename, url in urls.items():
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            print(f"[Dataset] Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)

    # Automatically download conll.py to the project root so it can be imported in functions.py
    if not os.path.exists("conll.py"):
        print("[Dataset] Downloading conll.py evaluation script...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py", "conll.py")

def load_data(path):
    """Load a JSON dataset file and return it as a Python object."""
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


# =========================================================================
# VOCABULARY
# =========================================================================

class Lang():
    """
    Vocabulary container for the ATIS NLU task.

    Builds and holds three bidirectional mappings:
      - word  <-> id  (with unknown token and optional frequency cutoff)
      - slot  <-> id  (with pad token at index 0)
      - intent <-> id (no pad token)
    """
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        """
        Build a word-to-id mapping, filtering tokens that appear <= cutoff times.

        Args:
            elements (list[str]): Flat list of all word tokens in the corpus.
            cutoff (int): Minimum frequency required for a word to enter the vocab.
            unk (bool): Reserve an 'unk' token for out-of-vocabulary words.

        Returns:
            dict: Mapping from token string to integer id.
        """
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        """
        Build a label-to-id mapping (intents or slots).

        Args:
            elements (iterable): Unique label strings, already in a deterministic
                order (e.g. sorted()). Unlike part_B's equivalent, this method
                does NOT sort internally — it assigns ids by iterating `elements`
                as given, so the caller (prepare_atis_data) is responsible for
                passing a pre-sorted iterable. Passing a raw, unsorted set()
                would assign a different label<->id mapping on every process run
                (Python randomizes string hash seeds per-process), silently
                corrupting any checkpoint reloaded in a later process.
            pad (bool): If True, reserve index 0 for a pad label (used for slots, not intents).

        Returns:
            dict: Mapping from label string to integer id.
        """
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab


# =========================================================================
# DATASET
# =========================================================================

class IntentsAndSlots(data.Dataset):
    """
    PyTorch Dataset for the ATIS joint intent classification and slot filling task.

    Each item is a dict with:
        'utterance' : LongTensor of word ids, shape [SeqLen]
        'slots'     : LongTensor of slot ids, shape [SeqLen]
        'intent'    : int intent id
    """
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    def mapping_lab(self, data, mapper):
        """Map a flat list of label strings to integer ids (used for intents)."""
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):
        """
        Map a list of whitespace-delimited sequences to lists of integer ids.

        Unknown tokens fall back to the 'unk' id.
        """
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


# =========================================================================
# COLLATE — dynamic padding within each batch
# =========================================================================

def collate_fn_atis(data, device):
    """
    Collate a batch of samples into padded tensors for the ATIS dataset.

    Sorts by descending sequence length so pack_padded_sequence works without explicit sorting.

    Returns:
        dict with keys 'utterances', 'intents', 'y_slots', 'slots_len'.
    """
    def merge(sequences):
        """Pad a list of variable-length tensors to the same length."""
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            padded_seqs[i, :lengths[i]] = seq
        return padded_seqs.detach(), lengths

    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


# =========================================================================
# TOP-LEVEL DATA PREPARATION
# =========================================================================

def prepare_atis_data():
    """
    Downloads dataset, runs stratified train/dev split, constructs Lang, and returns everything.
    """
    download_atis_and_conll()
    
    tmp_train_raw = load_data(os.path.join('dataset', 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join('dataset', 'ATIS', 'test.json'))
    
    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw] 
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    # sklearn's stratify= requires every class to have at least 2 examples, so any
    # intent that only appears once in the corpus can't be split at all — those go
    # straight into mini_train and get added back to train_raw after the split,
    # rather than being dropped or crashing train_test_split.
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Stratified split so every intent class is proportionally represented in both train and dev
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw 
    # sorted(), not raw set() iteration: Python randomizes string hash seeds per
    # process, so an unsorted set would assign a different label<->id mapping on
    # every run, silently corrupting any checkpoint reloaded in a later process.
    slots = sorted(set(sum([line['slots'].split() for line in corpus],[])))
    intents = sorted(set([line['intent'] for line in corpus]))

    lang = Lang(words, intents, slots, cutoff=0)
    
    return train_raw, dev_raw, test_raw, lang