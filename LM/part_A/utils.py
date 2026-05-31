# Add functions or classes used for data loading and preprocessing
import os
import urllib.request
import torch
import torch.utils.data as data

def download_ptb(dest_dir="dataset/PennTreeBank"):
    """
    Checks for the local presence of the Penn TreeBank (PTB) dataset and automatically 
    downloads train, validation, and test splits from the laboratory repository if missing.
    
    Args:
        dest_dir (str): Relative directory pathway to store raw .txt datasets.
    """
    os.makedirs(dest_dir, exist_ok=True)
    urls = {
        "ptb.train.txt": "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
        "ptb.valid.txt": "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
        "ptb.test.txt": "https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt"
    }
    
    for filename, url in urls.items():
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            print(f"[Dataset] Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"[Dataset] Found cached version of {filename}. Skipping download.")

def read_file(path, eos_token="<eos>"):
    """
    Parses a text file line-by-line, stripping whitespace, and appends an explicit 
    end-of-sentence (EOS) token to the end of each parsed sequence.
    
    Args:
        path (str): Filepath pointing to the raw text document.
        eos_token (str): Special string token marking sequence boundaries.
        
    Returns:
        list of str: A collection of whitespace-formatted token strings with appended boundary markers.
    """
    output = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

class Lang:
    """
    Handles bidirectional vocabulary tracking and numericalization.
    Maps arbitrary text tokens to discrete integers (ids) and maps indexes back to tokens.
    """
    def __init__(self, corpus, special_tokens=[]):
        """
        Initializes vocabulary indices over an input training corpus.
        
        Args:
            corpus (list of str): The training dataset split used to build the vocabulary.
            special_tokens (list of str): Set of structural tokens (e.g. padding, sentence boundaries).
        """
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}
        
    def get_vocab(self, corpus, special_tokens=[]):
        """
        Iterates over vocabulary tokens, prioritizing special reserved tokens at initial indexes.
        """
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class PennTreeBank(data.Dataset):
    """
    Custom PyTorch Dataset representation for autoregressive language modeling.
    Aligns each corpus sequence to define:
      - Source Sequence (X): Input tokens from sequence start up to index (N-1)
      - Target Sequence (Y): Gold-label targets shifted by one index (from index 1 to N)
    """
    def __init__(self, corpus, lang):
        """
        Preprocesses and numericalizes raw strings into integer index representations.
        """
        self.source = []
        self.target = []

        # Autoregressive shifting structure: X = tokens[0:N-1], Y = tokens[1:N]
        for sentence in corpus:
            tokens = sentence.split()
            self.source.append(tokens[0:-1])  # Input tokens context
            self.target.append(tokens[1:])    # Target tokens to predict

        # Convert strings into structural vocabulary indices
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        """
        Retrieves a paired source-target index dictionary at a given index.
        """
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        return {'source': src, 'target': trg}

    def mapping_seq(self, data_list, lang):
        """
        Translates raw token listings to their corresponding integer IDs.
        """
        res = []
        for seq in data_list:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    # Standard preprocessed PTB has no out-of-vocabulary words.
                    # Safety check is kept for validation integrity.
                    print(f'[Warning] Out-Of-Vocabulary (OOV) token discovered: {x}!')
                    break
            res.append(tmp_seq)
        return res

def collate_fn(data, pad_token, device):
    """
    Dynamic batch batching collator. Pads sequences in the current batch to the 
    length of its longest sentence. Minimizes memory consumption compared to global maximum padding.
    
    Args:
        data (list of dict): Individual dict items containing 'source' and 'target' PyTorch Tensors.
        pad_token (int): Integer index allocated for padding tokens.
        device (torch.device): Execution device destination to load tensors.
        
    Returns:
        dict: A dictionary of padded and device-mapped input-target tensors with absolute token tallies.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        
        # Create a zeroed tensor pre-filled with pad token values
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        
        # Detach resulting sequence from computational graphs to prevent gradient leaks
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    # Sort structures from longest to shortest to optimize batch packing operations
    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    # Map directly onto processing hardware to avoid CPU-to-GPU latency gaps inside the train loop
    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item