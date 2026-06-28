import os
import torch
import urllib.request
from torch.utils.data import Dataset

# --------------------------------------------------------
# ATIS Preprocessing & Dataloading
# --------------------------------------------------------

class Lang:
    """
    Handles bidirectional vocabulary tracking for Words, Slots, and Intents for the ATIS dataset.
    """
    def __init__(self, corpus, special_tokens=["<pad>", "<eos>", "<unk>"]):
        # Word Vocab
        self.word2id = {}
        self.id2word = {}
        
        # Slot Vocab
        self.slot2id = {}  
        self.id2slot = {}
        
        # Intent Vocab
        self.intent2id = {}
        self.id2intent = {}
        
        # Add special tokens to words
        for idx, token in enumerate(special_tokens):
            self.word2id[token] = idx
            self.id2word[idx] = token
            
        # NOTE: You will need to implement your corpus parsing logic here 
        # to populate word2id, slot2id, and intent2id from the ATIS dataset files.

class ATISDataset(Dataset):
    """
    Custom PyTorch Dataset representation for Joint Intent & Slot Filling on ATIS.
    """
    def __init__(self, corpus, lang):
        self.utterances = []
        self.slots = []
        self.intents = []
        # NOTE: Populate the above lists using your parsed ATIS corpus

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return {
            'utterances': torch.LongTensor(self.utterances[idx]),
            'y_slots': torch.LongTensor(self.slots[idx]),
            'intents': torch.LongTensor([self.intents[idx]]),
            'slots_len': len(self.utterances[idx])
        }

def collate_fn_atis(data, pad_token, device):
    """
    Dynamic batch batching collator tailored for ATIS samples.
    """
    data.sort(key=lambda x: x['slots_len'], reverse=True)
    
    lengths = [sample['slots_len'] for sample in data]
    max_len = max(lengths)
    
    batch_size = len(data)
    
    padded_utterances = torch.LongTensor(batch_size, max_len).fill_(pad_token)
    padded_slots = torch.LongTensor(batch_size, max_len).fill_(pad_token)
    intents = torch.LongTensor(batch_size)
    
    for i, sample in enumerate(data):
        end = sample['slots_len']
        padded_utterances[i, :end] = sample['utterances']
        padded_slots[i, :end] = sample['y_slots']
        intents[i] = sample['intents'][0]
        
    return {
        'utterances': padded_utterances.to(device),
        'y_slots': padded_slots.to(device),
        'intents': intents.to(device),
        'slots_len': torch.LongTensor(lengths).to(device)
    }

def get_atis_dataloaders(batch_size, pad_token, device):
    """
    Returns populated ATIS DataLoaders and the Language object.
    (Replace this mock setup with your actual ATIS loading/parsing code)
    """
    # NOTE: Replace with actual dataset instantiations
    lang = Lang([])
    train_loader = [] 
    dev_loader = []
    test_loader = []
    return train_loader, dev_loader, test_loader, lang