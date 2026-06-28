import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.hid_size = hid_size

        # 1. Embedding Layer (No Dropout)
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # 2. LSTM Encoder (Baseline: Unidirectional, batch_first=True)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)
        
        # 3. Output Projections
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) 

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        # Note: enforce_sorted=False handles cases where sequences aren't strictly length-sorted
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Get the last hidden state for Intent Prediction (Unidirectional)
        last_hidden = last_hidden[-1, :, :]
            
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes -> permute to compute Loss over classes
        slots = slots.permute(0, 2, 1) 
        
        return slots, intent