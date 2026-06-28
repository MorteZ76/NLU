import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout_rate=0.1):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    
        self.slot_out = nn.Linear(hid_size * 2, out_slot)  # * 2 for bidirectional
        self.intent_out = nn.Linear(hid_size * 2, out_int)  # * 2 for bidirectional
        
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance.long()) # utt_emb.size() = batch_size X seq_len X emb_size

        # Apply the first dropout layer immediately after the embedding layer
        utt_emb = self.dropout(utt_emb)
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the last hidden state, which is the concatenation of the last hidden states from both directions
        last_hidden = torch.cat((last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=-1)

        # Apply the second dropout immediately before the final linear projection layers
        utt_encoded = self.dropout(utt_encoded)
        last_hidden = self.dropout(last_hidden)
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        
        return slots, intent