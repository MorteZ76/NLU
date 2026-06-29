import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    """
    Joint Intent Classification and Slot Filling model.

    Architecture:
      Input (Token IDs) [Batch, SeqLen]
      ---> nn.Embedding [Batch, SeqLen, EmbSize]
      ---> Dropout
      ---> Bidirectional LSTM [Batch, SeqLen, HidSize*2]
      ---> Dropout
      ---> Slot head:   nn.Linear -> [Batch, NumSlots, SeqLen]  (permuted for CrossEntropyLoss)
      ---> Intent head: nn.Linear -> [Batch, NumIntents]        (from concatenated final hidden states)
    """
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0,
                 dropout_rate=0.1, bidirectional=True):
        """
        Args:
            hid_size (int): Hidden state size of the LSTM (per direction).
            out_slot (int): Number of slot classes (output size for slot filling).
            out_int (int): Number of intent classes (output size for intent classification).
            emb_size (int): Word embedding dimensionality.
            vocab_len (int): Total vocabulary size.
            n_layer (int): Number of LSTM layers.
            pad_index (int): Padding index ignored by embedding gradients.
            dropout_rate (float): Dropout probability applied after embedding and before output heads.
            bidirectional (bool): If True, the LSTM encodes both forward and backward directions.
        """
        super(ModelIAS, self).__init__()

        self.bidirectional = bidirectional
        enc_dim = hid_size * 2 if bidirectional else hid_size

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)
        self.slot_out = nn.Linear(enc_dim, out_slot)
        self.intent_out = nn.Linear(enc_dim, out_int)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, utterance, seq_lengths):
        """
        Args:
            utterance (torch.Tensor): Token ID tensor of shape [Batch, SeqLen].
            seq_lengths (torch.Tensor): True (unpadded) length of each sequence in the batch.

        Returns:
            tuple:
                slots  (torch.Tensor): Slot logits, shape [Batch, NumSlots, SeqLen].
                intent (torch.Tensor): Intent logits, shape [Batch, NumIntents].
        """
        # Step 1: Embed input tokens and apply dropout
        # Shape transition: [B, T] -> [B, T, Emb]
        utt_emb = self.dropout(self.embedding(utterance.long()))

        # Step 2: Pack padded sequences to skip computation over padding positions,
        # then pass through the bidirectional LSTM
        # Shape transition: [B, T, Emb] -> [B, T, Hid*2]
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, _) = self.utt_encoder(packed_input)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Step 3: Build the sentence representation for intent classification
        # Bidirectional: concatenate forward and backward final hidden states
        # Unidirectional: use the single final hidden state directly
        if self.bidirectional:
            last_hidden = torch.cat((last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=-1)
        else:
            last_hidden = last_hidden[-1, :, :]

        # Step 4: Apply dropout before both output heads
        utt_encoded = self.dropout(utt_encoded)
        last_hidden = self.dropout(last_hidden)

        # Step 5: Project to slot and intent label spaces
        # Permute slots to [B, NumSlots, T] as required by CrossEntropyLoss
        slots = self.slot_out(utt_encoded).permute(0, 2, 1)
        intent = self.intent_out(last_hidden)

        return slots, intent