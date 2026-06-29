import torch.nn as nn


class VariationalDropout(nn.Module):
    """
    Applies a consistent dropout mask across the time (sequence) dimension.
    This preserves sequential coherence compared to standard independent element-wise dropout.
    """
    def __init__(self, p=0.5):
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)


class LM_RNN(nn.Module):
    """
    Standard Elman Recurrent Neural Network (RNN) Language Model.
    
    Architecture:
      Input (Token IDs) [Batch, SeqLen]
      ---> nn.Embedding [Batch, SeqLen, EmbSize]
      ---> nn.RNN [Batch, SeqLen, HiddenSize]
      ---> nn.Linear (Projection) [Batch, SeqLen, VocabSize]
      ---> Output (Logits) Permuted to [Batch, VocabSize, SeqLen] (as expected by CrossEntropyLoss)
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, weight_tying=False):
        """
        Defines the layer blocks required for vanilla recurrence processing.
        
        Args:
            emb_size (int): Continuous embedding space dimensionality.
            hidden_size (int): Hidden state dimensions of the RNN.
            output_size (int): Total unique classes in vocabulary.
            pad_index (int): Padding index ignored by embedding gradients.
            out_dropout (float): Dropout probability for regularization.
            emb_dropout (float): Dropout probability applied to inputs.
            n_layers (int): Recurrent layer stack height.
            weight_tying (bool): Share embedding and output projection weights.
        """
        super(LM_RNN, self).__init__()

        self.weight_tying = weight_tying
        
        # Continuous representation space for discrete tokens
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Variational dropout layers for embeddings and outputs
        self.emb_dropout = VariationalDropout(emb_dropout)
        self.out_dropout = VariationalDropout(out_dropout)
        
        # Standard uni-directional Elman RNN layer
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        # Linear decoder projection mapping back to vocabulary space logits
        self.output = nn.Linear(hidden_size, output_size)

        if self.weight_tying:
            if emb_size != hidden_size:
                raise ValueError("Weight tying requires emb_size == hidden_size.")
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        """
        Executes a forward pass over batched sequences.
        
        Args:
            input_sequence (torch.Tensor): Tensor shape of [Batch Size, Sequence Length]
            
        Returns:
            torch.Tensor: Logits shaped as [Batch Size, Vocabulary Size, Sequence Length]
        """
        # Step 1: Map input IDs to dense continuous vectors
        # Shape transition: [B, T] -> [B, T, Emb]
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        
        # Step 2: Pass embeddings through the Elman recurrent structure
        # Shape transition: [B, T, Emb] -> [B, T, Hid]
        rnn_out, _ = self.rnn(emb)
        rnn_out = self.out_dropout(rnn_out)
        
        # Step 3: Project back to vocabulary dimensions and permute dimensions
        # Shape transitions: [B, T, Hid] -> [B, T, Vocab] -> [B, Vocab, T]
        output = self.output(rnn_out).permute(0, 2, 1)
        return output


class LM_LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) Language Model.

    Architecture:
      Input (Token IDs) [Batch, SeqLen]
      ---> nn.Embedding [Batch, SeqLen, EmbSize]
      ---> nn.Dropout (embedding regularization)
      ---> nn.LSTM [Batch, SeqLen, HiddenSize]
      ---> nn.Dropout (output regularization)
      ---> nn.Linear (Projection) [Batch, SeqLen, VocabSize]
      ---> Output (Logits) Permuted to [Batch, VocabSize, SeqLen] (as expected by CrossEntropyLoss)
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        """
        Defines the layer blocks required for LSTM-based sequence modeling.

        Args:
            emb_size (int): Continuous embedding space dimensionality.
            hidden_size (int): Hidden state dimensions of the LSTM.
            output_size (int): Total unique classes in vocabulary.
            pad_index (int): Padding index ignored by embedding gradients.
            out_dropout (float): Dropout probability applied after LSTM output.
            emb_dropout (float): Dropout probability applied after embedding.
            n_layers (int): Recurrent layer stack height.
        """
        super(LM_LSTM, self).__init__()

        # Continuous representation space for discrete tokens
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Dropout applied after embedding to regularize input representations
        self.emb_dropout = nn.Dropout(emb_dropout)

        # LSTM layer for sequential context modeling
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        self.pad_token = pad_index

        # Dropout applied before the final projection to prevent feature co-adaptation
        self.out_dropout = nn.Dropout(out_dropout)

        # Linear decoder projection mapping hidden states back to vocabulary logits
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        """
        Executes a forward pass over batched sequences.

        Args:
            input_sequence (torch.Tensor): Tensor shape of [Batch Size, Sequence Length]

        Returns:
            torch.Tensor: Logits shaped as [Batch Size, Vocabulary Size, Sequence Length]
        """
        # Step 1: Map input IDs to dense continuous vectors; apply embedding regularization
        # Shape transition: [B, T] -> [B, T, Emb]
        emb = self.emb_dropout(self.embedding(input_sequence))

        # Step 2: Pass regularized embeddings through the LSTM
        # Shape transition: [B, T, Emb] -> [B, T, Hid]
        lstm_out, _ = self.lstm(emb)

        # Step 3: Apply output dropout, project to vocabulary dimensions, and permute
        # Shape transitions: [B, T, Hid] -> [B, T, Vocab] -> [B, Vocab, T]
        output = self.output(self.out_dropout(lstm_out)).permute(0, 2, 1)

        return output