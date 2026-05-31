import torch
import torch.nn as nn

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
                 emb_dropout=0.1, n_layers=1):
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
        """
        super(LM_RNN, self).__init__()
        
        # Continuous representation space for discrete tokens
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Standard uni-directional Elman RNN layer
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        # Linear decoder projection mapping back to vocabulary space logits
        self.output = nn.Linear(hidden_size, output_size)

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
        
        # Step 2: Pass embeddings through the Elman recurrent structure
        # Shape transition: [B, T, Emb] -> [B, T, Hid]
        rnn_out, _ = self.rnn(emb)
        
        # Step 3: Project back to vocabulary dimensions and permute dimensions
        # Shape transitions: [B, T, Hid] -> [B, T, Vocab] -> [B, Vocab, T]
        output = self.output(rnn_out).permute(0, 2, 1)
        return output