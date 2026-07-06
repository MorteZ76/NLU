import torch.nn as nn
from transformers import BertModel


class JointBERT(nn.Module):
    """
    Fine-tuned BERT joint model for intent classification and slot filling,
    following https://arxiv.org/abs/1902.10909.

    Sub-tokenization strategy:
        Slot labels are aligned to the FIRST sub-token of each word at the
        data level (utils.py). All other positions ([CLS], [SEP], padding,
        and non-first sub-tokens) are marked -100 so CrossEntropyLoss
        ignores them automatically. This model just produces logits for
        every token position — no special handling needed here.

    Outputs:
        slot_logits   : [B, num_slots, seq_len]  (permuted for CrossEntropyLoss)
        intent_logits : [B, num_intents]
    """
    def __init__(self, num_slots, num_intents, dropout_rate=0.1,
                 model_name='bert-base-uncased'):
        """
        Args:
            num_slots (int): Number of slot label classes (output size for slot filling).
            num_intents (int): Number of intent classes (output size for intent classification).
            dropout_rate (float): Dropout probability applied after BERT, before both output heads.
            model_name (str): Hugging Face checkpoint name, e.g. 'bert-base-uncased' or
                               'bert-large-uncased'.
        """
        super(JointBERT, self).__init__()
        self.bert    = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size  = self.bert.config.hidden_size   # 768 base / 1024 large

        self.slot_classifier   = nn.Linear(hidden_size, num_slots)
        self.intent_classifier = nn.Linear(hidden_size, num_intents)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): Token id tensor of shape [Batch, SeqLen].
            attention_mask (torch.Tensor): 1 for real tokens, 0 for padding, shape [Batch, SeqLen].

        Returns:
            tuple:
                slot_logits   (torch.Tensor): Shape [Batch, NumSlots, SeqLen].
                intent_logits (torch.Tensor): Shape [Batch, NumIntents].
        """
        outputs         = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # [B, seq_len, H]
        cls_output      = self.dropout(sequence_output[:, 0, :])   # [B, H]  ([CLS] token)

        # Permute slots to [B, num_slots, seq_len] for CrossEntropyLoss
        slot_logits   = self.slot_classifier(sequence_output).permute(0, 2, 1)
        intent_logits = self.intent_classifier(cls_output)
        return slot_logits, intent_logits