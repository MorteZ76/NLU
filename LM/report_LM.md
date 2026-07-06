# NLU course project

*[Your Name] (mat. [Your Matriculation Number])*

University of Trento
[your.email]@studenti.unitn.it

The following is the report for the first part of the project: Language Modeling.

## 1. Introduction

This part of the project builds an autoregressive next-word predictor and improves it step by step, using test-set Perplexity (PPL, lower is better) as the evaluation metric throughout. Part A starts from a vanilla Elman RNN and incrementally swaps in an LSTM cell, adds embedding/output dropout, and replaces SGD with AdamW, followed by a sequential hyperparameter search. Part B restarts from that same LSTM baseline and layers in three regularization techniques from Merity et al. (2017) — weight tying, variational dropout, and non-monotonically triggered ASGD — again followed by targeted tuning. Every modification was applied one at a time and validated before moving to the next.

## 2. Implementation details

### 2.1 Part A

Starting from a vanilla `nn.RNN` baseline, I swapped it for `nn.LSTM` with no other change, to isolate the architectural effect. I then added two `nn.Dropout` layers — one right after the embedding lookup, one right before the output projection — as standalone modules rather than relying on `nn.LSTM`'s built-in dropout argument, which has no effect at `num_layers=1`. Finally, I replaced SGD with AdamW while keeping everything else fixed, isolating the optimizer's own contribution. Hyperparameters (learning rate, hidden/embedding size, gradient clipping) were then tuned sequentially — fixing each parameter at its best value before searching the next, rather than a full grid — to keep the search tractable. Batch size and patience were left at their defaults throughout.

### 2.2 Part B

Part B reuses Part A's LSTM as its starting point. On top of it, I implemented weight tying by directly assigning the output projection's weight matrix to the embedding matrix, which requires `emb_size == hidden_size` (enforced with an explicit check) [5]. I then implemented variational dropout myself as a small custom module, since PyTorch has no built-in layer that samples one dropout mask per sequence and reuses it at every timestep instead of resampling per step [7]. Finally, I implemented NT-AvSGD following the trigger rule in Merity et al. (2017) [9]: validation loss is monitored every epoch, and the optimizer switches from SGD to ASGD once it fails to beat its best value from more than `non_mono` epochs ago. While ASGD is active, the model's live weights are temporarily swapped for ASGD's running average before every evaluation, then restored, so validation and testing always reflect the averaged parameters.

## 3. Results

All figures are test-set Perplexity, `PPL = exp(total loss / total non-pad tokens)`, computed on the PTB test split; validation PPL (same formula, dev split) drove early stopping and the hyperparameter search.

### 3.1 Part A

Moving from RNN to LSTM gave the largest single architectural gain, consistent with LSTM gating mitigating vanishing gradients. The two dropout layers added a further, smaller improvement. Switching SGD for AdamW produced the single biggest reduction in this part, reflecting AdamW's per-parameter adaptive rates and decoupled weight decay. The subsequent sequential search — mainly enlarging hidden/embedding size to 400 and tightening gradient clipping to 0.1 — reached a final Test PPL of **115.73**, well under the required 250.

*Table 1: Best perplexity results for each configuration (Part A).*

| Configuration | Modification | Val PPL | Test PPL |
|:---|:---|:---:|:---:|
| Baseline RNN | Vanilla recurrent baseline | 169.33 | 161.83 |
| Baseline LSTM | Replaced RNN with LSTM | 154.68 | 149.52 |
| LSTM + Dropout | Added embedding + output dropout | 151.66 | 146.90 |
| + AdamW | Replaced SGD with AdamW | 133.25 | 122.84 |
| + Tuning: batch size | `batch_size=16` | 132.97 | 122.48 |
| + Tuning: hidden size | `hidden_size=400` | 125.86 | 117.88 |
| + Tuning: learning rate | `lr=0.001` (unchanged from baseline) | 125.86 | 117.88 |
| + Tuning: embedding size | `emb_size=400` | 125.29 | 116.66 |
| **Final** | `clip=0.1` | **123.43** | **115.73** |

### 3.2 Part B

A first tuning pass over the learning rate was necessary since the default SGD rate diverged entirely; `lr=5` stabilized training at a Test PPL on par with Part A's LSTM despite using plain SGD. Weight tying alone barely changed the score numerically, but it made embedding size a meaningful hyperparameter — tuning it gave the first result below the Part A baseline. Variational dropout then delivered the largest single gain in this part. Finally, NT-AvSGD with a tight trigger window (`non_mono=1`) gave the final improvement; wider windows (2–5) never triggered early enough to beat the pre-ASGD result. The final model reaches Test PPL **128.73**, below both the 250 threshold and Part A's LSTM baseline (149.52).

*Table 2: Best perplexity results for each configuration (Part B).*

| Configuration | Modification | Val PPL | Test PPL |
|:---|:---|:---:|:---:|
| Baseline RNN + SGD (`lr=1`) | Default settings, no regularization | 7891.02 | 7918.07 (diverged) |
| RNN + SGD | Tuning: `lr=5` | 156.76 | 150.92 |
| + Weight Tying | `emb_size = hidden_size = 200` | 155.42 | 150.84 |
| + Tuning: embedding size | `emb_size=200` confirmed optimal | 149.24 | 145.35 |
| + Variational Dropout | `emb_drop=0.1`, `out_drop=0.1` | 140.89 | 136.01 |
| **+ NT-AvSGD** | `non_mono=1` | **132.97** | **128.73** |

## 4. References

[1] T. Mikolov, M. Karafiát, L. Burget, J. Černocký, and S. Khudanpur, "Recurrent neural network based language model," in *INTERSPEECH 2010*, 2010, pp. 1045–1048.

[2] PyTorch, "Long Short-Term Memory (LSTM)," https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html.

[3] PyTorch, "Dropout," https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html.

[4] PyTorch, "AdamW," https://docs.pytorch.org/docs/stable/generated/torch.nn.AdamW.html.

[5] O. Press and L. Wolf, "Using the Output Embedding to Improve Language Models," *arXiv preprint arXiv:1608.05859*, 2017.

[6] PyTorch, "ASGD," https://docs.pytorch.org/docs/stable/generated/torch.optim.ASGD.html.

[7] Y. Gal, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks," *arXiv preprint arXiv:1512.05287*, 2016.

[8] PyTorch, "LSTMCell," https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html.

[9] S. Merity, N. S. Keskar, and R. Socher, "Regularizing and Optimizing LSTM Language Models," in *International Conference on Learning Representations (ICLR)*, 2018. Code: https://github.com/salesforce/awd-lstm-lm.
