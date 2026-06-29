# **Language Modeling (LM) \- Part 1.A**

This repository contains the clean, fully-modularized implementation of Part 1.A (Baseline Vanilla Recurrent Neural Network Language Model) evaluated on the Penn TreeBank (PTB) dataset.

The baseline model is trained autoregressively to predict the next word token by optimizing cross-entropy loss over shifted inputs.

## **Assignment Requirements & Guidelines**

As instructed by the Teaching Assistants (TAs), this sub-folder is designed to fulfill the following strict operational and academic criteria:

* **Modular Architecture:** The code must be completely separated into distinct files (utils.py, model.py, functions.py, and main.py) inside this subdirectory.  
* **Mandatory Performance Threshold:** Through hyperparameter optimization and incremental modifications, the final achieved test perplexity must be strictly below 250 (![][image1]).  
* **Incremental Experimentation:** Modifications to the baseline architecture must be added one at a time. If a specific technique degrades performance, it can be removed before testing the next modification. Both successful and unsuccessful experiments must be recorded.  
* **Hyperparameter Tuning:** Active optimization of critical parameters (specifically the learning rate, embedding/hidden dimensions, and batch sizes) is required to minimize Perplexity.  
* **No Notebooks:** Only clean, bug-free, and well-documented Python scripts are accepted.

## **File and Directory Structure**

LM/  
└── Part\_A/  
    ├── utils.py              \# Data preprocessors, Lang vocabulary class, and batch collators  
    ├── model.py              \# PyTorch model definition (LM\_RNN & LM\_LSTM)  
    ├── functions.py          \# Seeding, weight initializers, train/eval loops, and plot utilities  
    ├── main.py               \# Command-line-driven execution orchestrator  
    ├── requirements.txt      \# Project library dependencies (Standard/GPU)  
    ├── cpu\_requirements.txt  \# CPU-only PyTorch library dependencies  
    ├── README.md             \# This academic documentation, execution guide, and experimental log  
    ├── dataset/              \# Stored raw Penn TreeBank text split files  
    └── bin/                  \# Saved model checkpoints, config parameters, and metrics  
        └── LSTM\_Dropout\_AdamW\_Final/  
            ├── LSTM\_Dropout\_AdamW\_Final.pt    \# Final PyTorch model state-dict checkpoint  
            ├── config.json                    \# Final recorded hyperparameters  
            └── loss\_plot.png                  \# Training vs. Validation Loss curves

## **Technical Features**

* **Strict Seeding for Reproducibility:** A fixed global seed of 1234 is applied across all random number generators (Python, NumPy, and PyTorch CPU/CUDA backends) to ensure deterministic weights initialization and sequence batching.  
* **Smart Parameter Initializations:** Applies professional initializations (Xavier Uniform for input-to-hidden projections, Orthogonal Initialization for hidden-to-hidden recurrent weights to mitigate gradient instability, and Uniform bounding for linear decoding layers).  
* **Automated Experiment Tracking & Visualization:** The save\_experiment function automatically serializes model weights, saves the exact hyperparameter configuration (config.json), and generates visual plots of the Training vs. Validation Loss curves (loss\_plot.png) for immediate visual evaluation.  
* **Flexible Action Modes:** Supports running a full training loop with early stopping, hyperparameter tuning, or loading pre-trained parameters directly to evaluate performance on the test set.

## **Mathematical Foundations**

### **Autoregressive Dataset Shifts**

For a natural sequence of tokens ![][image2], the training configuration aligns:

* **Inputs (X):** ![][image3]  
* **Targets (Y):** ![][image4]

### **Perplexity (PPL)**

The core metric evaluated is Perplexity, which mathematically represents the exponential of the average Cross-Entropy loss computed over all non-padded tokens (![][image5] total tokens):

## **![][image6]Installation & Setup**

Before running the model, make sure to install all the necessary dependencies. Navigate to the Part\_A directory and choose the appropriate requirements file based on your hardware:

**Option A: Standard / GPU Setup**

For environments with CUDA-compatible GPUs:

cd LM/Part\_A  
pip install \-r requirements.txt

**Option B: CPU-Only Setup**

If you do not have a dedicated GPU and wish to install lighter, CPU-only PyTorch binaries:

cd LM/Part\_A  
pip install \-r cpu\_requirements.txt

## **Execution Guidelines**

**Option 1: Standard Training**

Runs the full training corpus, validates performance, and outputs artifacts (checkpoints, configs, and plots) to the /bin directory.

python main.py

**Option 2: Direct Model Evaluation**

Skips training to immediately evaluate a saved model binary on the validation and test datasets.

python main.py \--eval\_only \--model\_path "bin/LSTM\_Dropout\_AdamW\_Final/LSTM\_Dropout\_AdamW\_Final.pt"

**Option 3: Hyperparameter Tuning**

Runs the sequential grid search mapped out in the configuration file.

python main.py \--tune

## **Experimental Log & Results**

Below is the record of incremental architectural modifications, hyperparameter configurations, and their corresponding validation and test Perplexities. Modifications were applied sequentially to map their exact impact on the model.

| Exp ID | Model Architecture & State | Modifications / Tuning Step | Validation PPL | Test PPL | Status |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **0** | **Baseline RNN** (LM\_RNN) | Vanilla Recurrent Baseline | 169.334 | 161.832 | Base |
| **1** | Baseline LSTM | Replaced RNN with LSTM | 154.677 | 149.522 | Kept |
| **2** | LSTM \+ Dropout | Added 2 Dropout layers | 151.661 | 146.902 | Kept |
| **3** | LSTM \+ Dropout \+ AdamW | Replaced SGD with AdamW | 133.246 | 122.838 | Kept |
| **4** | LSTM Tuning: Batch Size | batch\_size \= 16 | 132.970 | 122.477 | Kept |
| **5** | LSTM Tuning: Hidden Size | hidden\_size \= 400 | 125.861 | 117.881 | Kept |
| **6** | LSTM Tuning: Learning Rate | lr \= 0.001 (Retained Best) | 125.861 | 117.881 | Kept |
| **7** | LSTM Tuning: Embedding Size | emb\_size \= 400 | 125.286 | 116.656 | Kept |
| **8** | **Final Model** | **clip \= 0.1** | **123.429** | **115.732** | **FINAL** |

### **Experimental Observations & Discussion**

**Architectural Improvements:** The transition from a vanilla RNN to an LSTM immediately granted a massive performance boost (Test PPL dropped from \~161 to \~149), successfully mitigating the vanishing gradient problem inherent in standard RNNs. Adding dropout provided a minor but welcome regularization boost, ensuring the model generalizes better over the PTB corpus.

**Optimizer Impact:** The most significant single leap in performance came from switching the optimizer from standard SGD to AdamW. Test PPL plummeted by an impressive 24 points (146.9 → 122.8), proving that adaptive learning rate optimization with decoupled weight decay is far more effective for navigating this loss landscape.

**Hyperparameter Optimization (Sequential Grid Search):** Following the architectural lockdown, a sequential grid search systematically minimized the perplexity further:

* Increasing the network's capacity (hidden\_size to 400 and emb\_size to 400\) allowed the LSTM to capture much deeper semantic relationships in the text, shedding an additional 6 points of Test PPL.  
* Lowering the gradient clipping threshold (clip \= 0.1) ultimately stabilized the final parameter updates alongside AdamW, arriving at the absolute best performance.

**Target Achievement:** The final optimized configuration (LSTM\_Dropout\_AdamW\_Final) achieved a **Test Perplexity of 115.732**. This dramatically exceeds the mandatory project requirement of achieving ![][image1], firmly demonstrating a highly performant and stable language model.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGYAAAAaCAYAAABFPynYAAAFDUlEQVR4Xu1YXYhVVRQ+gxb9/1DT5PzcNXdmYHCI/qYflBIKiUSIKB8ifZN8CgIDBQnsBx+ql9CIGJR+QIIa8EmMZkDLCMsoH0SCikwGBSXnJR80nOn77l77zHKdc67n3hnmoc4HH/eetdbeZ+299l577ZMkFSpUqFChwiJARJ4Az4CzhufBs/r/7/7+/veGhoZuoX13d/edtVrta8gvG/tL4JTKLlMPjhS865xpF9tO9PT03OHtFxFL4MNa8AMSvq/r7e293hpgDvqhexF+9tIej9fV6/X7YLuJ/60t5MOwfQccYxvfV0tAB7vBf/Cix5z8QfAsXv5VZ2fnTUa+Rid2h7VnACHbD06DD1hdhL5rBn0+7XWLjdHR0Wvgyy6Me3tfX98gfl/C8wXwOCZYoh2eV4IXdcyRtFtr+8Pz8+AJjO1+zhf+vwVODAwM3GrtSmF4ePhmND4M/g5nuqxOOz8EzsDp1VGO523q3BprT8Buq+p25uhuh/woeBIrqcfrFwqc8K6urhu93IOLA75MWl/g4wb1fzcel6psFM8nwF/AY2j3Bn6XpR0Bg4ODfZD9Cq6PMjPel61tKaDRcvAvcDxRRyJMx+lu4taVsCvOQDZg7QnId0jOblJd4bsWArpjXwOPgCu93sMsIgahAaYrCak5XagamF1zLbOAfj14gbZG3AHZXvCQzTilgC38jDq32esgWyFhCx+J25GrC88nwUm/KmlDW22zwuoIdX6WE+J18wH6XAa+Dx6UEJAl3iYPGPtDsD8GfzZFmfb1p7KxK0oGZqdkA8OF/IkULOKm0A4z54tO8iR4jgOIctithmxGsjuiA7otqnuVz05f+K52gb7q4B7wAFb3vUnOO1sFfaOP4L54sGtg9oGfgr/h+RS43R7sGoCiwGTkTWHOEK5wbjlWEmPa2Wl09pFWIim42qGbZUEQ7ZU/gz9iglYlORNUm0uLmbOsVWjl8wXJ/17fLng2cczodxr9PhLlnFTwewRiiM+sIjH+HzhutjHzmAlAW4GRuZw/wSpEwjZuMK4WC3O+TGMXPe7sb/P2FuZduefLyMjItRyklxt0cFeg/QEJu6TuDeYLCVUV005a6BB5xYSEAqiRsqmTkF0yAWgrMOZ82eZ1eTDny2FWc17fDKLni+ScZUk4JN+EP/d4RQSqnrtgcxAD/FhcRbQQ4A5Bvz/Bh4e9Lg8xc8TxFAWgSN4UEnL+jF8hRaAd7dnO664GKbgrEUwR0O0pcRlLdw0HvFBpTIPybUxVwFL4uYFZgGlLQgo+zsUR28TA8JfPEqrRTAA0MFP+SCiEub+UvlOI3l+407yuGWpN7i96yeMZldb/ZcCgoM24zPPgr4cUPs7fKOMXDvqkcxSrtCsC4+dCsw+/fKSL3KT+/XlHQy6YNiTc0Es14mrGS7+UcE4s9/pm4CqScFO+4nzRQY9Bf0raPDO0j0apjHE8mrQQINjfDX6DtufpQ6SET1KfJ8FX7p4Pte8GWLFqu8l4jYgFAfh6tNNMMAXZC1FWCD20uQJmDfl9bKO3JXQrT0j4rhXt+X8vVtYN3t5CU99p047f03h5i9/WovyzJKcgaAXq57sSssCTSYm7TExHBUyvA7qr2C/73yihAv3Opye9F/2BfreA6/D/KILy9lWKmv8HWCFhUl7BpDzndfMBJxcBWqUTzoyRG3i+H3ZPISDP8jON11eoUKFChQoVKlSoUKHCfwX/AvNAykej+QA0AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIIAAAAaCAYAAAB7NoTTAAAFEklEQVR4Xu1YW2iURxT+l01BobdU05DbP7tJcPGphW0FoS0+WGko1eJDseZNsb7YB41VE0qphUALFmq8FIJCW0iDF/ClNkIfGppCiz6IYFJRRFtqgkLpU4vpQ/X7Mmd2T8ZdzV6yGpkPDjPnzPznzJw5c2bmj6KAgICAgICAgICAgICAgICA0pCM4zibTqdfy2azTyh5IpPJPMWSTGtrayf6vOD4SkFbtEnbYJOqKSl2iQRt0rZqfyzQ0tKyxBjT1dHR0ablqVRqEYl1zHsxfdTZ2fm07lN1cDFg9AsMqA/lTyg/d21tbW1rwf/NhWhubl6K+jhoCgvXrnWUg/b29megawi0EzQB2uzaoH83+Kuw24hyOegv0DnI67WOhQz49iXM6RT9LnPNUN7Q0PAk+FHQEfJYk62o3wH1z9ZQZcDQKhj5UBZmDDQUyY5HfQA0ziCgDAuxC/xkNQIBejaBurErWlH+SYdQzp2A+mkS6wxU1AfNYxQIMsdDzHIo14P+RWC8zDYjgc8AII8AMahfNvMdCHDuFjG+kgOC0Q0ir6fzjQoMWbQRCQwiwQlgsGucvjmijrseKfF52qNd2mcDgwz1KSOBQdAG+GE6kKkSfd4DP8iSfF7twgCPAsxluwT5MOhXbkS2od5Nf7jAEBmz9UxgCL8R/FnM/w/U1+t+oAm2gd528pKADz82dmemyRsvMpVsH6oJDgb0HWici5pTVBrqjOcI6FoN/j+WrhPrdJz070PfZRIQX4H/1rvXLBjQ1/Q5fa9kOgs72SexvUflgG/ehfw85GfUZmDW/ohHqu47Z6hz6STYOsqMl7II7l4OIPehlX1dbiBgAi2wcd2otMeFpnOYfZwMNvbAxiuQN4F+B+2gXGSTvMO4vgsJcgeb5jzI84JsvONZjuzD8exjMcGNAfmboFsmn03rZcOUd5lXDs6lYy6uyJpExN24z0jGcLhfIKDtWZIvd2CUQ98/RqU36gM/yuAkL44YkIyRwKKvcLtFssckaLn7HkjKkTNz6y4E6sIiNEfFHZZMWVSk40HjEB/ngt6tg/Yn+JXgd+W/iiK5uH8gWfEM6gcjmw2yKW+jlgT1IpjZmXJ2HTfqhYBJvwojvZE38WKBgAXL4PubnKjxgseBC2jsi6CbPJ9T0HfWqIsh21y7BwbmEfT7knUnhGyzsbfsYS13kCfbBdC0kZ3koxY6iJR9EczysejMrQP67PefzlxwZhPRsSG2d4U0ZWzTfUuGsUcBF+UE6AcofB/lz6BfuNgoB905rlEsEORiSWf9j/a3/HYB7xo9tCs2fozt5fUKaASyY6C9he4AMt4B/7JIW5DfBl01+WyWgxyD30PvZQSr8dsJpyO2O/GeHe90gH7TR5jGg8ZBSLYbAV2U+R9H2WvsBvrG2NdTl/8dF98dh27zcL34rb5blA35wdPo0jKQoCES67qvQ7FAcJCFvWcyGin7nGpiKSKXVgseK9QH6uF42Q/B8JxuF30HOBctrzXmOg7Ok/OI5KdaAX9oJCDfHqs7A/puA12DfC/b811riNT9A6HgvaISwKkroK+P5zOdBdrp70rao92oSEquFeZjHO5+oGVih6+P3AuvZoDRF2H8qLFv/kvgP2Vke33eAH0WVSlK1dl8R9GY+h1NJOl8BOfrSvYwUPVxyCuDr4ob3qLziO2PK70fzBP40+idQveK+YSxKXVdVKXgKxePyjgCAgICAgICAgICKsZdfYZ2IW5JH9gAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAABgUlEQVR4Xu2TP0sDQRDFc5BCQRs1CLlL9mITLEOw8Q9YaBlL/QA21jbRzsZCEMHaUiRNSgURC7+EpWCRYBUEUQtF4++ZPblsJHAWVvfgMXszb2fn3t5lMin+FcaYefgAu5a3+Xx+ytVFKBaLq2g+rFbx2vf9SVf3AwSHsA1bCAO3LqgB9QZ8gk1SWVfTh1wuN4bwNAzDY+ILU1VdDfCob8FdNJ9w2xUMgEYzCE+I63pFYs3VkK/Yxjus39EsupoBFAqFNU1CnGPTqztNEASj1PeIPrULeFcqlabjml+hTUywwoZZ2IH7Tn1Dddv43iTxVxemKTQNPKPkqU7TkKZ1lll7eDJ/2T9iD7kRtc70mtXV3GrrJqm/9tHTtJGHmlA2qKCDzV/8jZ7lL+yQX4YHujjlZRX5lkngb0N2RDn5Z3pfRhNWonxSfxcQnpfL5fFYrkaua+35vkDBvslwf/F1CdGjGli+wU3V2Fil6VX0/5M/gs9xLZpLLJro75oiRYqh+AIQR3fknMUiygAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABFCAYAAAD3qbryAAALuUlEQVR4Xu3df4hlZR3H8bv4A6NfWu0uu7NznntnpzbHwmI0MzUIFDT7Yf5gV9aiEspIJCQ1V6LF8o/yj3JZC9SyiEXafkjIuksuMSmk7Ea6oSbWgoYoGhpKSio6fT73PGd8fPb+mrn3zs7V9wsezjnP85znPPfegfOd55zznFoNAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgzabRaASnPH+QQgj3Ke3P8vYr3ZXm9WGZ2rp2enr6sLwAAABgpE1MTLxTgc6OPH/QdIxvKj1UbRdFMVGv1/doeVRarx9q/6dq7/I8HwAAYKQpyNm3du3aFXn+oCmQukTpCq83Go2P67gnKm3M6/VLx7hgMT4PAADAoqjX6z/yCFuePwweUVM6eWxs7N1O4+Pjnxnk6FpKgeAzeR4AAMBIUmDzdJ43DAoMj3DS8Y52oOY8rV+T1xsUtT2jxaF5PgAAwEhRUNNQELU5z885yFL6o9IreVmvtO/9RVGct3z58rfpmHVt/0bpf8o7Ka87CP5sSlu1uiwvAwAAGAnxQYN7HEDlZa2o7iqlR/P8pUz9fVWx4YY8HwAAYCSE8ob/F/P8dkY0YHvCI3l5PgAAwEhQILPDAU2e386IBmyblGbzfAAAgJHgy4VKN+X57cwnYGs0GitDOSHurNJjRVH8K0/KfyqWp8mXME/P21uoUN579/TKlSvfmpcNiz/7Yh6vEi9tH5LnAwCAERYDpEvz/HbCPAI2U+C13scYHx8/JS9L+cnRNWvWvFd17419uiWvs1CrV69+j9p7wH3Py4bF/Ve6OM8fJgXAUzrmDZ6iRemrefl8+UGQUAbcM05q8295nWHQsTbpWHvC4N58AQDA6PJojE6Kz+vEPJ2XteOgJ5QBW89PXcaTr4OwG/KyVhS4vSWUo349H6Mb9eEX+pyfzvOHRf3/ntKTef6w6LPtCsn0KAqQP5CWL1T8rf2bjKnN4/Ny87F0/B/m+f3wb+W/tTwfAPAmp5PDhUrPxMDCl+m8/qTSZbHcN667rCqf9UhG1sZ/qzo64fzco0Zp+VKjPk6or/t9+S4vG7Tqe9GxPpKXtdPp+1Nbv6vazMta8YhTmMel3374e/VSx9vi4NPrDmrGxsbWaHWZ+vJ+lV2vehc4xbwv+q0Pq1evHlfZTudpeWFs5xbPWaflbycmJo7X8vY44fBcUObvoVVAqvy/u90VK1asjMdy3u1KH1Y6W3knK52Wrmf7P6r0oNJLSZ7rTviSr9bP8HEdELvMn8PHVNqudKtHVvUdTPqzu88qP1/9OdZt1srP2Awylf+NWnIpNw/Y3F/vH9evV9q2bt26t2t5qdJWB5NxAuae//kAAIyoeAKauwFfJ5HPhvIJyubEq1q/ySe1qlzbT6XTYcT9dw9qdGPYfHJTf2d6ndKjHzrWzaEMcu/My+ZLbXxIbV0Yv++eArYYAGzL84dB/dug452r5ferv4V6OWHwhhgcO1D5p+pc4eTv3/1L2/C2y+K+Hh10MPUn7XeNv0O3n+6j/BeyNpp/s97Xy6J8u0QzmHGe2wnlPxhnKuuQZP1QLVepzpGuG+IIm5Znu2xqaupwH6f6m4n9nAvY0vVQBnqr4kjuxvg53Qcfo/nUrj+T0lHaPtrbFbfjfb3uvri/VVls98+hvHzuvm7W8iktT6gNcFQWALBE6SRxanpi8ElO20/EZfPG9Voya7629xZJABf3f7XaXuriSdFBzKKc5IpyJGdW6Zy8bD5CMg1JNerSjQOFUL71YOh0rEuS9V3VKJuOf3eIo3xa3u4ATqvLqiAo1nc/74t1rlbaqHpnTU9PH+Y8rZ/ugM3raveDzYPUmiN4HoF7oNpWO5+K9ZvBU5youPkgh+rdpnS2g0mPTIVyhGpuvWoj1p27X7Ha331tFbApXVStx33vdrsOUv25vI+P4d9MZffEOheHFm/ZiH+bDtiWqb0r3d/qe9T2H5S+5TLl/1jpKh/DdYoY5AIA3qDq5euSdhTxclbMe1h593tdyy1KL1dlccLZf1fbsc68psiYD7X7D/XtC+rT77W+s16OMr2kNOv7i7wM5WVcz6vmy2N+CtMvWfeTmL/O2zOf3KqT62JRX3bHvu7Oy3rhoGEhJ+UYCPX8sESvQvl3cZ9/G6Vj8vKUA7Mq8Krk2xX/PcbVQ31Z0StaHlfEd662e++rX3af7HuAbD8H6tVlyHR93tp9jlrSbjXCVvXP34e+u4byvvK6PdrwMZL+H5J+R96uLdI/HgCAgygGPY8U5aU7P2n3Ky2vnZycfIfLQ/mk3IuxzJeUHo/3JM0J5X1uHQMRHeddMZhqmfL6ppPweD2+OsojFiGOSCjvBK0/VytPXld66fzY13tj/VPc71qLk5kDn8UO2Hw/U3jt3rMD+tSN9ttSJKOavXKgEDoEbCr7S/5bdPtd4qjVnVUQoTbOyOsMgEeY1ntF/TipGmUaRUV56fNj1bbWt+uzfcl/E2k9AADacvASg4iWYpCxKc9PxTpzJ22fXOM9S32JAeKtSjdUqSqLT1Q+m9YPcRqGatv7twpyugVs8fPMK+VttKLj/sR1dezr8rJOYtD1fK1NoKf2vla0Ga0J85yOpBehHC18XOkhHffcvBwAAAyYTrqPhTb3OMWn0R7xKFxeVinK+9zmbuCPl0xvy+v5so5P7u1SXt8UiJyutjZW28llId/Dc5n2O6+ITwCa+1F9lupdoa0Cx24B27DouJerT3fk+d1ov13q7540T5/vfQ7k4gMIN7a7VOg6nQI2lZ2Z/xbdfpeQzRPmS3zpNgAAGLBQzvi/Jc+3eBlyR73DvUHxpu/0STbfz9a8sbpfcRRt7+Tk5HJvhxi8hXIqEgc+Dtz+04hTZoQyYNvvdV9O0/orc40llH/pwQjYdNx96mvI87vx5wjlSJ5HtE6NlyT3OAhW3ueUdiqdmO9n3QK2hdCxNxfxnrJ4qfuAUUwAAPAm44Cxlyk4YsA2o/pHdrgZ3EGMn8TrGIgOmo71cJ7Xi5BM++DPpHbOcv/TOg6g0u1UfAJyJs/vl7/jwASvAABgvqqALc/PVUFMtyBQdX4W4uhWXhZH/2adFEB9Pi9PeVQt9DmlRzt+/ZSCp4+q/a/nZVYFp3k+AADAoquX0300H0zwel6e8shQKGey7zpCFC/5Ni+zphQIfcf5re6RS6nOHZ1GwHKqW+/WZq7Ty9aL8qGSjg+NAAAALEWecf7lbvdexZnqf6m0LQ2ifB+ZU+jy8njVubzdwwC5+PDAsyGZHHcAfI/fNvc1LwAAAFjyQnk5068daqson4C9MZRzoDUnFo7vhfRN9369UqeAr/kaplCO5rVNamuDlj8Irz1YMJM3tFDxKd+7Qvb6IwAAgJGgIOa5EN/r2E4MzDya1nwNUcy7qFYGY4/Uktd1peJo2VUhmT+ul6S2r/Ml0by9hYr36r1QazN/GwAAwJIWylcreTLatlR+o+KnI+JIm+t/OSmbe13XUqU+3tTtMwIAACxZ8WGCA57+TKn8u176/jWt/1Xp2952EBdaPIiw1KiPe8OA5sQDAAA4GJYVRXFzq7c4xIBsNqa7nVfENypo+8GqTPXOf/2eS0so3wHbckJdAACAkRBfX9XxPrZ+NRqNY3WMrXl+ReXrVH5bGODDBuZAzQ805PkAAAAjR4HNc3newTCEgG2rn2jN8wEAAEaO72UriuKYPH8QFDQdrbTJT5tqebWDsjT5qdCk7kyya7/8FOtMngkAADCyFNzsU9B2Wp4/CGr7Dl96rcd3b6YpHQEbVIAVXwi/vcZUHgAA4I1EAc45Snvz/EHwBLtq+5MOzhYjYFNguF5t7czzAQAARl4cmfpEnt8vv22gtkijXVNTU4ePj48fl+cDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPCG8X+PIPbmLq8bLgAAAABJRU5ErkJggg==>