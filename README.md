

## Legal Clause Similarity (Deep Learning Assignment 02)

This repository contains the implementation and analysis for **FAST-NUCES Deep Learning Assignment 02** (Fall 2025).
The task is to identify **semantic similarity between legal clauses** using non-transformer baseline NLP models — **BiLSTM** and **Self-Attention Encoder** — trained on the *Kaggle Legal Clause Dataset*.

---


###  Task Description

> **Goal:** Determine whether two legal clauses express the same legal meaning (similar = 1, different = 0).

* **Dataset:** [Kaggle – Legal Clause Dataset](https://www.kaggle.com/datasets/bahushruth/legalclausedataset)
* **Classes:** ~350 clause categories (e.g., *Access to Information*, *Accounting Terms*)
* **Pairs:** Balanced positive / negative pairs built via in-class vs cross-class sampling.
* **Split:** 70 % train  |  15 % validation  |  15 % test.

---

###  Methodology

#### Data Pre-processing

* Texts lower-cased and tokenized via `TextVectorization` (`max_tokens = 20 000`, `seq_len = 128`).
* Pair generator with capping (`max_items_per_class = 150`) to avoid O(n²) explosion.

#### Siamese Architecture

Both models share the same encoder on each branch and combine outputs as:
`[fa, fb, |fa − fb|, fa × fb] → Dense → Sigmoid`

#### Baseline A — BiLSTM Encoder

```python
Embedding → BiLSTM(128, return_sequences=True)
→ GlobalMaxPooling1D → Dense(128→64→1)
```

#### Baseline B — Self-Attention Pooling Encoder

```python
Embedding → AttentionPooling(units=128)
→ Dense(128→64→1)
```

#### Training Setup

| Setting    | Value                                            |
| :--------- | :----------------------------------------------- |
| Optimizer  | Adam (LR = 1e-3)                                 |
| Batch size | 128                                              |
| Epochs     | ≤ 15 (Early Stopping + ReduceLROnPlateau)        |
| Loss       | Binary Cross-Entropy                             |
| Metrics    | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |

---

### Results

| Metric            | **BiLSTM** | **Self-Attention** |
| :---------------- | :--------: | :----------------: |
| Accuracy          | **0.9988** |       0.9985       |
| Precision         | **0.9975** |       0.9970       |
| Recall            | **1.0000** |     **1.0000**     |
| F1-Score          | **0.9988** |       0.9985       |
| ROC-AUC           |   0.9986   |     **0.9996**     |
| PR-AUC            |   0.9973   |     **0.9993**     |
| Training Time (s) |    883.7   |      **154.9**     |

 **Observation:**
Both models achieved near-perfect accuracy. The Self-Attention encoder matched BiLSTM performance while training **≈ 6× faster**, making it better suited for real-time legal-NLP systems.

---




### Discussion

* **Performance:** Both models generalize extremely well due to lexical consistency of legal clauses.
* **Speed:** Attention model trains much faster.
* **Error Analysis:** Remaining false positives arise from surface-level lexical overlap despite semantic differences.
* **Future Work:** Evaluate on unseen clause categories, try margin-based contrastive losses, or pre-trained transformers.

---

### Environment / Dependencies

```bash
Python 3.12
TensorFlow 2.15
Keras 3.x
NumPy / Pandas / Matplotlib / Seaborn
```

Install:

```bash
pip install -r requirements.txt
```

---


