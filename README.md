# NLP Tweet Classification Challenge

Fine-tuning [BERTweet](https://huggingface.co/vinai/bertweet-base) for two tweet classification tasks: **irony detection** and **emotion classification**. Both tasks are evaluated using macro F1-score.

---

## Tasks

### Task 1 — Irony Detection
Binary classification: predict whether a tweet is ironic (`1`) or non-ironic (`0`).

- **Dataset:** 2,862 training / 955 validation / 784 test tweets (test labels not released — challenge submission only)
- **Model:** `vinai/bertweet-base` + custom FC head (Dropout → Linear)
- **Best checkpoint val F1:** `0.7667` (model saved by best val loss)

### Task 2 — Emotion Classification
Multi-class classification: predict the dominant emotion in a tweet.

| Label | Emotion  |
|-------|----------|
| 0     | Anger    |
| 1     | Joy      |
| 2     | Optimism |
| 3     | Sadness  |

- **Dataset:** 3,257 training / 374 validation / 1,421 test tweets (test labels not released — challenge submission only; class imbalance: anger >> joy, optimism, sadness)
- **Model:** `vinai/bertweet-base` + custom FC head (Dropout → Linear)
- **Best checkpoint val F1:** `0.8147` (model saved by best val F1)

---

## Approach

### Tweet Preprocessing
Both tasks use domain-specific preprocessing tailored for BERTweet:
- URLs → `HTTPURL`, mentions → `@USER` (matching BERTweet's pretraining vocabulary)
- Emojis converted to textual form (e.g., 😊 → `:smiling_face:`); unknown emoji tokens added to the model vocabulary
- Emoticons converted via `emot` library
- Letter repetitions reduced (e.g., `soooo` → `soo`)
- Hashtags segmented using `wordsegment`

### Model Architecture
Custom classifier head on top of the fine-tuned BERTweet encoder:
```
BERTweet [CLS] representation → Dropout → Linear → logits
```
Training with early stopping on validation loss (irony) / F1 (emotion). Hyperparameter search over learning rate, dropout, and batch size.

### Data Augmentation (Emotion Task)
The emotion dataset is imbalanced (anger is the majority class). To address this:
1. Cosine similarity computed between hand-crafted seed tweets and each class distribution using [Twitter4SSE](https://huggingface.co/digio/Twitter4SSE) sentence embeddings — to verify that seeds are representative.
2. DeepSeek-V3 used to generate paraphrases of minority-class tweets (joy, optimism, sadness), preserving emotional tone.
3. Augmented samples merged with the original training set.

---

## Results

Reported on the **validation set** (test set labels were not released — test tweets were used for challenge submission only, see `predictions/`).

| Task | Model | Val F1 |
|------|-------|--------|
| Irony Detection | BERTweet-base fine-tuned | **0.7667** |
| Emotion Classification | BERTweet-base fine-tuned + data aug. | **0.8147** |

---

## Repository Structure

```
NLP-Tweet-CLS-Challenge/
├── irony_submission.ipynb           # Irony detection pipeline
├── emotion_submission.ipynb         # Emotion classification pipeline
├── data_augmentation_process.ipynb  # DeepSeek-V3 augmentation process
├── irony/                           # Irony task data files
│   ├── train_text.txt / train_labels.txt
│   ├── val_text.txt / val_labels.txt
│   ├── test_text.txt
│   ├── mapping.txt
│   └── abbreviations.json
├── emotion/                         # Emotion task data files
│   ├── train_text.txt / train_labels.txt
│   ├── val_text.txt / val_labels.txt
│   ├── test_text.txt
│   ├── mapping.txt
│   └── abbreviations.json
├── emotion_train_aug/               # Augmented training data for emotion task
└── predictions/
    ├── irony_predictions.csv
    └── emotion_predictions.csv
```

---

## Tech Stack

- **PyTorch** — model training and custom classifier
- **HuggingFace Transformers** — BERTweet tokenizer and model
- **sentence-transformers** — Twitter4SSE embeddings for augmentation validation
- **DeepSeek-V3** — LLM-based tweet paraphrasing for data augmentation
- **scikit-learn** — metrics (macro F1), hyperparameter grid search
- **emoji / emot / wordsegment** — tweet-specific text preprocessing
