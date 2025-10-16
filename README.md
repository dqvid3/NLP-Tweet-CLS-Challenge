# NLP Tweet Classification 

This repository contains code and datasets for two tweet classification tasks:

- **Irony Detection**: Classifying tweets as ironic (1) or non-ironic (0).
- **Emotion Classification**: Classifying tweets into one of the following emotion categories:
  - **0**: Anger
  - **1**: Joy
  - **2**: Optimism
  - **3**: Sadness

## Models and Techniques Used

- **Fine-tuning** of NLP models for both tasks.
- **Data Augmentation** for the emotion task, generating paraphrases using DeepSeek-V3 to address class imbalance.

## Usage Instructions

1. **Run the training notebook** for the desired task:
   - For irony detection: run `irony_submission.ipynb`.
   - For emotion classification: run `emotion_submission.ipynb`.
2. **Make predictions** using the fine-tuned model and save results.

