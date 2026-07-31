# DistilBERT Emotion Classification

A Natural Language Processing (NLP) project that fine-tunes **DistilBERT** for multi-class emotion classification using the Hugging Face Transformers ecosystem.

The model predicts one of six emotions from an input sentence:

- 😊 Joy
- 😢 Sadness
- ❤️ Love
- 😠 Anger
- 😨 Fear
- 😲 Surprise

---

## Project Overview

This project demonstrates the complete workflow of fine-tuning a pre-trained transformer model for text classification, including:

- Exploratory Data Analysis (EDA)
- Text preprocessing
- Tokenization
- Fine-tuning DistilBERT
- Model evaluation
- Emotion prediction on custom text
- Model deployment to Hugging Face Hub

---

## Dataset
 
- Source: HuggingFace [`emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset (English Twitter messages)
- Splits: 16,000 train / 2,000 validation / 2,000 test
- Labels are heavily imbalanced (`joy` and `sadness` dominate; `surprise` is rare)


Each sample consists of:

| Text | Label |
|------|------|
| "I am feeling amazing today." | Joy |
| "Everything is going wrong." | Sadness |

### Emotion Classes

| Label ID | Emotion |
|----------|----------|
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

---

## Project Workflow

```text
Load Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Text Tokenization
      │
      ▼
Fine-tune DistilBERT
      │
      ▼
Model Evaluation
      │
      ▼
Emotion Prediction
      │
      ▼
Push Model to Hugging Face
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Evaluate
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Model Architecture

```
Input Sentence
       │
       ▼
Tokenizer
       │
       ▼
Input IDs + Attention Mask
       │
       ▼
DistilBERT Encoder
       │
       ▼
Classification Head
       │
       ▼
Emotion Prediction
```

---

## Training

The model was fine-tuned using the Hugging Face `Trainer` API.

Training included:

- Dataset tokenization
- Batch processing
- Evaluation after each epoch
- Accuracy and F1-score metrics
- Model checkpointing

---
## Pipeline
 
1. **Load & explore data** — load via `datasets`, convert to pandas, inspect class distribution and tweet length by label.
2. **Tokenization** — `AutoTokenizer` (`bert-base-uncased` vocab), padding + truncation, batch-mapped over the dataset.
3. **Model** — `AutoModelForSequenceClassification` on top of `distilbert-base-uncased` (6-way classification head), with `id2label`/`label2id` mappings for the 6 emotions.
4. **Training** — HuggingFace `Trainer` + `TrainingArguments`:
   - epochs: 2
   - learning rate: 2e-5
   - batch size: 64 (train & eval)
   - weight decay: 0.01
   - eval strategy: per epoch
   - metrics: accuracy, weighted F1
5. **Evaluation** — predictions on the test split, full `classification_report`, plus a manual single-sentence inference check.
6. **Save & serve** — model pushed to the Hugging Face Hub (`push_to_hub`), then reloaded via `pipeline("text-classification", ...)` for inference, with a bar chart of per-label confidence scores for a sample sentence.

---

## Results
 
| Metric | Score |
|---|---|
| Test Accuracy | 91.3% |
| Test F1 (weighted) | 0.913 |
| Test Loss | 0.230 |
 
**Per-class performance:**
 
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| sadness | 0.95 | 0.96 | 0.96 | 581 |
| joy | 0.93 | 0.93 | 0.93 | 695 |
| love | 0.77 | 0.79 | 0.78 | 159 |
| anger | 0.91 | 0.93 | 0.92 | 275 |
| fear | 0.89 | 0.88 | 0.88 | 224 |
| surprise | 0.77 | 0.65 | 0.70 | 66 |
 
`love` and `surprise` lag behind the others — likely due to fewer training examples (class imbalance) and semantic overlap with `joy`/`fear`.
 

---

## Usage
 
To use the already fine-tuned model directly (skipping training), run only the "Loading Pipeline" section near the end:
 
```python
from transformers import AutoModelForSequenceClassification, DistilBertTokenizerFast, pipeline
 
model_id = "NOtda/distilbert-finetuned-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
 
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
classifier("I saw a movie today and it was really good.")
```
---

## Project Structure

```
distilbert-emotion-classification/
│
├── distilbert-emotion-classification.ipynb
├── README.md
├── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vanshR18/distilbert-emotion-classification.git
cd distilbert-emotion-classification
```



## License

This project is licensed under the MIT License.
