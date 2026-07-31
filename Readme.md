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

The project uses the **Emotion Dataset** available through the Hugging Face Datasets library.

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

## Evaluation Metrics

The notebook evaluates the model using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report

---

## Example Prediction

**Input**

```text
I finally achieved my goal after months of hard work.
```

**Prediction**

```text
Joy
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
git clone https://github.com/YOUR_USERNAME/distilbert-emotion-classification.git
cd distilbert-emotion-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
distilbert-emotion-classification.ipynb
```

Run all cells sequentially.

---

## Hugging Face Model

The fine-tuned model is hosted on Hugging Face.

Replace this link with your model:

```
https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_NAME
```

---

## Future Improvements

- Improve hyperparameter tuning
- Visualize attention weights
- Add confusion matrix
- Compare DistilBERT with BERT and RoBERTa
- Deploy the model as a FastAPI service
- Build a Streamlit web application

---

## Learning Outcomes

Through this project, I gained practical experience with:

- Transformer-based NLP models
- Hugging Face Transformers
- Text tokenization
- Fine-tuning pre-trained language models
- Model evaluation
- Emotion classification
- End-to-end NLP workflow

---

## Author

**Rohit Pal**

GitHub: https://github.com/vanshR18

---

## License

This project is licensed under the MIT License.
