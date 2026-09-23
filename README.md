# 🧠 Smart MCQ Solver Challenge

A Deep Learning and Machine Learning solution for the **Smart MCQ Solver Challenge**, built using **PyTorch, Hugging Face Transformers, and Scikit-learn**. This project predicts the **Top-3 most probable answers** for multiple-choice science questions using three different approaches and compares their performance.

---

## 📌 Project Overview

This project implements and evaluates three different models for solving multiple-choice science questions:

1. **Artificial Neural Network (ANN)**
2. **DistilBERT Transformer**
3. **TF-IDF + Logistic Regression (Classical Machine Learning)**

The final submission predicts the **Top-3 answer choices** for each question.

---

## 🚀 Features

- Complete Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Label Encoding
- TF-IDF Feature Extraction
- ANN implemented using PyTorch
- DistilBERT Fine-tuning using Hugging Face Transformers
- Logistic Regression baseline
- Weights & Biases experiment tracking
- Model Saving & Loading
- Automatic Submission File Generation

---

## 📂 Project Structure

```
├── notebook.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
├── saved_models/
│   ├── ann_model.pth
│   ├── distilbert_model.pth
│   └── logistic_regression.pkl
├── submission.csv
└── README.md
```

---

## 📊 Exploratory Data Analysis

The notebook includes:

- Dataset Shape
- Dataset Preview
- Dataset Information
- Statistical Summary
- Missing Value Analysis
- Duplicate Row Detection
- Class Distribution
- Boxplots
- Feature Analysis

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Missing value handling
- Duplicate removal
- Label Encoding
- Input Formatting
- Train–Validation Split
- TF-IDF Vectorization
- Tensor Conversion for PyTorch
- Dataset & DataLoader creation

---

# Model 1 — Artificial Neural Network (ANN)

### Architecture

- Input Layer
- Hidden Dense Layers
- ReLU Activation
- Dropout
- Output Layer

### Training

- Framework: PyTorch
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- GPU/CPU Support

---

# Model 2 — DistilBERT

Fine-tuned **distilbert-base-uncased** for text classification.

### Components

- Hugging Face Tokenizer
- DistilBERT Model
- PyTorch DataLoader
- AdamW Optimizer
- Validation Loop
- Best Model Saving

---

# Model 3 — Classical Machine Learning

Pipeline:

```
Text
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
Prediction
```

Used as a lightweight baseline model.

---

## 📈 Experiment Tracking

Experiments are tracked using **Weights & Biases (W&B)**.

Logged metrics include:

- Training Loss
- Validation Loss
- Accuracy
- Macro F1 Score
- Learning Curves

---

## 💾 Model Saving

The notebook saves trained models for future inference.

Examples:

- ANN Model (.pth)
- DistilBERT Model (.pth)
- Logistic Regression (.pkl)

---

## 📤 Prediction Pipeline

The notebook performs:

1. Load Test Data
2. Preprocess Inputs
3. Generate Predictions
4. Rank Top-3 Labels
5. Create Submission File

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Weights & Biases

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/smart-mcq-solver.git

cd smart-mcq-solver
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Notebook

Open Jupyter Notebook or Kaggle Notebook:

```bash
jupyter notebook
```

Run all cells sequentially.

---

## 📈 Workflow

```
Dataset
     │
     ▼
EDA
     │
     ▼
Data Cleaning
     │
     ▼
Preprocessing
     │
     ├─────────────┐
     ▼             ▼
 ANN          DistilBERT
     │             │
     └──────┬──────┘
            ▼
 Logistic Regression
            │
            ▼
 Comparison
            │
            ▼
 Submission.csv
```

---

## 📊 Evaluation

Models are evaluated using:

- Accuracy
- Macro F1 Score
- Validation Loss
- Training Loss

---

## 📌 Future Improvements

- RoBERTa
- DeBERTa
- Ensemble Learning
- Cross Validation
- Hyperparameter Optimization
- Better Text Augmentation

---

## 🤝 Acknowledgements

- Kaggle
- Hugging Face
- PyTorch
- Scikit-learn
- Weights & Biases

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👤 Author

**Vikash Kumar**

IIT Madras BS Degree in Data Science and Applications


