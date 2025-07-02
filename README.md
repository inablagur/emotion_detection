# 😊 Emotion Detection Project

## 📖 Overview  
Build and compare multiple text-based emotion classifiers
- 🏗️ **Shallow baseline**: (e.g. Logistic Regression, SVM, Naïve Bayes over TF–IDF features)
- 🤖 **Transformer fine-tuning**: BERT-base-uncased & DistilBERT

We’ll evaluate each on accuracy, macro-F1, training time, and memory footprint, then draw final conclusions about their trade-offs.\

---

## 📦 Dataset  
We use the **Kaggle “Emotion”** dataset (28 k samples, 6 labels - anger; fear; joy; love; sadness; surprise):  
- ✔️ Already labeled (CSV)  
- ⚖️ Balanced for clear comparisons  
- ⏱️ Small enough to fine-tune in <30 min on GPU  

---

## 🗂️ Project Structure
```text
emotion_detection_project/
├── data/                # raw & processed data
├── notebooks/           # exploration & training notebooks
├── scripts/             # batch/CI scripts
├── models/              # saved checkpoints
├── README.md            # this file
├── requirements.txt     # Python deps
└── .gitignore           # ignore rules
```
---

## 🔄 General Pipeline

### 1️⃣ Shallow Baseline  
1. **Preprocessing**: lowercase, strip punctuation, optional stop-words  
2. **Vectorization**: TF–IDF (`TfidfVectorizer(ngram_range=(1,2), max_features=10k)`)  
3. **Classification**: Logistic Regression (or SVM / Naïve Bayes)  
4. **Evaluation**: train/dev/test split → accuracy, macro-F1, per-class F1

### 2️⃣ Transformer Fine-Tuning  
1. **Tokenization**: `AutoTokenizer` (`bert-base-uncased`)  
2. **Model**: `AutoModelForSequenceClassification` (+ 6-label head)  
3. **Training**: Hugging Face `Trainer` API  
4. **Evaluation**: same splits & metrics as baseline

---

## 📊 Comparison & Visualization  
- **Metrics Table**: training time, test accuracy, macro-F1 for each model  
- **Bar Charts**: side-by-side accuracy and F1 comparisons  
- **Confusion Matrices**: per-class error patterns  
- **Optional Explainability**:  
  - SHAP / LIME for classical & deep models  
  - Attention-map visuals with BertViz or Captum  

---

## ⚙️ Setup

1. **Clone the repo**  
   ```bash
   git clone git@github.com:YOUR_USERNAME/emotion_detection_project.git
   cd emotion_detection_project

2. **Install Requirements**  
   ```bash
   pip install -r requirements.txt

3. **Kick off**  
   ```bash
   python scripts/tfidf_baseline.py --data_path data/emotion.csv --output_dir models/tfidf
