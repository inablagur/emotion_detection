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
emotion_detection/        
│
├── data/                             ← All raw & split datasets
│   ├── train.csv                     ← Training examples (text + emotion)
│   ├── validation.csv                ← Validation examples
│   └── test.csv                      ← Test examples
│
├── scripts/                          ← Helper scripts for data prep & model runs, executable entry points.
│   └── load_data.py                  ← Loads HF “emotion” dataset, writes train/val/test CSVs
│
├── notebooks/                        ← Jupyter notebooks, aligned with milestones
│   ├── 01_data_exploration.ipynb     ← Explore data: counts, lengths, quirks
│   ├── 02_tfidf_baseline.ipynb       ← Baseline models with TF–IDF + classifiers
│   ├── 03_transformer_finetune.ipynb ← Fine-tune BERT & DistilBERT
│   └── 04_compare_and_plot.ipynb     ← Compare metrics, times, and produce visuals
│
├── models/                           ← Saved model checkpoints (populates later)
│
├── README.md                         ← High-level overview, setup, and workflow
├── requirements.txt                  ← Pinned dependencies (Python 3.10)
└── .gitignore                        ← Files/folders to omit from Git

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

1. **Python & environment**  
   - Requires Python 3.8–3.10 (tested on 3.10)  
     
2. **Clone the repo**  
   ```bash
   git clone git@github.com:YOUR_USERNAME/emotion_detection.git
   cd emotion_detection

3. **Install Requirements**  
   ```bash
   pip install -r requirements.txt

4. **Prepare the data**  
   ```bash
   python scripts/load_data.py

5. **(To be added)**
    - Model training scripts (TF-IDF, transformer)
    - Evaluation & comparison tools
    - Demo deployment commands
---

## 📍 Roadmap & Status

Below is a high-level checklist of our project milestones.  
✔️ Completed ⚪️ Pending

- ✔️ **Project Kick-off**  
  - [x] Repo created  
  - [x] README & .gitignore added  
  - [x] Environment & deps installed

- ⚪️ **Data Gathering & Exploration**  
  - [x] Download & split dataset  
  - [x] `01_data_exploration.ipynb` completed

- ⚪️ **Data Cleaning & Preprocessing**  
  - [ ] Implement the modules individually and test them one by one
  - [ ] Integrate all individual modules into main clean_df function and test
  - [ ] Integrate into `scripts/2_clean_data.py`

- ⚪️ **Build Simple (Shallow) Models**  
  - [ ] `02_tfidf_baseline.ipynb`  
  - [ ] Evaluate & record metrics

- ⚪️ **Build Transformer Models**  
  - [ ] `03_transformer_finetune.ipynb`  
  - [ ] Track train time / memory

- ⚪️ **Compare & Visualize Results**  
  - [ ] `04_compare_and_plot.ipynb`

- ⚪️ **(Optional) Explainability**  
  - [ ] `05_explainability.ipynb`

- ⚪️ **Polish & Publish**  
  - [ ] Final README updates  
  - [ ] Demo deployment / release tag
