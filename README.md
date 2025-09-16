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
├── data/                                               ← All raw & split datasets
│   ├── original/                                       ← Raw dataset files
│   │   ├── train.csv                                   ← Training examples (text + emotion)                    (Output of `scripts/1_load_data.py`)
│   │   ├── validation.csv                              ← Validation examples                                   (Output of `scripts/1_load_data.py`)
│   │   └── test.csv                                    ← Test examples                                         (Output of `scripts/1_load_data.py`)
│   ├── clean/                                          ← Preprocessed dataset files
│   │   ├── train_clean.csv                             ← Preprocessed Training examples (text + emotion)       (Output of `scripts/2_clean_data.py`)
│   │   ├── validation_clean.csv                        ← Preprocessed Validation examples                      (Output of `scripts/2_clean_data.py`)
│   │   └── test_clean.csv                              ← Preprocessed Test examples                            (Output of `scripts/2_clean_data.py`)
│   └── labels/                                         ← Label mapping files
│       └── label2id.json                               ← Consistent class ordering mapping                     (anger:0, fear:1, joy:2, love:3, sadness:4, surprise:5)
│
├── scripts/                                            ← Helper scripts for data prep & model runs, executable entry points.
│   ├── 1_load_data.py                                  ← Loads HF "emotion" dataset, writes train/val/test CSVs
│   └── 2_clean_data.py                                 ← Preprocesses text for both shallow and transformer models
│
├── notebooks/                                          ← Jupyter notebooks, aligned with milestones
│   ├── 01_data_exploration.ipynb                       ← Explore data: counts, lengths, quirks
│   ├── 02_shallow_models_comparison.ipynb              ← Visualize & compare shallow models (metrics, confusion matrices, top terms)
│   ├── 03_transformer_finetune.ipynb                   ← Fine-tune transformer models (BERT, DistilBERT) on the dataset
│   ├── 03b_transformer_models_comparison.ipynb         ← Visualize & compare transformer models (metrics, confusion matrices, key examples)
│   └── 04_compare_shallow_vs_transformers.ipynb        ← Compare best shallow vs transformer models (performance, efficiency, trade-offs)
│                                                       
├── reports/                                            ← JSON reports and interpretation files for each trained model
│   └── transformers/                                   ← Transformer model reports and results
│
├── models/                                             ← Saved model checkpoints (populates later)
│   └── transformers/                                   ← Transformer model checkpoints and saved models
│
├── README.md                                           ← High-level overview, setup, and workflow
├── requirements.txt                                    ← Pinned dependencies (Python 3.10)
└── .gitignore                                          ← Files/folders to omit from Git

```
---

## 🔄 General Pipeline
(both) **Preprocessing**: lowercase, strip punctuation, expand contractions, pruning.

### 1️⃣ Shallow Baseline  
1. **Vectorization**: TF–IDF (`TfidfVectorizer(ngram_range=(1,2), max_features=50k)`)  
2. **Classification**:  
   - **lr - Logistic Regression (multinomial, L2)**  
     Strong on high-dimensional sparse text; probabilistic outputs; coefficients interpretable; fast; handles multiclass directly; supports class weighting.  
   - **lsvm - LinearSVC (one-vs-rest, squared hinge)**  
     Very strong baseline for text; efficient on sparse features; interpretable via linear weights; robust when `n_features > n_samples`.  
   - **cnb - Complement Naïve Bayes**  
     Probabilistic model tailored for text; uses complement statistics to stabilize rare features; extremely fast; robust to imbalance.  
  


### 2️⃣ Transformer Fine-Tuning  
1. **Tokenization**: `AutoTokenizer` (`bert-base-uncased`)  
2. **Model**: `AutoModelForSequenceClassification` (+ 6-label head)  
3. **Training**: Hugging Face `Trainer` API  

(both) **Evaluation**: train/dev/test split → accuracy, macro-F1, per-class F1

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

### Prerequisites
- **Python**: 3.8–3.10 (tested on 3.10)
- **Git**: For cloning the repository
- **CUDA** (optional): For GPU acceleration with transformers

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone git@github.com:<YOUR_USERNAME>/emotion_detection.git
   cd emotion_detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv emotion_env
   
   # Activate (Windows)
   emotion_env\Scripts\activate
   
   # Activate (Linux/Mac)
   source emotion_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

4. **Download and prepare the dataset**
   ```bash
   # Load the emotion dataset from Hugging Face
   python scripts/1_load_data.py
   
   # Clean and preprocess each data file individually
   python scripts/2_clean_data.py --input-csv data/original/train.csv --output-csv data/clean/train_clean.csv
   python scripts/2_clean_data.py --input-csv data/original/validation.csv --output-csv data/clean/validation_clean.csv
   python scripts/2_clean_data.py --input-csv data/original/test.csv --output-csv data/clean/test_clean.csv
   
   ```

5. **Verify data preparation**
   ```bash
   # Check that all data files exist
   ls data/clean/
   # Should show: test_clean.csv, train_clean.csv, validation_clean.csv
   ```

### Running the Project

6. **Start with data exploration**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

7. **Run shallow models comparison**
   ```bash
   jupyter notebook notebooks/02_shallow_models_comparison.ipynb
   ```

### Troubleshooting
- **CUDA issues**: If you encounter CUDA-related errors, the project will fall back to CPU training. CUDA is optional but recommended for faster transformer training
- **Memory issues**: For large models, consider reducing batch size in training scripts
- **Dataset download**: First run may take time to download the emotion dataset from Hugging Face
- **Data cleaning errors**: Ensure input CSV files exist in `data/original/` before running cleaning scripts
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
  - [x] Implement the modules individually and test them one by one
  - [x] Integrate all individual modules into main clean_df function and test
  - [x] Integrate into `scripts/2_clean_data.py`

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
  - [ ] Ensure persisnence between code and files
  - [ ] Final README updates  
  - [ ] Demo deployment / release tag