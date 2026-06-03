# Two NLP Projects in Google Colab

Two independent BERT-based deep learning projects for sentiment analysis and question answering.

## 📍 Project 1: GitHub Sentiment Analysis

**File**: `GitHub_Sentiment_Analysis.ipynb`

- Sentiment classification for PR comments (Positive/Negative)
- Scrapes comments from VSCode, React, TensorFlow repos
- Fine-tunes BERT on SST2 dataset
- **Results**: 93.12% Accuracy | 93.12% F1 Score

**Output**: Predictions saved to Drive with confidence scores

---

## 📍 Project 2: Extractive Question Answering (SQuAD v2.0)

**File**: `Extractive_QA_SQuAD.ipynb`

- Extracts answers from paragraphs
- Handles both answerable & unanswerable questions
- Fine-tunes BERT on SQuAD v2.0 dataset (130k training samples)
- **Results**: 45.10% Exact Match | 48.51% F1 Score

**Output**: Inference results saved to Drive

---

## 🚀 Quick Start (Both Projects)

1. Open either notebook in Google Colab
2. Run cells in order (1-16+)
3. Cell 2 mounts Drive & creates project folders
4. Data/models saved to Drive (persist between sessions)

---

## ⚙️ Cell Execution Order

**First Run**: Cells 1 → 2 → 3 → ... → last cell

**After Runtime Restart**: Cells 1 → 2 → 3 → 10-12 (data auto-skips)

Early cells auto-skip if data already exists on Drive.

---

## 📁 Google Drive Structure

Each project creates its own folder:
```
GitHub_Sentiment_Analysis/
  ├── data/ (CSV, tokenized datasets)
  ├── model/ (best_model checkpoint)
  └── results/ (plots, JSON, predictions)

Extractive_QA_SQuAD/
  ├── data/ (SQuAD dataset, tokenized)
  ├── model/ (best_model checkpoint)
  └── results/ (evaluation JSON, inference)
```

---

## 💻 Requirements

```
transformers, datasets, torch, requests, pandas, scikit-learn, matplotlib
```

Installed automatically in Cell 1 of each notebook.

---

## ⏱️ Runtime

- **Project 1** - First: 10-15 min | Restart: 2-3 min
- **Project 2** - First: 15-20 min | Restart: 4-5 min

First runs include data scraping/downloading & model training.
