# Roman Urdu Sentiment Bot

[![Streamlit App](https://shields.io)](https://streamlit.app)

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-yellow)](https://huggingface.co/spaces/Hifsa65/roman-urdu-sentiment-bot)
[![Model v2](https://img.shields.io/badge/Model-v2_74%25_accuracy-green)](https://huggingface.co/Hifsa65/roman-urdu-sentiment-v2)
[![GitHub](https://img.shields.io/badge/GitHub-hifsaiftikhar-black)](https://github.com/hifsaiftikhar)

A deep learning chatbot that analyzes real-time YouTube comments in Roman Urdu
to understand public sentiment about Pakistani brands and topics.

---

## Overview

Most sentiment analysis tools are built for English. Roman Urdu — the way Pakistanis
actually write online — is almost entirely unsupported. This project fills that gap.

- Fetches live YouTube comments for any Pakistani brand or topic
- Filters and processes Roman Urdu text specifically
- Classifies sentiment as Positive, Negative, or Neutral using fine-tuned DistilBERT
- Presents results through an interactive chatbot interface
- Visualizes data with pie charts, bar charts, confidence histograms, and word clouds
- Supports CSV export of full analysis

---

## Model Versions

| Version | Training Samples | Test Accuracy | Notes |
|---|---|---|---|
| v1 | 19,626 | 63% | Baseline, imbalanced dataset |
| v2 | 24,989 | 73% | Augmented + balanced, +10% accuracy |

---

## Project Structure

```
roman-urdu-sentiment-bot/
│
├── app.py                  ← Streamlit UI with 4 color themes
├── chatbot.py              ← Intent detection and response logic
├── predictor.py            ← Model loading and sentiment inference
├── youtube_fetcher.py      ← YouTube comment scraping pipeline
├── train_model.py          ← Model training and evaluation code
├── augment_dataset.py      ← Dataset augmentation script
├── models/
│   ├── v1/README.md        ← v1 training details and results
│   └── v2/README.md        ← v2 training details and results
├── .gitignore
└── README.md
```

Note: `saved_model/` (fine-tuned weights) and `data/` (training dataset) are excluded
from this repo due to file size. See Setup section for instructions.

---

## Tech Stack

| Component | Tool |
|---|---|
| Language Model | DistilBERT (distilbert-base-multilingual-cased) |
| Framework | PyTorch + HuggingFace Transformers |
| UI | Streamlit |
| Data Fetching | scrapetube + youtube-comment-downloader |
| Visualization | Plotly + Matplotlib + WordCloud |
| Language | Python 3.10+ |

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/hifsaiftikhar/roman-urdu-sentiment-bot.git
cd roman-urdu-sentiment-bot
```

**2. Install dependencies**
```bash
pip install streamlit transformers torch plotly wordcloud matplotlib pandas numpy scikit-learn scrapetube youtube-comment-downloader
```

**3. Add your trained model**

Place your fine-tuned DistilBERT model files in a `saved_model/` folder:
```
saved_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## Live Demo

**Live App:** [huggingface.co/spaces/Hifsa65/roman-urdu-sentiment-bot](https://huggingface.co/spaces/Hifsa65/roman-urdu-sentiment-bot)
> Currently running v2 (74% accuracy).

Type any Pakistani brand or topic into the chatbot:

```
Daraz       Jazz        Telenor
Foodpanda   Easypaisa   PTI
PMLN        PIA         Bykea
```

Commands supported:
```
show positive     show negative     show neutral     help
```

---

## Model Performance

Current model (v2):

| Metric | Score |
|---|---|
| Model | DistilBERT multilingual |
| Training Samples | 19,991 |
| Validation Accuracy | 74.07% |
| Test Accuracy | 73% |

### Per Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Positive | 0.75 | 0.77 | 0.76 |
| Negative | 0.75 | 0.73 | 0.74 |
| Neutral | 0.69 | 0.69 | 0.69 |

### Why not higher?

Roman Urdu NLP has unique challenges that limit achievable accuracy:

- Inconsistent spelling — the same word appears in 5+ different forms (e.g. nahi / nahin / nai / ni)
- Code-mixing — Pakistanis mix Roman Urdu and English in the same sentence
- No dedicated pretrained model — DistilBERT was not trained on Roman Urdu
- Label noise — some samples in the dataset are genuinely ambiguous

These are documented limitations, not bugs. 73% on Roman Urdu is strong given
the lack of dedicated pretrained models for this language variety.

---

## Features

- Real-time YouTube comment fetching by keyword
- Roman Urdu language filtering with custom word lists
- Profanity and spam filtering
- 24-hour comment caching for faster repeat searches
- Sentiment breakdown with confidence scores per comment
- Interactive pie chart, bar chart, confidence histogram
- Word cloud visualization per sentiment class
- Top comments filtered by sentiment
- Analysis history tracking across brands
- CSV export of full analysis results
- 4 switchable color themes — Purple, Cyan, Green, Amber
- Conversational chatbot interface with intent detection

---

## Deep Learning Concepts Used

| Concept | Application |
|---|---|
| Transformer Architecture | DistilBERT model backbone |
| Multi-Head Self-Attention | Understanding word context |
| Transfer Learning | Pretrained DistilBERT fine-tuned on Roman Urdu |
| Embeddings | Text to 768-dimensional vectors |
| Softmax Activation | 3-class probability output |
| Cross Entropy Loss | Training loss function |
| Backpropagation | Model weight updates |
| Dropout Regularization | Prevent overfitting |
| AdamW Optimizer | Gradient-based optimization |
| Batch Processing | Efficient GPU/CPU inference |

---

## Dataset

- Name: Roman Urdu Sentiment Dataset
- Size: 24,989 samples (augmented from original 19,626)
- Classes: Positive, Negative, Neutral (~8,500 each)
- Source: Pakistani social media text

Preprocessing steps applied:
- Fixed typo label ("Neative" → "Negative")
- Removed null values and duplicates
- Applied stratified 80/10/10 train/val/test split
- Augmented and balanced classes to address class imbalance

---

## Future Work

- Expand dataset with more diverse Roman Urdu sources
- Add brand comparison feature (side-by-side analysis)

## Experiments

| Approach | Result |
|---|---|
| XLM-RoBERTa fine-tuning | Underperformed DistilBERT on this dataset |
| Dataset augmentation + balancing | +10% accuracy over baseline (v1 → v2) |

---

## Author

**Hifsa Iftikhar**  
GitHub: [@hifsaiftikhar](https://github.com/hifsaiftikhar)
