# Roman Urdu Sentiment Bot

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

## Project Structure

```
roman-urdu-sentiment-bot/
│
├── app.py                  ← Streamlit UI with 4 color themes
├── chatbot.py              ← Intent detection and response logic
├── predictor.py            ← Model loading and sentiment inference
├── youtube_fetcher.py      ← YouTube comment scraping pipeline
├── train_model.py          ← Model training and evaluation code
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

| | Score |
|---|---|
| Model | DistilBERT multilingual |
| Training samples | 15,700 |
| Validation samples | 1,963 |
| Test samples | 1,963 |
| Validation Accuracy | 65% |
| Test Accuracy | 63% |

### Why 63%?

Roman Urdu NLP has unique challenges that limit achievable accuracy:

- Inconsistent spelling — the same word appears in 5+ different forms (e.g. nahi / nahin / nai / ni)
- Code-mixing — Pakistanis mix Roman Urdu and English in the same sentence
- No dedicated pretrained model — DistilBERT was not trained on Roman Urdu
- Label noise — some samples in the dataset are genuinely ambiguous

These are documented limitations, not bugs. 63% on Roman Urdu is comparable to
what other published work achieves on this language variety.

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
- Size: 20,229 samples (after cleaning: 19,626)
- Classes: Positive, Negative, Neutral
- Source: Pakistani social media text

Preprocessing steps applied:
- Fixed typo label ("Neative" → "Negative")
- Removed null values and duplicates
- Applied stratified 80/10/10 train/val/test split

---

## Future Work

- Retrain with XLM-RoBERTa for better multilingual support
- Add class weights to handle Neutral class dominance
- Expand dataset with more diverse Roman Urdu sources
- Deploy on Hugging Face Spaces for public access
- Add brand comparison feature (side-by-side analysis)

---

## Author

**Hifsa Iftikhar**
GitHub: [@hifsaiftikhar](https://github.com/hifsaiftikhar)
