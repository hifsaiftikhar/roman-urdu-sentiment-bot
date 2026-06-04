import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH = "Hifsa65/roman-urdu-sentiment-distilbert"
MAX_LEN     = 128
ID_TO_LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}
EMOJI       = {"Positive": "✅", "Negative": "❌", "Neutral": "😐"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Model Once ───────────────────────────────────────────────
print("Loading sentiment model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model     = model.to(device)
model.eval()
print("Model ready.\n")


# ── Single Text Prediction ────────────────────────────────────────
def predict(text):
    """
    Predict sentiment for a single text

    Returns:
        dict with label, confidence, and all scores
    """
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=1).squeeze()

    scores = {
        "Positive": round(probs[0].item() * 100, 1),
        "Negative": round(probs[1].item() * 100, 1),
        "Neutral":  round(probs[2].item() * 100, 1),
    }

    label      = max(scores, key=scores.get)
    confidence = scores[label]

    return {
        "label":      label,
        "emoji":      EMOJI[label],
        "confidence": confidence,
        "scores":     scores
    }


# ── Batch Prediction ──────────────────────────────────────────────
def predict_batch(texts):
    """
    Predict sentiment for a list of texts

    Returns:
        list of prediction dicts
    """
    results = []
    for text in texts:
        try:
            result = predict(text)
            result["text"] = text
            results.append(result)
        except Exception as e:
            results.append({
                "text":       text,
                "label":      "Neutral",
                "emoji":      "😐",
                "confidence": 0.0,
                "scores":     {"Positive": 0, "Negative": 0, "Neutral": 100}
            })
    return results


# ── Aggregate Results ─────────────────────────────────────────────
def aggregate(results):
    """
    Summarize sentiment across multiple predictions

    Returns:
        dict with counts, percentages, top comments
    """
    total = len(results)
    if total == 0:
        return None

    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in results:
        counts[r["label"]] += 1

    percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}

    # top 3 most confident per sentiment
    top = {}
    for sentiment in ["Positive", "Negative", "Neutral"]:
        filtered = [r for r in results if r["label"] == sentiment]
        filtered.sort(key=lambda x: x["confidence"], reverse=True)
        top[sentiment] = filtered[:3]

    return {
        "total":       total,
        "counts":      counts,
        "percentages": percentages,
        "top":         top,
        "dominant":    max(percentages, key=percentages.get)
    }


# ── Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "Daraz ki delivery bohat achi hai, mujhe bohot pasand aya",
        "Ye bilkul bekar service hai, order aaya hi nahi",
        "Theek hai, kuch zyada acha nahi kuch zyada bura nahi",
        "Zabardast experience tha, zaroor dobara order karunga",
        "Bahut bura hua mera paise gaye aur product nahi mila"
    ]

    print("Testing individual predictions:")
    print("-" * 50)
    for text in test_texts:
        result = predict(text)
        print(f"Text       : {text}")
        print(f"Sentiment  : {result['emoji']} {result['label']} ({result['confidence']}%)")
        print(f"All scores : {result['scores']}")
        print()

    print("\nTesting batch + aggregation:")
    print("-" * 50)
    results = predict_batch(test_texts)
    summary = aggregate(results)
    print(f"Total analyzed : {summary['total']}")
    print(f"Positive       : {summary['counts']['Positive']} ({summary['percentages']['Positive']}%)")
    print(f"Negative       : {summary['counts']['Negative']} ({summary['percentages']['Negative']}%)")
    print(f"Neutral        : {summary['counts']['Neutral']} ({summary['percentages']['Neutral']}%)")
    print(f"Dominant mood  : {summary['dominant']}")
