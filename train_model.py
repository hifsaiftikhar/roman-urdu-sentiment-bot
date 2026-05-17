import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import json

# ── Config ────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
SAVE_PATH  = "saved_model"
DATA_PATH  = "data/Roman Urdu DataSet.csv"

LABEL_MAP = {"Positive": 0, "Negative": 1, "Neutral": 2}
ID_TO_LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ── Step 1: Load and Clean Dataset ───────────────────────────────
print("=" * 50)
print("STEP 1: Loading dataset...")
print("=" * 50)

df = pd.read_csv(DATA_PATH, header=None, names=["text", "label", "extra"])

# drop extra column
df = df[["text", "label"]]

# fix typo label
df["label"] = df["label"].str.strip()
df["label"] = df["label"].replace("Neative", "Negative")

# drop nulls and unknowns
df = df.dropna()
df = df[df["label"].isin(["Positive", "Negative", "Neutral"])]
df = df[df["text"].str.strip() != ""]

# remove duplicates
df = df.drop_duplicates(subset="text")

# encode labels
df["label_id"] = df["label"].map(LABEL_MAP)

print(f"Total samples after cleaning: {len(df)}")
print(f"\nLabel distribution:")
print(df["label"].value_counts())

# ── Step 2: Split Data ────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Splitting data...")
print("=" * 50)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label_id"])

print(f"Train size : {len(train_df)}")
print(f"Val size   : {len(val_df)}")
print(f"Test size  : {len(test_df)}")

# ── Step 3: Tokenizer ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Loading tokenizer...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded.")

# ── Step 4: Dataset Class ─────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = SentimentDataset(train_df["text"], train_df["label_id"], tokenizer, MAX_LEN)
val_dataset   = SentimentDataset(val_df["text"],   val_df["label_id"],   tokenizer, MAX_LEN)
test_dataset  = SentimentDataset(test_df["text"],  test_df["label_id"],  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

print(f"Datasets created.")

# ── Step 5: Load Model ────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Loading model...")
print("=" * 50)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=LR)
print("Model loaded and ready.\n")

# ── Step 6: Train ─────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss_sum += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / len(loader), correct / total


print("=" * 50)
print("STEP 5: Training...")
print("=" * 50)

best_val_acc = 0
history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for i, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_acc  = correct / total
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Acc : {train_acc:.4f}")
    print(f"  Val Acc   : {val_acc:.4f}")
    print(f"  Val Loss  : {val_loss:.4f}\n")

    history.append({
        "epoch": epoch + 1,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_loss": val_loss
    })

    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(SAVE_PATH, exist_ok=True)
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"  ✓ Best model saved (val_acc={val_acc:.4f})\n")

# ── Step 7: Test Evaluation ───────────────────────────────────────
print("=" * 50)
print("STEP 6: Final evaluation on test set...")
print("=" * 50)

model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=["Positive", "Negative", "Neutral"]
))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# save training history
with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nBest Validation Accuracy : {best_val_acc:.4f}")
print(f"Model saved to           : {SAVE_PATH}/")
print(f"Training history saved   : training_history.json")
print("\nTraining complete!")