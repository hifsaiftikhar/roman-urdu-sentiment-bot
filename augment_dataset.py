import pandas as pd
import random
import re
import os

# ── Config ────────────────────────────────────────────────────────
INPUT_PATH  = "data/Roman Urdu DataSet.csv"
OUTPUT_PATH = "data/augmented_dataset.csv"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Spelling Variants Dictionary ──────────────────────────────────
# Roman Urdu has no fixed spelling — same word written many ways
# We use this to generate realistic alternate versions of each sentence
SPELLING_VARIANTS = {
    # negation
    "nahi":     ["nahin", "nai", "ni", "nhe"],
    "nahin":    ["nahi", "nai", "ni"],
    "nai":      ["nahi", "nahin", "ni"],

    # is/are
    "hai":      ["ha", "hy", "hei"],
    "hain":     ["hain", "han", "hn"],
    "ha":       ["hai", "hy"],

    # very/much
    "bohat":    ["bahut", "bhot", "bht"],
    "bahut":    ["bohat", "bhot", "bht"],
    "bhot":     ["bohat", "bahut"],

    # good
    "acha":     ["achha", "accha", "acha"],
    "achha":    ["acha", "accha"],
    "accha":    ["acha", "achha"],
    "achi":     ["achhi", "acchi"],
    "achhi":    ["achi", "acchi"],

    # bad
    "bura":     ["bura", "bra", "burra"],
    "bekar":    ["be kar", "beakr", "bekaar"],

    # also/too
    "bhi":      ["b", "bi"],

    # then/so
    "toh":      ["to", "tou"],
    "to":       ["toh", "tou"],
    "tou":      ["toh", "to"],

    # brother/friend (common filler)
    "bhai":     ["bhi", "bhaee", "bhaai"],
    "yaar":     ["yar", "yr"],

    # this/that
    "yeh":      ["ye", "yh"],
    "ye":       ["yeh", "yh"],
    "woh":      ["wo", "wh"],
    "wo":       ["woh", "wh"],

    # you (formal)
    "aap":      ["ap", "aap"],
    "ap":       ["aap"],

    # me/I
    "main":     ["mein", "mn", "me"],
    "mein":     ["main", "mn"],

    # again/more
    "phir":     ["pher", "fir", "fer"],
    "pher":     ["phir", "fir"],

    # but
    "lekin":    ["lkn", "lakin", "lekin"],
    "magar":    ["mgr", "mger"],

    # why
    "kyun":     ["kyunke", "kyn", "kiun"],

    # okay/fine
    "theek":    ["thik", "theekh", "thk"],
    "thik":     ["theek", "theekh"],

    # very good
    "zabardast":["zabrdast", "zbrdst", "zabar dast"],

    # nonsense
    "bakwas":   ["bakwaas", "bakwass", "bkws"],

    # how much/many
    "kitna":    ["kitna", "ktna", "kitna sa"],
    "kitni":    ["kitni", "ktni"],

    # come/came
    "aaya":     ["aya", "aaia"],
    "aya":      ["aaya", "aaia"],
    "aayi":     ["aayi", "ayi"],
    "ayi":      ["aayi", "aayi"],

    # went/gone
    "gaya":     ["gya", "gia"],
    "gyi":      ["gayi", "gai"],
    "gayi":     ["gyi", "gai"],

    # will be
    "hoga":     ["hga", "hoega"],
    "hogi":     ["hgi", "hoegi"],

    # do/did
    "karo":     ["kro", "karo"],
    "kiya":     ["kia", "kya"],
    "kia":      ["kiya", "kya"],

    # see/look
    "dekho":    ["dkho", "dekhoo"],
    "dekha":    ["dkha", "dekha"],

    # more
    "zyada":    ["zada", "zyda", "ziada"],
    "zada":     ["zyada", "zyda"],

    # a little
    "thoda":    ["thora", "thda", "thora sa"],
    "thora":    ["thoda", "thda"],

    # only
    "sirf":     ["srf", "sirf hi"],
    "bas":      ["bs", "bas hi"],

    # if
    "agar":     ["agr", "agar ke"],

    # confirmed/sure
    "pakka":    ["paka", "pkka", "pakka se"],

    # anyway
    "waise":    ["wesy", "wese", "waisy"],

    # I think/seems
    "lagta":    ["lgta", "lagtha"],
    "lagti":    ["lgti", "lagthi"],

    # understand
    "samjhe":   ["smjhe", "samjha"],

    # absolutely
    "bilkul":   ["bilkl", "bilkul se", "blkl"],
    "ekdum":    ["ekdm", "ekdum se"],

    # today/yesterday/now
    "aaj":      ["aj", "aaj kal"],
    "kal":      ["kl", "kal ka"],
    "abhi":     ["abhe", "abhi tak"],

    # always
    "hamesha":  ["hmsha", "hamesha se"],

    # network/service related
    "service":  ["srvice", "servis"],
    "network":  ["ntwrk", "netwrk"],
    "delivery": ["dlvry", "delivry"],
    "problem":  ["problm", "probem", "msla"],
    "account":  ["accnt", "acount"],
}


def augment_sentence(text):
    """
    Generate one augmented version of a sentence by replacing
    Roman Urdu words with their spelling variants.
    Returns None if no replaceable word found.
    """
    words = text.split()
    new_words = words.copy()
    changed = False

    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in SPELLING_VARIANTS:
            variants = SPELLING_VARIANTS[word_lower]
            replacement = random.choice(variants)
            # preserve original capitalization style
            if word[0].isupper():
                replacement = replacement.capitalize()
            new_words[i] = replacement
            changed = True

    if not changed:
        return None

    new_text = " ".join(new_words)

    # make sure it's actually different
    if new_text.lower() == text.lower():
        return None

    return new_text


def augment_class(df, label, target_size):
    """
    Augment a class to reach target_size samples.
    """
    class_df   = df[df["label"] == label].copy()
    current    = len(class_df)
    needed     = target_size - current

    if needed <= 0:
        print(f"  {label}: already at {current} — no augmentation needed")
        return pd.DataFrame()

    print(f"  {label}: {current} → {target_size} (adding {needed} samples)")

    augmented_rows = []
    attempts       = 0
    max_attempts   = needed * 10  # try hard but don't loop forever

    texts = class_df["text"].tolist()

    while len(augmented_rows) < needed and attempts < max_attempts:
        # pick a random original sample
        original_text = random.choice(texts)
        augmented     = augment_sentence(original_text)

        if augmented is not None:
            augmented_rows.append({
                "text":  augmented,
                "label": label
            })

        attempts += 1

    print(f"    Generated {len(augmented_rows)} augmented samples ({attempts} attempts)")
    return pd.DataFrame(augmented_rows)


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("ROMAN URDU DATASET AUGMENTATION")
    print("=" * 60)

    # ── Load original dataset ─────────────────────────────────────
    print("\nLoading original dataset...")
    df = pd.read_csv(INPUT_PATH, header=None, names=["text", "label", "extra"])
    df = df[["text", "label"]]
    df["label"] = df["label"].str.strip().replace("Neative", "Negative")
    df = df.dropna()
    df = df[df["label"].isin(["Positive", "Negative", "Neutral"])]
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset="text")
    df = df.reset_index(drop=True)

    print(f"\nOriginal distribution:")
    print(df["label"].value_counts())
    print(f"Total: {len(df)}")

    # ── Target size = size of largest class (Neutral) ─────────────
    target_size = df["label"].value_counts().max()
    print(f"\nTarget size per class: {target_size}")

    # ── Augment Positive and Negative only ────────────────────────
    print("\nAugmenting minority classes...")
    pos_augmented = augment_class(df, "Positive", target_size)
    neg_augmented = augment_class(df, "Negative", target_size)

    # ── Combine ───────────────────────────────────────────────────
    combined = pd.concat([df, pos_augmented, neg_augmented], ignore_index=True)

    # remove any duplicates that slipped through
    combined = combined.drop_duplicates(subset="text")

    # shuffle
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ── Results ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nFinal distribution:")
    print(combined["label"].value_counts())
    print(f"\nTotal samples: {len(combined)}")
    print(f"Added: {len(combined) - len(df)} new samples")

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    combined[["text", "label"]].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print("\nNext step: update DATA_PATH in train_model.py to use augmented_dataset.csv")