from youtube_fetcher import fetch_comments
from predictor import predict_batch, aggregate

# ── Chatbot State ─────────────────────────────────────────────────
state = {
    "last_keyword":  None,
    "last_results":  None,
    "last_summary":  None,
}


# ── Intent Detection ──────────────────────────────────────────────
def detect_intent(message):
    """Figure out what the user wants"""
    msg = message.lower().strip()

    # show specific sentiment
    if any(w in msg for w in ["negative", "negatif", "bura", "bekar", "complaints"]):
        return "show_negative"
    if any(w in msg for w in ["positive", "acha", "achi", "good", "best"]):
        return "show_positive"
    if any(w in msg for w in ["neutral", "theek"]):
        return "show_neutral"

    # analyze request
    if any(w in msg for w in ["analyze", "analyse", "check", "dekho", "batao", "karo", "sentiment"]):
        return "analyze"

    # help
    if any(w in msg for w in ["help", "madad", "kya kar", "how"]):
        return "help"

    # greeting
    if any(w in msg for w in ["hi", "hello", "salam", "assalam", "hey"]):
        return "greeting"

    # default — treat as brand/topic to analyze
    return "analyze"


def extract_keyword(message):
    """Extract brand or topic from message"""
    # remove common filler words
    fillers = [
        "analyze", "analyse", "check", "karo", "batao", "dekho",
        "sentiment", "of", "for", "about", "ka", "ki", "ke",
        "please", "plz", "kindly", "mujhe", "bata"
    ]
    words = message.strip().split()
    keyword_words = [w for w in words if w.lower() not in fillers]
    return " ".join(keyword_words).strip() if keyword_words else message.strip()


# ── Response Builders ─────────────────────────────────────────────
def build_analysis_response(keyword, summary):
    """Build the main analysis response"""
    dominant = summary["dominant"]
    pct      = summary["percentages"]
    counts   = summary["counts"]
    total    = summary["total"]

    # dominant mood message
    if dominant == "Positive":
        mood_msg = f"Loog **{keyword}** ke barey mein mostly positive hain! 🎉"
    elif dominant == "Negative":
        mood_msg = f"Loog **{keyword}** ke barey mein mostly negative hain. ⚠️"
    else:
        mood_msg = f"Loog **{keyword}** ke barey mein mixed feelings rakhte hain. 🤔"

    response = f"""
{mood_msg}

📊 **{total} Roman Urdu comments ka analysis:**

| Sentiment | Comments | Percentage |
|-----------|----------|------------|
| ✅ Positive | {counts['Positive']} | {pct['Positive']}% |
| ❌ Negative | {counts['Negative']} | {pct['Negative']}% |
| 😐 Neutral  | {counts['Neutral']}  | {pct['Neutral']}%  |

💬 **Sample negative comments:**
"""
    # add top negative comments
    neg_comments = summary["top"]["Negative"]
    if neg_comments:
        for i, c in enumerate(neg_comments[:3]):
            response += f"\n{i+1}. *\"{c['text'][:100]}\"* ({c['confidence']}% confident)"
    else:
        response += "\nKoi negative comments nahi mile."

    response += "\n\n💡 *Aur dekhne ke liye likhein: 'show positive' ya 'show negative'*"
    return response


def build_sentiment_filter_response(sentiment, summary, keyword):
    """Show filtered comments by sentiment"""
    comments = summary["top"][sentiment]
    emoji    = {"Positive": "✅", "Negative": "❌", "Neutral": "😐"}[sentiment]

    if not comments:
        return f"{emoji} **{keyword}** ke barey mein koi {sentiment.lower()} comments nahi mile."

    response = f"{emoji} **{keyword} — Top {sentiment} Comments:**\n\n"
    for i, c in enumerate(comments):
        response += f"{i+1}. *\"{c['text'][:120]}\"*\n   Confidence: {c['confidence']}%\n\n"

    return response


# ── Main Chat Function ────────────────────────────────────────────
def chat(message):
    """
    Main chatbot function — takes user message, returns response

    Args:
        message: user input string

    Returns:
        response string (markdown formatted)
    """
    message = message.strip()
    if not message:
        return "Kuch likhein — koi brand ya topic ka naam, jaise 'Daraz' ya 'Jazz network'"

    intent = detect_intent(message)

    # ── Greeting ──────────────────────────────────────────────────
    if intent == "greeting":
        return """👋 **Salam! Main Roman Urdu Sentiment Bot hun.**

Main kisi bhi Pakistani brand ya topic ke barey mein YouTube comments analyze kar sakta hun.

**Kaise use karein:**
- Brand ka naam likhein: `Daraz`
- Ya likhein: `analyze Jazz`
- Results ke baad: `show negative` ya `show positive`

Shuru karein — koi brand ka naam likhein! 🚀"""

    # ── Help ──────────────────────────────────────────────────────
    if intent == "help":
        return """🆘 **Help:**

**Commands:**
- `[brand name]` — Us brand ke YouTube comments analyze karo
- `analyze [topic]` — Kisi bhi topic ka sentiment dekho
- `show positive` — Sirf positive comments dekho
- `show negative` — Sirf negative comments dekho
- `show neutral` — Sirf neutral comments dekho

**Examples:**
- `Daraz`
- `analyze Telenor`
- `Jazz network Pakistan`
- `show negative`"""

    # ── Show filtered sentiment ───────────────────────────────────
    if intent in ["show_negative", "show_positive", "show_neutral"]:
        if not state["last_summary"]:
            return "⚠️ Pehle koi brand analyze karein. Likhein: `Daraz` ya `analyze Jazz`"

        sentiment_map = {
            "show_negative": "Negative",
            "show_positive": "Positive",
            "show_neutral":  "Neutral"
        }
        sentiment = sentiment_map[intent]
        return build_sentiment_filter_response(sentiment, state["last_summary"], state["last_keyword"])

    # ── Analyze brand/topic ───────────────────────────────────────
    if intent == "analyze":
        keyword = extract_keyword(message)

        if not keyword or len(keyword) < 2:
            return "⚠️ Brand ya topic ka naam likhein. Jaise: `Daraz` ya `Jazz`"

        # fetch comments
        thinking_msg = f"🔍 **'{keyword}' ke barey mein YouTube comments fetch ho rahe hain...**\n\n*(Thora wait karein — 30-60 seconds)*"
        print(f"[BOT] Fetching comments for: {keyword}")

        try:
            comments = fetch_comments(keyword, max_comments=300)
        except Exception as e:
            return f"⚠️ Comments fetch nahi ho sake: {e}"

        if not comments:
            return f"⚠️ **'{keyword}'** ke barey mein koi Roman Urdu comments nahi mile. Koi aur keyword try karein."

        # run sentiment analysis
        print(f"[BOT] Analyzing {len(comments)} comments...")
        results = predict_batch(comments)
        summary = aggregate(results)

        # save to state
        state["last_keyword"] = keyword
        state["last_results"] = results
        state["last_summary"] = summary

        return build_analysis_response(keyword, summary)

    return "Samajh nahi aya. Koi brand ka naam likhein jaise `Daraz` ya `Jazz`"


# ── Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Roman Urdu Sentiment Chatbot — Test Mode")
    print("=" * 55)
    print("Type a brand name or 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue
        response = chat(user_input)
        print(f"\nBot: {response}\n")
        print("-" * 55)