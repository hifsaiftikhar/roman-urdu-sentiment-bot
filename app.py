import streamlit as st
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import chat, state
from predictor import predict_batch, aggregate
from youtube_fetcher import fetch_comments

st.set_page_config(
    page_title="Urdu Sentiment",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after {
    font-family: 'Space Grotesk', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background: #f5f0eb;
    color: #1a1a1a;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── NAV ── */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.2rem 2.5rem;
    border-bottom: 1px solid #e0d9d0;
    background: #f5f0eb;
}
.nav-logo {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #1a1a1a;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.nav-tag {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.nav-dot {
    width: 7px;
    height: 7px;
    background: #2d7a4f;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── HERO ── */
.hero-wrap {
    padding: 5rem 2.5rem 4rem 2.5rem;
    background: #f5f0eb;
    border-bottom: 1px solid #e0d9d0;
}
.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #2d7a4f;
    text-transform: uppercase;
    letter-spacing: 4px;
    margin-bottom: 1.2rem;
    opacity: 0;
    animation: fadeUp 0.6s ease 0.1s forwards;
}
.hero-title {
    font-size: clamp(2.5rem, 5vw, 4.5rem);
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 1.4rem;
    max-width: 700px;
    opacity: 0;
    animation: fadeUp 0.6s ease 0.25s forwards;
}
.hero-title span {
    color: #2d7a4f;
    border-bottom: 3px solid #2d7a4f;
    padding-bottom: 2px;
}
.hero-desc {
    font-size: 1rem;
    color: #666;
    max-width: 500px;
    line-height: 1.7;
    opacity: 0;
    animation: fadeUp 0.6s ease 0.4s forwards;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── STATS BAR ── */
.stats-bar {
    display: flex;
    background: #1a1a1a;
    border-bottom: 1px solid #2a2a2a;
}
.stat-item {
    flex: 1;
    padding: 1.6rem 2rem;
    border-right: 1px solid #2a2a2a;
    opacity: 0;
    animation: fadeUp 0.5s ease forwards;
}
.stat-item:nth-child(1) { animation-delay: 0.1s; }
.stat-item:nth-child(2) { animation-delay: 0.2s; }
.stat-item:nth-child(3) { animation-delay: 0.3s; }
.stat-item:nth-child(4) { animation-delay: 0.4s; }
.stat-item:last-child { border-right: none; }
.stat-num {
    font-size: 1.8rem;
    font-weight: 700;
    color: #c8f04a;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── MARQUEE ── */
.marquee-wrap {
    overflow: hidden;
    background: #eee8e0;
    border-bottom: 1px solid #e0d9d0;
    padding: 0.8rem 0;
    white-space: nowrap;
}
.marquee-track {
    display: inline-block;
    animation: marquee 35s linear infinite;
}
.marquee-track:hover { animation-play-state: paused; }
.marquee-item {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 140px;
    height: 70px;
    padding: 0.6rem 1.4rem;
    margin: 0 0.8rem;
    border: 1px solid #ccc;
    border-radius: 10px;
    background: #fff;
    transition: border-color 0.2s, transform 0.2s;
}
.marquee-item:hover {
    border-color: #2d7a4f;
    transform: translateY(-2px);
}
.marquee-item img {
    max-height: 36px;
    max-width: 110px;
    width: auto;
    height: auto;
    object-fit: contain;
}

/* ── DISCLAIMER ── */
.disclaimer {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #444;
    text-align: center;
    line-height: 1.6;
    padding: 1.5rem 2.5rem 2.5rem 2.5rem;
    border-top: 1px solid #e0d9d0;
    max-width: 800px;
    margin: 2rem auto 0 auto;
}
@keyframes marquee {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ── MAIN CONTENT ── */
.main-content {
    padding: 2.5rem 2.5rem 4rem 2.5rem;
    max-width: 1400px;
    margin: 0 auto;
    background: #f5f0eb;
}

/* ── SEARCH ── */
.search-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #1a1a1a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
    margin-top: 2rem;
}
.stTextInput input {
    background: #fff !important;
    border: 1.5px solid #888 !important;
    border-radius: 4px !important;
    color: #1a1a1a !important;
    font-size: 1rem !important;
    padding: 1rem 1.2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: border-color 0.2s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
.stTextInput input:focus {
    border-color: #2d7a4f !important;
    box-shadow: 0 0 0 2px rgba(45,122,79,0.15) !important;
}
.stTextInput input::placeholder { color: #888 !important; }
.stTextInput label { display: none !important; }

/* ── ANALYZE BUTTON ── */
[data-testid="stFormSubmitButton"] button {
    background: #2d7a4f !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: background 0.2s !important;
    cursor: pointer !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #235f3d !important;
}

/* ── BRAND TAG BUTTONS ── */
[data-testid="stHorizontalBlock"] .stButton button {
    background: #fff !important;
    border: 1px solid #999 !important;
    color: #444 !important;
    border-radius: 100px !important;
    font-size: 0.72rem !important;
    padding: 0.15rem 0.6rem !important;
    margin: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 400 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    width: auto !important;
    min-height: unset !important;
    line-height: 1.4 !important;
    transition: all 0.2s !important;
}
[data-testid="stHorizontalBlock"] .stButton button:hover {
    border-color: #2d7a4f !important;
    color: #2d7a4f !important;
    background: #fff !important;
}
[data-testid="stHorizontalBlock"] { gap: 0.2rem !important; }

/* ── RESULTS HEADER ── */
.results-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e0d9d0;
}
.results-keyword {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -0.5px;
}
.results-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── METRIC GRID ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #e0d9d0;
    border: 1px solid #e0d9d0;
    margin-bottom: 3rem;
    border-radius: 6px;
    overflow: hidden;
}
.metric-cell {
    background: #fff;
    padding: 2rem 1.5rem;
}
.metric-pct {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
    letter-spacing: -2px;
}
.metric-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #666;
}
.metric-count { font-size: 0.8rem; color: #888; margin-top: 0.3rem; }
.m-pos .metric-pct { color: #2d7a4f; }
.m-neg .metric-pct { color: #e05555; }
.m-neu .metric-pct { color: #666; }

/* ── SECTION HEADER ── */
.s-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 3px;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #e0d9d0;
    margin-bottom: 1.5rem;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #e0d9d0 !important;
    gap: 0 !important; padding: 0 !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #aaa !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1.2rem !important;
    border-radius: 0 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Space Mono', monospace !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #2d7a4f !important;
    border-bottom: 2px solid #2d7a4f !important;
}

/* ── CHAT ── */
.stChatMessage {
    background: #fff !important;
    border: 1px solid #e0d9d0 !important;
    border-radius: 4px !important;
}
[data-testid="stChatMessageContent"] p {
    color: #444 !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #e0d9d0 !important;
    border-radius: 4px !important;
    background: #fff !important;
}

/* ── COMMENT CARD ── */
.c-card {
    padding: 1rem 1.2rem;
    border-left: 3px solid #2d7a4f;
    background: #fff;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: #555;
    line-height: 1.6;
    border-radius: 0 4px 4px 0;
}
.c-card.neg { border-left-color: #e05555; }
.c-card.neu { border-left-color: #777; }
.c-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #888;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── HISTORY ── */
.h-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem 0;
    border-bottom: 1px solid #e0d9d0;
    font-size: 0.85rem;
}
.h-kw { color: #333; }
.h-dom { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #666; }

/* ── EMPTY ── */
.empty {
    padding: 5rem 2rem;
    text-align: center;
    border: 1px dashed #ddd;
    border-radius: 6px;
    margin-top: 2rem;
    background: #fff;
}
.empty-title {
    font-size: 0.85rem;
    color: #555;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: #fff !important;
    border: 1px solid #ddd !important;
    color: #1a1a1a !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
}

/* ── RADIO ── */
.stRadio label { color: #888 !important; font-size: 0.82rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #888 !important; }

/* ── DOWNLOAD ── */
.stDownloadButton button {
    background: transparent !important;
    border: 1px solid #ddd !important;
    color: #999 !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
}
.stDownloadButton button:hover {
    border-color: #2d7a4f !important;
    color: #2d7a4f !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
for key, val in [
    ("messages", []),
    ("analysis_history", []),
    ("current_summary", None),
    ("current_keyword", None),
    ("current_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Chart helpers ─────────────────────────────────────────────────
BG   = "rgba(0,0,0,0)"
GRID = "rgba(0,0,0,0.05)"

def pie_chart(summary):
    labels = ["Positive", "Negative", "Neutral"]
    values = [summary["counts"][l] for l in labels]
    colors = ["#2d7a4f", "#e05555", "#cccccc"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.7,
        marker=dict(colors=colors, line=dict(color="#f5f0eb", width=3)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#333"),
        hovertemplate="<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10), height=260,
        annotations=[dict(
            text=f"<b>{summary['total']}</b>",
            x=0.5, y=0.5,
            font=dict(size=22, color="#1a1a1a"),
            showarrow=False
        )]
    )
    return fig

def bar_chart(summary):
    cats   = ["Positive", "Negative", "Neutral"]
    values = [summary["percentages"][c] for c in cats]
    colors = ["#2d7a4f", "#e05555", "#cccccc"]
    fig = go.Figure(go.Bar(
        x=cats, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="#333", size=12),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=30, b=10, l=10, r=10), height=260,
        yaxis=dict(showgrid=True, gridcolor=GRID,
                   tickfont=dict(color="#aaa"), ticksuffix="%",
                   range=[0, max(values) + 20]),
        xaxis=dict(tickfont=dict(color="#888", size=12)),
        bargap=0.4
    )
    return fig

def conf_chart(results):
    confs = [r["confidence"] for r in results]
    fig = go.Figure(go.Histogram(
        x=confs, nbinsx=20,
        marker=dict(color="#2d7a4f", opacity=0.7,
                    line=dict(color="#f5f0eb", width=1)),
        hovertemplate="Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=10, b=10, l=10, r=10), height=240,
        xaxis=dict(title=dict(text="Confidence %", font=dict(color="#aaa", size=11)),
                   tickfont=dict(color="#aaa"), gridcolor=GRID),
        yaxis=dict(title=dict(text="Count", font=dict(color="#aaa", size=11)),
                   tickfont=dict(color="#aaa"), gridcolor=GRID)
    )
    return fig

def wc_fig(results, sentiment):
    texts = " ".join(r["text"] for r in results if r["label"] == sentiment)
    if not texts.strip():
        return None
    color_map = {"Positive": "Greens", "Negative": "Reds", "Neutral": "Greys"}
    wc = WordCloud(
        width=700, height=280,
        background_color="#ffffff",
        colormap=color_map[sentiment],
        max_words=60, collocations=False,
        min_font_size=10
    ).generate(texts)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

# ── NAV ──────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
    <div class="nav-logo">Urdu Sentiment</div>
    <div class="nav-tag">
        <span class="nav-dot"></span>
        DistilBERT &nbsp;·&nbsp; Roman Urdu NLP
    </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">Natural Language Processing · Pakistan</div>
    <div class="hero-title">
        Analyze what Pakistanis<br>say about <span>any brand.</span>
    </div>
    <div class="hero-desc">
        Real-time Roman Urdu sentiment analysis from YouTube comments.
        Fine-tuned DistilBERT on 24,989 samples across 3 sentiment classes.
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATS BAR ────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-num">24,989</div>
        <div class="stat-label">Training Samples</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">74%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">3</div>
        <div class="stat-label">Sentiment Classes</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">DistilBERT</div>
        <div class="stat-label">Base Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── MARQUEE (brand logos) ────────────────────────────────────────
import base64

LOGO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logos")
logo_files = ["daraz", "jazz", "pti", "pmln", "foodpanda", "telenor",
              "bykea", "easypaisa", "hbl", "zong", "ufone", "netflix"]

def load_logo_b64(name):
    path = os.path.join(LOGO_DIR, f"{name}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_data = {name: load_logo_b64(name) for name in logo_files}
logo_data = {k: v for k, v in logo_data.items() if v is not None}

if logo_data:
    marquee_names = list(logo_data.keys()) * 2  # duplicate for seamless loop
    items_html = "".join(
        f'<span class="marquee-item"><img src="data:image/png;base64,{logo_data[n]}" alt="{n}"></span>'
        for n in marquee_names
    )
    st.markdown(f"""
    <div class="marquee-wrap">
        <div class="marquee-track">{items_html}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # fallback to text tags if logos not found
    brands_marquee = ["Daraz", "Jazz", "PTI", "PMLN", "Foodpanda", "Telenor",
                      "Bykea", "Easypaisa", "HBL", "Zong", "Ufone", "Netflix"] * 2
    items_html = "".join(f'<span class="marquee-item" style="border-radius:100px;padding:0.5rem 1.2rem;height:auto;font-family:Space Mono,monospace;font-size:0.75rem;color:#888;">{b}</span>' for b in brands_marquee)
    st.markdown(f"""
    <div class="marquee-wrap">
        <div class="marquee-track">{items_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ─────────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.markdown('<div class="search-label">Enter brand or topic</div>', unsafe_allow_html=True)
with st.form("search", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        query = st.text_input("q", placeholder="Daraz, Jazz, PTI, Telenor...", label_visibility="collapsed")
    with c2:
        go_btn = st.form_submit_button("Analyze")

brands = ["Daraz", "Jazz", "PTI", "PMLN", "Foodpanda", "Telenor", "Bykea", "Easypaisa"]
brand_cols = st.columns(len(brands), gap="small")
clicked_brand = None
for i, b in enumerate(brands):
    with brand_cols[i]:
        if st.button(b, key=f"brand_{b}"):
            clicked_brand = b

active_query = None
if go_btn and query.strip():
    active_query = query.strip()
elif clicked_brand:
    active_query = clicked_brand

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.spinner(""):
        resp = chat(active_query)
        if state["last_summary"]:
            st.session_state.current_summary = state["last_summary"]
            st.session_state.current_keyword = state["last_keyword"]
            st.session_state.current_results = state["last_results"]
            st.session_state.analysis_history.append({
                "keyword": state["last_keyword"],
                "summary": state["last_summary"]
            })
    st.session_state.messages.append({"role": "bot", "content": resp})
    st.rerun()

st.markdown("<br><br>", unsafe_allow_html=True)

if st.session_state.current_summary:
    summary = st.session_state.current_summary
    keyword = st.session_state.current_keyword
    results = st.session_state.current_results
    pct     = summary["percentages"]
    counts  = summary["counts"]

    st.markdown(f"""
    <div class="results-header">
        <div class="results-keyword">{keyword.upper()}</div>
        <div class="results-meta">{summary['total']} comments analyzed</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-cell m-pos">
            <div class="metric-pct">{pct['Positive']}%</div>
            <div class="metric-name">Positive</div>
            <div class="metric-count">{counts['Positive']} comments</div>
        </div>
        <div class="metric-cell m-neg">
            <div class="metric-pct">{pct['Negative']}%</div>
            <div class="metric-name">Negative</div>
            <div class="metric-count">{counts['Negative']} comments</div>
        </div>
        <div class="metric-cell m-neu">
            <div class="metric-pct">{pct['Neutral']}%</div>
            <div class="metric-name">Neutral</div>
            <div class="metric-count">{counts['Neutral']} comments</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_charts, col_comments = st.columns([1.2, 1], gap="large")

    with col_charts:
        st.markdown('<div class="s-header">Visualization</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Breakdown", "Confidence", "Word Cloud"])
        with tab1:
            st.plotly_chart(pie_chart(summary), use_container_width=True)
        with tab2:
            st.plotly_chart(bar_chart(summary), use_container_width=True)
        with tab3:
            st.plotly_chart(conf_chart(results), use_container_width=True)
            avg = sum(r["confidence"] for r in results) / len(results)
            st.markdown(
                f'<div style="font-family:Space Mono,monospace;font-size:0.65rem;'
                f'color:#aaa;text-align:center;letter-spacing:1px;text-transform:uppercase">'
                f'Average confidence: {avg:.1f}%</div>',
                unsafe_allow_html=True
            )
        with tab4:
            wc_sent = st.radio("", ["Positive", "Negative", "Neutral"], horizontal=True)
            fig = wc_fig(results, wc_sent)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close()
            else:
                st.markdown(f'<div style="color:#bbb;font-size:0.85rem">No {wc_sent} comments.</div>', unsafe_allow_html=True)

    with col_comments:
        st.markdown('<div class="s-header">Top Comments</div>', unsafe_allow_html=True)
        filt = st.selectbox("", ["Negative", "Positive", "Neutral"], label_visibility="collapsed")
        css  = {"Positive": "pos", "Negative": "neg", "Neutral": "neu"}[filt]
        top  = summary["top"][filt]
        if top:
            for c in top:
                st.markdown(f"""
                <div class="c-card {css}">
                    {c['text'][:180]}{"..." if len(c['text']) > 180 else ""}
                    <div class="c-conf">Confidence {c['confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:#bbb;font-size:0.85rem">No {filt.lower()} comments found.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="s-header">Export</div>', unsafe_allow_html=True)
        df_exp = pd.DataFrame([{
            "text": r["text"], "sentiment": r["label"],
            "confidence": r["confidence"],
            "positive_%": r["scores"]["Positive"],
            "negative_%": r["scores"]["Negative"],
            "neutral_%":  r["scores"]["Neutral"]
        } for r in results])
        st.download_button(
            "Download CSV",
            data=df_exp.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"{keyword}_sentiment.csv",
            mime="text/csv"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_hist, col_chat = st.columns([1, 1.5], gap="large")

    with col_hist:
        if st.session_state.analysis_history:
            st.markdown('<div class="s-header">History</div>', unsafe_allow_html=True)
            for item in reversed(st.session_state.analysis_history[-6:]):
                kw  = item["keyword"]
                dom = item["summary"]["dominant"]
                pct2 = item["summary"]["percentages"][dom]
                st.markdown(f"""
                <div class="h-item">
                    <span class="h-kw">{kw}</span>
                    <span class="h-dom">{dom} {pct2}%</span>
                </div>
                """, unsafe_allow_html=True)

    with col_chat:
        st.markdown('<div class="s-header">Conversation</div>', unsafe_allow_html=True)
        with st.container(height=300, border=True):
            for msg in st.session_state.messages[-8:]:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])

else:
    st.markdown("""
    <div class="empty">
        <div class="empty-title">No analysis yet — enter a brand above</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    Disclaimer: All product names, logos, and brands are property of their respective owners.
    All company, product, and service names used in this application are for identification
    and non-commercial educational purposes only. Use of these names, logos, and brands
    does not imply endorsement.
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)