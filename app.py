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
    background: #0d0d0d;
    color: #f0f0f0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2.5rem 4rem 2.5rem;
    max-width: 1400px;
}

/* ── NAV ── */
.nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 0;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 4rem;
}
.nav-logo {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #f0f0f0;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.nav-tag {
    font-size: 0.72rem;
    color: #444444;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.nav-dot {
    width: 8px;
    height: 8px;
    background: #c8f04a;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── HERO ── */
.hero {
    margin-bottom: 4rem;
}
.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #c8f04a;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 700;
    color: #f0f0f0;
    line-height: 1.05;
    letter-spacing: -1.5px;
    margin-bottom: 1.5rem;
}
.hero-title span {
    color: #c8f04a;
}
.hero-desc {
    font-size: 0.95rem;
    color: #555555;
    max-width: 500px;
    line-height: 1.7;
}

/* ── SEARCH BAR ── */
.search-section {
    margin-bottom: 3rem;
}
.search-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #444444;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}
.stTextInput input {
    background: #111111 !important;
    border: 1px solid #222222 !important;
    border-radius: 2px !important;
    color: #f0f0f0 !important;
    font-size: 1rem !important;
    padding: 1rem 1.2rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus {
    border-color: #c8f04a !important;
    box-shadow: none !important;
}
.stTextInput input::placeholder { color: #333333 !important; }
.stTextInput label { display: none !important; }

/* ── ANALYZE BUTTON (form submit) ── */
[data-testid="stFormSubmitButton"] button {
    background: #c8f04a !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: background 0.2s, transform 0.1s !important;
    cursor: pointer !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #d8ff5a !important;
    transform: translateY(-1px) !important;
}

/* ── BRAND TAG BUTTONS ── */
[data-testid="stHorizontalBlock"] .stButton button {
    background: transparent !important;
    border: 1px solid #222222 !important;
    color: #555555 !important;
    border-radius: 100px !important;
    font-size: 0.72rem !important;
    padding: 0.15rem 0.5rem !important;
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
    border-color: #c8f04a !important;
    color: #c8f04a !important;
    transform: none !important;
    background: transparent !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 0.2rem !important;
}

/* ── RESULTS HEADER ── */
.results-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1a1a1a;
}
.results-keyword {
    font-size: 2rem;
    font-weight: 700;
    color: #f0f0f0;
    letter-spacing: -0.5px;
}
.results-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #444444;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── METRIC GRID ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #1a1a1a;
    border: 1px solid #1a1a1a;
    margin-bottom: 3rem;
}
.metric-cell {
    background: #0d0d0d;
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
    color: #444444;
}
.metric-count {
    font-size: 0.8rem;
    color: #333333;
    margin-top: 0.3rem;
}
.m-pos .metric-pct { color: #c8f04a; }
.m-neg .metric-pct { color: #ff4d4d; }
.m-neu .metric-pct { color: #666666; }

/* ── SECTION HEADER ── */
.s-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #444444;
    text-transform: uppercase;
    letter-spacing: 3px;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 1.5rem;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a1a1a !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #444444 !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1.2rem !important;
    border-radius: 0 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid transparent !important;
    font-family: 'Space Mono', monospace !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #c8f04a !important;
    border-bottom: 1px solid #c8f04a !important;
}

/* ── CHAT ── */
.stChatMessage {
    background: #111111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 2px !important;
}
[data-testid="stChatMessageContent"] p {
    color: #c0c0c0 !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #1a1a1a !important;
    border-radius: 2px !important;
    background: #0d0d0d !important;
}

/* ── COMMENT CARD ── */
.c-card {
    padding: 1rem 1.2rem;
    border-left: 2px solid #c8f04a;
    background: #111111;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: #aaaaaa;
    line-height: 1.6;
}
.c-card.neg { border-left-color: #ff4d4d; }
.c-card.neu { border-left-color: #333333; }
.c-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #333333;
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
    border-bottom: 1px solid #111111;
    font-size: 0.85rem;
}
.h-kw { color: #c0c0c0; }
.h-dom { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #444444; }

/* ── EMPTY ── */
.empty {
    padding: 5rem 2rem;
    text-align: center;
    border: 1px dashed #1a1a1a;
}
.empty-title {
    font-size: 0.85rem;
    color: #2a2a2a;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: #111111 !important;
    border: 1px solid #222222 !important;
    color: #f0f0f0 !important;
    border-radius: 2px !important;
    font-size: 0.85rem !important;
}

/* ── RADIO ── */
.stRadio label { color: #555555 !important; font-size: 0.82rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #555555 !important; }

/* ── DOWNLOAD ── */
.stDownloadButton button {
    background: transparent !important;
    border: 1px solid #222222 !important;
    color: #555555 !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
.stDownloadButton button:hover {
    border-color: #c8f04a !important;
    color: #c8f04a !important;
    transform: none !important;
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
GRID = "rgba(255,255,255,0.04)"

def pie_chart(summary):
    labels = ["Positive", "Negative", "Neutral"]
    values = [summary["counts"][l] for l in labels]
    colors = ["#c8f04a", "#ff4d4d", "#333333"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.7,
        marker=dict(colors=colors, line=dict(color="#0d0d0d", width=3)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#f0f0f0"),
        hovertemplate="<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10), height=260,
        annotations=[dict(
            text=f"<b style='font-size:22px'>{summary['total']}</b>",
            x=0.5, y=0.5,
            font=dict(size=22, color="#f0f0f0"),
            showarrow=False
        )]
    )
    return fig

def bar_chart(summary):
    cats   = ["Positive", "Negative", "Neutral"]
    values = [summary["percentages"][c] for c in cats]
    colors = ["#c8f04a", "#ff4d4d", "#333333"]
    fig = go.Figure(go.Bar(
        x=cats, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="#f0f0f0", size=12),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=30, b=10, l=10, r=10), height=260,
        yaxis=dict(showgrid=True, gridcolor=GRID,
                   tickfont=dict(color="#444444"), ticksuffix="%",
                   range=[0, max(values) + 20]),
        xaxis=dict(tickfont=dict(color="#888888", size=12)),
        bargap=0.4
    )
    return fig

def conf_chart(results):
    confs = [r["confidence"] for r in results]
    fig = go.Figure(go.Histogram(
        x=confs, nbinsx=20,
        marker=dict(color="#c8f04a", opacity=0.7,
                    line=dict(color="#0d0d0d", width=1)),
        hovertemplate="Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=10, b=10, l=10, r=10), height=240,
        xaxis=dict(
            title=dict(text="Confidence %", font=dict(color="#444444", size=11)),
            tickfont=dict(color="#444444"), gridcolor=GRID
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(color="#444444", size=11)),
            tickfont=dict(color="#444444"), gridcolor=GRID
        )
    )
    return fig

def wc_fig(results, sentiment):
    texts = " ".join(r["text"] for r in results if r["label"] == sentiment)
    if not texts.strip():
        return None
    color_map = {"Positive": "YlGn", "Negative": "Reds", "Neutral": "Greys"}
    wc = WordCloud(
        width=700, height=280,
        background_color="#0d0d0d",
        colormap=color_map[sentiment],
        max_words=60, collocations=False,
        min_font_size=10
    ).generate(texts)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
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
        DistilBERT &nbsp;·&nbsp; 74% accuracy &nbsp;·&nbsp; 24,989 samples
    </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">Natural Language Processing</div>
    <div class="hero-title">
        Analyze what<br>Pakistanis say<br>about <span>any brand.</span>
    </div>
    <div class="hero-desc">
        Real-time Roman Urdu sentiment analysis from YouTube comments.
        Enter any Pakistani brand or topic to see public opinion.
    </div>
</div>
""", unsafe_allow_html=True)

# ── SEARCH ───────────────────────────────────────────────────────
st.markdown('<div class="search-label">Enter brand or topic</div>', unsafe_allow_html=True)
with st.form("search", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        query = st.text_input("q", placeholder="Daraz, Jazz, PTI, Telenor...", label_visibility="collapsed")
    with c2:
        go_btn = st.form_submit_button("Analyze")

# ── BRAND TAG BUTTONS (clickable) ────────────────────────────────
brands = ["Daraz", "Jazz", "PTI", "PMLN", "Foodpanda", "Telenor", "Bykea", "Easypaisa"]
brand_cols = st.columns(len(brands), gap="small")
clicked_brand = None
for i, b in enumerate(brands):
    with brand_cols[i]:
        if st.button(b, key=f"brand_{b}"):
            clicked_brand = b

# resolve input: form submit or brand button click
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

# ── RESULTS ──────────────────────────────────────────────────────
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
                f'color:#444444;text-align:center;letter-spacing:1px;text-transform:uppercase">'
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
                st.markdown(f'<div style="color:#333333;font-size:0.85rem">No {wc_sent} comments.</div>', unsafe_allow_html=True)

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
            st.markdown(f'<div style="color:#333333;font-size:0.85rem">No {filt.lower()} comments found.</div>', unsafe_allow_html=True)

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