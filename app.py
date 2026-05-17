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

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Roman Urdu Sentiment Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme Definitions ─────────────────────────────────────────────
THEMES = {
    "Purple": {
        "accent":     "#7c6af7",
        "accent2":    "#a78bfa",
        "bg":         "#0a0a0f",
        "bg2":        "#0f0f1a",
        "bg3":        "#16161f",
        "border":     "rgba(124,106,247,0.2)",
        "pos":        "#4ade80",
        "neg":        "#f87171",
        "neu":        "#60a5fa",
        "text":       "#e8e8f0",
        "muted":      "#666680",
        "label":      "🟣 Purple",
    },
    "Cyan": {
        "accent":     "#06b6d4",
        "accent2":    "#67e8f9",
        "bg":         "#030a0f",
        "bg2":        "#071219",
        "bg3":        "#0c1a24",
        "border":     "rgba(6,182,212,0.2)",
        "pos":        "#4ade80",
        "neg":        "#f87171",
        "neu":        "#a78bfa",
        "text":       "#e0f7ff",
        "muted":      "#4a7a8a",
        "label":      "🩵 Cyan",
    },
    "Green": {
        "accent":     "#22c55e",
        "accent2":    "#86efac",
        "bg":         "#030a05",
        "bg2":        "#071209",
        "bg3":        "#0c1a0e",
        "border":     "rgba(34,197,94,0.2)",
        "pos":        "#86efac",
        "neg":        "#f87171",
        "neu":        "#60a5fa",
        "text":       "#e0ffe8",
        "muted":      "#3a6644",
        "label":      "🟢 Green",
    },
    "Amber": {
        "accent":     "#f59e0b",
        "accent2":    "#fcd34d",
        "bg":         "#0f0a00",
        "bg2":        "#1a1200",
        "bg3":        "#241900",
        "border":     "rgba(245,158,11,0.2)",
        "pos":        "#4ade80",
        "neg":        "#f87171",
        "neu":        "#60a5fa",
        "text":       "#fff8e8",
        "muted":      "#7a6030",
        "label":      "🟡 Amber",
    },
}

# ── Session State ─────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("analysis_history", []),
    ("current_summary", None),
    ("current_keyword", None),
    ("current_results", None),
    ("theme", "Purple"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

T = THEMES[st.session_state.theme]

# ── Dynamic CSS ───────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600;700&display=swap');

* {{ font-family: 'DM Sans', sans-serif; box-sizing: border-box; }}

.stApp {{ background: {T['bg']}; color: {T['text']}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }}

.app-header {{
    text-align: center;
    padding: 1.5rem 0 1.2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}}
.app-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0;
}}
.app-title span {{ color: {T['accent']}; }}
.app-subtitle {{
    color: {T['muted']};
    font-size: 0.9rem;
    margin-top: 0.4rem;
}}

section[data-testid="stSidebar"] {{
    background: {T['bg2']} !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}}
section[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}

.sidebar-section {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {T['accent']} !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.2rem 0 0.6rem 0;
}}
.history-pill {{
    background: rgba(255,255,255,0.04);
    border: 1px solid {T['border']};
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: {T['text']};
}}
.cmd-item {{
    color: {T['muted']} !important;
    font-size: 0.82rem;
    padding: 0.15rem 0;
    font-family: 'IBM Plex Mono', monospace !important;
}}

.stChatMessage {{
    background: {T['bg3']} !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}}
[data-testid="stChatMessageContent"] p {{ color: {T['text']} !important; }}

.stTextInput input {{
    background: {T['bg3']} !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
}}
.stTextInput input:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 2px {T['border']} !important;
}}
.stTextInput input::placeholder {{ color: {T['muted']} !important; }}

.stButton button {{
    background: {T['accent']} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}}
.stButton button:hover {{ opacity: 0.85 !important; }}

.metric-row {{
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1rem;
}}
.metric-box {{
    flex: 1;
    background: {T['bg2']};
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}}
.metric-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.3rem;
}}
.metric-lbl {{
    font-size: 0.75rem;
    color: {T['muted']};
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.pos {{ color: {T['pos']}; border-top: 2px solid {T['pos']}; }}
.neg {{ color: {T['neg']}; border-top: 2px solid {T['neg']}; }}
.neu {{ color: {T['neu']}; border-top: 2px solid {T['neu']}; }}

.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {T['accent']};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {T['border']};
}}

.stTabs [data-baseweb="tab-list"] {{
    background: {T['bg2']} !important;
    border-radius: 8px !important;
    padding: 0.2rem !important;
    gap: 0.2rem !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {T['muted']} !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 0.8rem !important;
}}
.stTabs [aria-selected="true"] {{
    background: {T['accent']} !important;
    color: white !important;
}}

.stSelectbox > div > div {{
    background: {T['bg3']} !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: {T['text']} !important;
    border-radius: 8px !important;
}}

.dash-empty {{
    background: {T['bg2']};
    border: 1px dashed rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
}}
.dash-empty-icon {{ font-size: 2.5rem; margin-bottom: 0.8rem; }}
.dash-empty-text {{ font-size: 0.9rem; color: {T['muted']}; }}

.stRadio label {{ color: {T['text']} !important; font-size: 0.85rem !important; }}

.stDownloadButton button {{
    background: {T['bg3']} !important;
    border: 1px solid {T['border']} !important;
    color: {T['accent']} !important;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    background: {T['bg2']} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Chart Helpers ─────────────────────────────────────────────────
CHART_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.05)"
COLORS     = {"Positive": T['pos'], "Negative": T['neg'], "Neutral": T['neu']}


def pie_chart(summary):
    labels = ["Positive", "Negative", "Neutral"]
    values = [summary["counts"][l] for l in labels]
    colors = [COLORS[l] for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors, line=dict(color=T['bg'], width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value} comments<br>%{percent}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        annotations=[dict(
            text=f"<b>{summary['total']}</b><br><span style='font-size:11px'>comments</span>",
            x=0.5, y=0.5,
            font=dict(size=16, color="white"),
            showarrow=False
        )]
    )
    return fig


def bar_chart(summary):
    cats   = ["Positive", "Negative", "Neutral"]
    values = [summary["percentages"][c] for c in cats]
    colors = [COLORS[c] for c in cats]
    fig = go.Figure(go.Bar(
        x=cats, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=13),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        margin=dict(t=20, b=10, l=10, r=10),
        height=250,
        yaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR,
            tickfont=dict(color=T['muted']),
            ticksuffix="%",
            range=[0, max(values) + 15]
        ),
        xaxis=dict(tickfont=dict(color=T['text'], size=13)),
        bargap=0.35
    )
    return fig


def confidence_chart(results):
    confs = [r["confidence"] for r in results]
    fig = go.Figure(go.Histogram(
        x=confs, nbinsx=20,
        marker=dict(color=T['accent'], line=dict(color=T['bg'], width=1)),
        hovertemplate="Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        margin=dict(t=10, b=10, l=10, r=10),
        height=230,
        xaxis=dict(
            title=dict(text="Confidence %", font=dict(color=T['muted'], size=11)),
            tickfont=dict(color=T['muted']),
            gridcolor=GRID_COLOR
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(color=T['muted'], size=11)),
            tickfont=dict(color=T['muted']),
            gridcolor=GRID_COLOR
        )
    )
    return fig


def wordcloud_fig(results, sentiment):
    texts = " ".join(r["text"] for r in results if r["label"] == sentiment)
    if not texts.strip():
        return None
    cmap = {"Positive": "Greens", "Negative": "Reds", "Neutral": "Blues"}
    wc = WordCloud(
        width=700, height=300,
        background_color=None, mode="RGBA",
        colormap=cmap[sentiment],
        max_words=60, collocations=False,
        min_font_size=10
    ).generate(texts)
    fig, ax = plt.subplots(figsize=(7, 3), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── Header ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <div class="app-title">Roman Urdu <span>Sentiment</span> Bot</div>
    <div class="app-subtitle">Pakistani brands aur topics ke YouTube comments analyze karein</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">🎨 Theme</div>', unsafe_allow_html=True)
    theme_options = {t: THEMES[t]["label"] for t in THEMES}
    selected_theme = st.radio(
        "Theme",
        options=list(theme_options.keys()),
        format_func=lambda x: theme_options[x],
        index=list(theme_options.keys()).index(st.session_state.theme),
        label_visibility="collapsed"
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.markdown('<div class="sidebar-section">Quick Analyze</div>', unsafe_allow_html=True)
    brands = ["Daraz", "Jazz", "Telenor", "Foodpanda", "Easypaisa", "PIA", "PTI", "PMLN", "Bykea"]
    pick   = st.selectbox("Brand:", ["-- select --"] + brands, label_visibility="collapsed")

    if st.button("Analyze →"):
        if pick != "-- select --":
            st.session_state.messages.append({"role": "user", "content": pick})
            with st.spinner(f"Analyzing {pick}..."):
                resp = chat(pick)
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

    st.markdown('<div class="sidebar-section">History</div>', unsafe_allow_html=True)
    if st.session_state.analysis_history:
        for item in reversed(st.session_state.analysis_history[-6:]):
            kw  = item["keyword"]
            dom = item["summary"]["dominant"]
            pct = item["summary"]["percentages"][dom]
            ico = {"Positive": "↑", "Negative": "↓", "Neutral": "→"}[dom]
            st.markdown(
                f'<div class="history-pill">{ico} <b>{kw}</b> — {dom} {pct}%</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(f'<div style="color:{T["muted"]};font-size:0.82rem;">No history yet</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Commands</div>', unsafe_allow_html=True)
    for cmd in ["`Daraz`", "`analyze Jazz`", "`show negative`", "`show positive`", "`show neutral`", "`help`"]:
        st.markdown(f'<div class="cmd-item">{cmd}</div>', unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────
col_chat, col_dash = st.columns([1, 1.1], gap="large")

# ── Chat Column ───────────────────────────────────────────────────
with col_chat:
    st.markdown('<div class="section-label">💬 Chat</div>', unsafe_allow_html=True)

    with st.container(height=440, border=True):
        if not st.session_state.messages:
            st.markdown(
                f"<p style='color:{T['muted']};text-align:center;margin-top:5rem;font-size:0.9rem'>"
                f"Koi brand ka naam likhein shuru karne ke liye 👇<br>"
                f"<span style='font-size:0.8rem'>e.g. Daraz, Jazz, Foodpanda</span></p>",
                unsafe_allow_html=True
            )
        else:
            for msg in st.session_state.messages[-12:]:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input(
                "msg",
                placeholder="Type a brand name or command...",
                label_visibility="collapsed"
            )
        with c2:
            send = st.form_submit_button("Send")

    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Analyzing..."):
            resp = chat(user_input)
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

# ── Dashboard Column ──────────────────────────────────────────────
with col_dash:
    st.markdown('<div class="section-label">📊 Dashboard</div>', unsafe_allow_html=True)

    dash_container = st.container()
    with dash_container:
        if st.session_state.current_summary:
            summary = st.session_state.current_summary
            keyword = st.session_state.current_keyword
            results = st.session_state.current_results
            pct     = summary["percentages"]
            counts  = summary["counts"]

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box pos">
                    <div class="metric-num">{pct['Positive']}%</div>
                    <div class="metric-lbl">✅ Positive</div>
                    <div style="color:{T['pos']};font-size:0.78rem;margin-top:0.2rem">{counts['Positive']} comments</div>
                </div>
                <div class="metric-box neg">
                    <div class="metric-num">{pct['Negative']}%</div>
                    <div class="metric-lbl">❌ Negative</div>
                    <div style="color:{T['neg']};font-size:0.78rem;margin-top:0.2rem">{counts['Negative']} comments</div>
                </div>
                <div class="metric-box neu">
                    <div class="metric-num">{pct['Neutral']}%</div>
                    <div class="metric-lbl">😐 Neutral</div>
                    <div style="color:{T['neu']};font-size:0.78rem;margin-top:0.2rem">{counts['Neutral']} comments</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            tab1, tab2, tab3, tab4 = st.tabs(["Pie", "Bar", "Confidence", "Word Cloud"])

            with tab1:
                st.plotly_chart(pie_chart(summary), use_container_width=True)

            with tab2:
                st.plotly_chart(bar_chart(summary), use_container_width=True)

            with tab3:
                st.plotly_chart(confidence_chart(results), use_container_width=True)
                avg = sum(r["confidence"] for r in results) / len(results)
                st.markdown(
                    f'<div style="color:{T["muted"]};font-size:0.8rem;text-align:center">'
                    f'Average confidence: {avg:.1f}%</div>',
                    unsafe_allow_html=True
                )

            with tab4:
                wc_sent = st.radio("Show:", ["Positive", "Negative", "Neutral"], horizontal=True)
                wc_fig  = wordcloud_fig(results, wc_sent)
                if wc_fig:
                    st.pyplot(wc_fig, use_container_width=True)
                    plt.close()
                else:
                    st.info(f"No {wc_sent} comments found")

            st.markdown(
                f'<div class="section-label" style="margin-top:1rem">🔍 Top Comments</div>',
                unsafe_allow_html=True
            )
            show_sent = st.selectbox(
                "Filter:", ["Negative", "Positive", "Neutral"],
                label_visibility="collapsed"
            )
            top = summary["top"][show_sent]
            if top:
                for i, c in enumerate(top):
                    color = COLORS[show_sent]
                    st.markdown(f"""
                    <div style="background:{T['bg2']};border:1px solid rgba(255,255,255,0.06);
                    border-left:3px solid {color};border-radius:8px;
                    padding:0.7rem 1rem;margin:0.4rem 0;font-size:0.84rem;color:{T['text']}">
                        {c['text'][:150]}{"..." if len(c['text']) > 150 else ""}
                        <div style="color:{T['muted']};font-size:0.75rem;margin-top:0.3rem">
                            Confidence: {c['confidence']}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No {show_sent} comments found")

            st.markdown(
                f'<div class="section-label" style="margin-top:1rem">💾 Export</div>',
                unsafe_allow_html=True
            )
            df_export = pd.DataFrame([{
                "text":       r["text"],
                "sentiment":  r["label"],
                "confidence": r["confidence"],
                "positive_%": r["scores"]["Positive"],
                "negative_%": r["scores"]["Negative"],
                "neutral_%":  r["scores"]["Neutral"]
            } for r in results])
            csv = df_export.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="⬇ Download CSV",
                data=csv,
                file_name=f"{keyword}_sentiment_analysis.csv",
                mime="text/csv"
            )

        else:
            st.markdown("""
            <div class="dash-empty">
                <div class="dash-empty-icon">📊</div>
                <div class="dash-empty-text">
                    Type a brand name in the chat<br>or select one from the sidebar
                </div>
            </div>
            """, unsafe_allow_html=True)