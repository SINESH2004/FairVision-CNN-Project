import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairVision AI · Age Intelligence",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — clean white + slate theme, high contrast, readable everywhere
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
}

/* ── App background ── */
.stApp {
    background: #f4f6f9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #94a3b8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #f1f5f9 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] .stMarkdown table td,
[data-testid="stSidebar"] .stMarkdown table th {
    color: #cbd5e1 !important;
    border-color: #1e293b !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
}

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e293b !important;
    color: #f1f5f9 !important;
}

/* ── Sidebar slider ── */
[data-testid="stSidebar"] [data-baseweb="slider"] [data-baseweb="thumb"] {
    background: #06b6d4 !important;
    border-color: #06b6d4 !important;
}
[data-testid="stSidebar"] [data-baseweb="track-foreground"] {
    background: #06b6d4 !important;
}
[data-testid="stSidebar"] [data-baseweb="track-background"] {
    background: #334155 !important;
}

/* ── Sidebar toggle ── */
[data-testid="stSidebar"] [data-baseweb="toggle"] {
    background: #06b6d4 !important;
}

/* ── Main area headings ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: #0f172a !important;
}
p, li { color: #334155 !important; }
label { color: #475569 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #06b6d4, #0284c7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(6,182,212,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(6,182,212,0.45) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 14px !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #06b6d4 !important;
}
[data-testid="stFileUploader"] * {
    color: #64748b !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #06b6d4, #0284c7) !important;
    border-radius: 99px !important;
}
.stProgress > div {
    background: #e2e8f0 !important;
    border-radius: 99px !important;
}
.stProgress p {
    color: #1e293b !important;
    font-size: 0.82rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── Alerts ── */
.stSuccess { background: #f0fdf4 !important; color: #166534 !important; border-color: #16a34a !important; }
.stError   { background: #fef2f2 !important; color: #991b1b !important; border-color: #dc2626 !important; }
.stInfo    { background: #eff6ff !important; color: #1e40af !important; border-color: #3b82f6 !important; }

/* ── HR ── */
hr { border-color: #e2e8f0 !important; }

/* ── Caption ── */
.stCaption, caption { color: #94a3b8 !important; font-size: 0.72rem !important; }

/* ── Custom components ── */
.fv-hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.fv-hero-accent {
    color: #06b6d4;
}
.fv-sub {
    color: #64748b;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.fv-tag {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 5px;
    margin-bottom: 4px;
}
.fv-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
.fv-card-accent {
    border-top: 3px solid #06b6d4;
}
.fv-status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.fv-status-green { background: #22c55e; }
.fv-status-yellow { background: #f59e0b; }
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin: 0.8rem 0;
}
.stat-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: #0f172a;
    font-weight: 700;
}
.stat-lbl {
    font-size: 0.62rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.rank-pill {
    display: inline-block;
    background: #06b6d4;
    color: #fff;
    border-radius: 99px;
    padding: 1px 8px;
    font-size: 0.65rem;
    font-weight: 700;
    margin-right: 6px;
}
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: #ffffff;
    border: 2px dashed #e2e8f0;
    border-radius: 16px;
}
.empty-icon { font-size: 3rem; margin-bottom: 0.75rem; }
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: #64748b;
    font-weight: 600;
}
.empty-sub { font-size: 0.78rem; color: #94a3b8; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
AGE_NAMES    = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","more than 70"]
AGE_ICONS    = ["👶","🧒","🧑","👨","🧔","👨‍🦳","👴","👴","🧓"]
GENDER_NAMES = ["Male","Female"]
RACE_NAMES   = ["East Asian","Indian","Black","White","Middle Eastern","Latino_Hispanic","Southeast Asian"]
IMG_SIZE     = 224

RANK_ICONS   = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣"]

# Matplotlib style — white background for all charts
CHART_BG   = "#ffffff"
CHART_FG   = "#0f172a"
CHART_GRID = "#f1f5f9"
CHART_CYAN = "#06b6d4"
CHART_BLUE = "#0284c7"
CHART_MUTED = "#94a3b8"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class BetterCNN_V2(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,256), nn.ReLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(256,num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


def apply_preprocessing(img, brightness, contrast, sharpen, denoise):
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    return img


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BetterCNN_V2(num_classes=9)
    ckpt   = torch.load(path, map_location=device)
    state  = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, device, pil_image, temperature=1.0, topk=3):
    tensor = val_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor) / temperature
        probs  = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, k=topk)
    results = [
        {"label": AGE_NAMES[i.item()], "icon": AGE_ICONS[i.item()],
         "prob": p.item(), "idx": i.item()}
        for i, p in zip(top_idx, top_probs)
    ]
    return results, probs.cpu().numpy()


def run_tta(model, device, pil_image, n_aug=5):
    tta_transforms = [
        transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE+20,IMG_SIZE+20)),
            transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ColorJitter(brightness=0.1,contrast=0.1), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((IMG_SIZE+10,IMG_SIZE+10)),
            transforms.RandomCrop(IMG_SIZE), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    ]
    all_probs = []
    with torch.no_grad():
        for t in tta_transforms[:n_aug]:
            tensor = t(pil_image).unsqueeze(0).to(device)
            all_probs.append(F.softmax(model(tensor), dim=1)[0].cpu().numpy())
    avg      = np.mean(all_probs, axis=0)
    top3_idx = avg.argsort()[::-1][:3]
    results  = [{"label": AGE_NAMES[i], "icon": AGE_ICONS[i],
                 "prob": float(avg[i]), "idx": int(i)} for i in top3_idx]
    return results, avg


def entropy(probs):
    p = probs + 1e-9
    return float(-np.sum(p * np.log(p)))


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS  — all white-background, high-contrast
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def make_gauge(conf: float):
    fig, ax = plt.subplots(figsize=(3.2, 2.0), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    theta      = np.linspace(np.pi, 0, 300)
    fill_theta = np.linspace(np.pi, np.pi - conf * np.pi, 300)
    color = CHART_CYAN if conf > 0.6 else "#f59e0b" if conf > 0.35 else "#ef4444"
    ax.plot(np.cos(theta), np.sin(theta), color="#e2e8f0", linewidth=16, solid_capstyle="round")
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=16, solid_capstyle="round")
    ax.text(0, 0.05, f"{conf*100:.1f}%", ha="center", va="center",
            fontsize=20, color=CHART_FG, fontweight="bold", fontfamily="monospace")
    ax.text(0, -0.35, "CONFIDENCE", ha="center", fontsize=7,
            color=CHART_MUTED, fontfamily="monospace")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.1)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def make_distribution_bar(probs):
    fig, ax = plt.subplots(figsize=(8, 3.0), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    max_idx = int(np.argmax(probs))
    colors  = [CHART_CYAN if i == max_idx else "#e0f2fe" for i in range(len(probs))]
    bars    = ax.bar(AGE_NAMES, probs * 100, color=colors, width=0.6, zorder=3,
                     edgecolor="#e2e8f0", linewidth=0.8)
    for bar, p in zip(bars, probs * 100):
        if p > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{p:.1f}", ha="center", va="bottom", fontsize=7.5,
                    color=CHART_FG, fontfamily="monospace", fontweight="bold" if p == max(probs*100) else "normal")
    ax.set_ylabel("Probability (%)", color=CHART_MUTED, fontsize=8, fontfamily="monospace")
    ax.tick_params(axis="x", rotation=35, colors=CHART_FG, labelsize=8)
    ax.tick_params(axis="y", colors=CHART_MUTED, labelsize=7)
    ax.set_ylim(0, max(probs * 100) * 1.3 + 3)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#e2e8f0")
    ax.set_title("Full Probability Distribution", color=CHART_FG, fontsize=9,
                 fontfamily="monospace", pad=10, fontweight="bold")
    ax.grid(axis="y", color=CHART_GRID, linewidth=0.8, zorder=0)
    fig.tight_layout()
    return fig


def make_horizontal_bars(top_results):
    n      = len(top_results)
    labels = [r["label"] for r in top_results]
    probs  = [r["prob"] * 100 for r in top_results]
    cmap   = [CHART_CYAN, "#0284c7", "#0ea5e9", "#38bdf8", "#7dd3fc",
              "#bae6fd", "#e0f2fe", "#f0f9ff", "#f8fcff"]

    fig, ax = plt.subplots(figsize=(5, max(1.6, n * 0.55 + 0.5)), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    bars = ax.barh(labels[::-1], probs[::-1], color=cmap[:n][::-1],
                   height=0.5, edgecolor="#e2e8f0", linewidth=0.6)
    ax.set_xlim(0, 110)
    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f"{p:.1f}%", va="center", fontsize=9,
                color=CHART_FG, fontfamily="monospace", fontweight="bold")
    ax.spines[["top","right","left","bottom"]].set_color("#e2e8f0")
    ax.tick_params(colors=CHART_FG, labelsize=9)
    ax.set_xlabel("Confidence (%)", color=CHART_MUTED, fontsize=8, fontfamily="monospace")
    ax.grid(axis="x", color=CHART_GRID, linewidth=0.6)
    ax.set_title("Top Predictions", color=CHART_FG, fontsize=9,
                 fontfamily="monospace", pad=8, fontweight="bold")
    fig.tight_layout()
    return fig


def make_radar(probs):
    N      = len(AGE_NAMES)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals   = probs.tolist()
    angles += angles[:1]
    vals   += vals[:1]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True), facecolor=CHART_BG)
    ax.set_facecolor("#f8fafc")
    ax.plot(angles, vals, color=CHART_CYAN, linewidth=2.5)
    ax.fill(angles, vals, color=CHART_CYAN, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AGE_NAMES, size=7, color=CHART_FG, fontfamily="monospace")
    ax.set_yticklabels([])
    ax.grid(color="#e2e8f0", linewidth=0.8)
    ax.spines["polar"].set_color("#e2e8f0")
    ax.set_title("Radar View", color=CHART_FG, fontsize=9,
                 fontfamily="monospace", pad=14, fontweight="bold")
    fig.tight_layout()
    return fig


def make_entropy_bar(ent_val, max_ent):
    fig, ax = plt.subplots(figsize=(4, 0.9), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    ratio = ent_val / max_ent
    color = "#22c55e" if ratio < 0.35 else "#f59e0b" if ratio < 0.65 else "#ef4444"
    ax.barh([0], [ratio*100], color=color, height=0.5)
    ax.barh([0], [100], color="#f1f5f9", height=0.5)
    ax.barh([0], [ratio*100], color=color, height=0.5)
    ax.set_xlim(0, 100)
    ax.text(ratio*100 + 1, 0, f"{ratio*100:.1f}%", va="center",
            fontsize=9, color=CHART_FG, fontfamily="monospace", fontweight="bold")
    ax.text(0, -0.55, "LOW UNCERTAINTY", fontsize=6.5, color="#22c55e", fontfamily="monospace")
    ax.text(100, -0.55, "HIGH UNCERTAINTY", fontsize=6.5, color="#ef4444",
            fontfamily="monospace", ha="right")
    ax.axis("off")
    ax.set_ylim(-0.8, 0.6)
    fig.tight_layout(pad=0.1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#f1f5f9;margin-bottom:0">FairVision AI</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;margin-top:0">Age Group Intelligence</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ Model")
    model_path = st.text_input("Weights file (.pth)", value="best_fairface_model.pth")
    load_btn   = st.button("🔄 Load / Reload Model", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔬 Inference")

    use_tta = st.toggle("Test-Time Augmentation", value=False,
                        help="Averages predictions over multiple augmented views.")
    tta_n = st.slider("TTA views", 2, 5, 3, disabled=not use_tta)

    temperature = st.slider("Temperature", 0.3, 2.0, 1.0, 0.1,
                            help="<1 = sharper  |  >1 = smoother")
    topk = st.slider("Top-K results", 2, 9, 3)

    st.markdown("---")
    st.markdown("### 🖼️ Preprocessing")
    brightness = st.slider("Brightness", 0.5, 1.8, 1.0, 0.05)
    contrast   = st.slider("Contrast",   0.5, 1.8, 1.0, 0.05)
    sharpen    = st.toggle("Sharpen", value=False)
    denoise    = st.toggle("Denoise", value=False)

    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.markdown("""
| | |
|---|---|
| **Arch** | BetterCNN_V2 |
| **Input** | 224 × 224 |
| **Classes** | 9 age groups |
| **Framework** | PyTorch |
| **Dataset** | FairFace 0.25 |
""")

    st.markdown("---")
    st.markdown("### ⚠️ Responsible Use")
    st.caption("Not for surveillance, biometric ID, or high-stakes decisions. Performance varies across demographic groups.")
    st.markdown("---")
    st.caption("IJSE · CAME · 2025/2026")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
if "model" not in st.session_state or load_btn:
    if not os.path.isfile(model_path):
        st.error(f"❌ `{model_path}` not found. Place your `.pth` file next to `app.py`.")
        st.stop()
    with st.spinner("Loading model …"):
        try:
            st.session_state.model, st.session_state.device = load_model(model_path)
            st.session_state.model_path = model_path
        except Exception as e:
            st.error(f"Load error: {e}")
            st.stop()

model  = st.session_state.model
device = st.session_state.device

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown(
        '<div class="fv-hero-title">Fair<span class="fv-hero-accent">Vision</span> AI</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="fv-sub" style="margin:0.4rem 0 0.8rem">CNN · Age Group Intelligence · FairFace Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
<span class="fv-tag">BetterCNN_V2</span>
<span class="fv-tag">9 Age Classes</span>
<span class="fv-tag">PyTorch</span>
<span class="fv-tag">FairFace 0.25</span>
<span class="fv-tag">97,698 Samples</span>
""", unsafe_allow_html=True)

with hcol2:
    dev_label = "CUDA" if torch.cuda.is_available() else "CPU"
    dev_dot   = "fv-status-green" if torch.cuda.is_available() else "fv-status-yellow"
    st.markdown(f"""
<div class="fv-card" style="text-align:center;padding:1rem;margin-top:0.2rem">
  <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.1em">Device</div>
  <div style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#0f172a;margin:0.2rem 0">
    <span class="fv-status-dot {dev_dot}"></span>{dev_label}
  </div>
  <div style="font-size:0.65rem;color:#94a3b8;margin-top:0.4rem">{os.path.basename(model_path)}</div>
  <div style="font-size:0.6rem;color:#cbd5e1">✅ Model ready</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_infer, tab_batch, tab_about = st.tabs(["🔍  Single Inference", "📂  Batch Analysis", "📖  About & Architecture"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    up_col, _ = st.columns([2, 1])
    with up_col:
        uploaded = st.file_uploader(
            "Upload a face image (JPG · PNG · WEBP)",
            type=["jpg","jpeg","png","webp"],
        )

    if uploaded:
        raw_img  = Image.open(uploaded).convert("RGB")
        proc_img = apply_preprocessing(raw_img, brightness, contrast, sharpen, denoise)

        # Run inference
        t0 = time.perf_counter()
        if use_tta:
            top_results, all_probs = run_tta(model, device, proc_img, n_aug=tta_n)
        else:
            top_results, all_probs = predict(model, device, proc_img,
                                             temperature=temperature, topk=topk)
        latency = (time.perf_counter() - t0) * 1000

        top1     = top_results[0]
        conf     = top1["prob"]
        ent      = entropy(all_probs)
        max_ent  = entropy(np.ones(9)/9)
        uncert   = ent / max_ent

        # ── Row 1: image + summary ────────────────────────────────────────────
        img_col, res_col = st.columns([1, 1.7], gap="large")

        with img_col:
            st.markdown('<div class="fv-card fv-card-accent">', unsafe_allow_html=True)
            st.image(proc_img, caption="Input (preprocessed)", use_container_width=True)
            w, h = proc_img.size
            st.markdown(f"""
<div class="stat-grid">
  <div class="stat-item">
    <div class="stat-val">{w}×{h}</div>
    <div class="stat-lbl">Size</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{"TTA" if use_tta else "STD"}</div>
    <div class="stat-lbl">Mode</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{latency:.0f}ms</div>
    <div class="stat-lbl">Latency</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{temperature:.1f}</div>
    <div class="stat-lbl">Temp</div>
  </div>
</div>
""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col:
            # Gauge + metrics
            g_col, m_col = st.columns([1, 1.3])
            with g_col:
                gauge_fig = make_gauge(conf)
                st.pyplot(gauge_fig, use_container_width=True)

            with m_col:
                st.metric("🏆 Top Prediction", f"{top1['icon']} {top1['label']}")
                conf_label = "High ✅" if conf > 0.6 else "Medium ⚠️" if conf > 0.35 else "Low ❌"
                st.metric("Confidence", f"{conf*100:.1f}%", delta=conf_label)
                unc_label  = "Low ✅" if uncert < 0.4 else "High ⚠️"
                st.metric("Uncertainty", f"{uncert*100:.1f}%", delta=unc_label,
                          delta_color="normal" if uncert < 0.4 else "inverse")

            st.markdown("---")

            # Uncertainty bar
            st.markdown("**Prediction Uncertainty**")
            ent_fig = make_entropy_bar(ent, max_ent)
            st.pyplot(ent_fig, use_container_width=True)

            st.markdown("---")

            # Top-K progress bars
            st.markdown(f"**Top-{len(top_results)} Predictions**")
            for rank, r in enumerate(top_results):
                icon  = RANK_ICONS[rank]
                label = f"{icon} {r['icon']} **{r['label']}**"
                st.progress(float(r["prob"]), text=f"{label} — `{r['prob']*100:.2f}%`")

        # ── Row 2: charts ─────────────────────────────────────────────────────
        st.markdown("---")
        ch1, ch2, ch3 = st.columns([2.2, 1.5, 1.3])

        with ch1:
            dist_fig = make_distribution_bar(all_probs)
            st.pyplot(dist_fig, use_container_width=True)

        with ch2:
            hbar_fig = make_horizontal_bars(top_results[:3])
            st.pyplot(hbar_fig, use_container_width=True)

        with ch3:
            radar_fig = make_radar(all_probs)
            st.pyplot(radar_fig, use_container_width=True)

        # ── Expanders ────────────────────────────────────────────────────────
        with st.expander("🔢 Raw Probability Table"):
            import pandas as pd
            order = np.argsort(all_probs)[::-1]
            df_raw = pd.DataFrame({
                "Rank":            [int(np.where(order == i)[0][0]+1) for i in range(9)],
                "Age Group":       AGE_NAMES,
                "Icon":            AGE_ICONS,
                "Probability (%)": [f"{p*100:.4f}" for p in all_probs],
            }).sort_values("Rank").reset_index(drop=True)
            st.dataframe(df_raw, use_container_width=True, hide_index=True)

        if brightness != 1.0 or contrast != 1.0 or sharpen or denoise:
            with st.expander("🖼️ Original vs Preprocessed"):
                c1, c2 = st.columns(2)
                c1.image(raw_img,  caption="Original",      use_container_width=True)
                c2.image(proc_img, caption="Preprocessed",  use_container_width=True)

    else:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">👁</div>
  <div class="empty-title">Upload a face image to begin</div>
  <div class="empty-sub">Supports JPG · PNG · WEBP · Clear forward-facing photos work best</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("Upload **multiple** face images for bulk inference and summary statistics.")

    batch_files = st.file_uploader(
        "Drop multiple images",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        key="batch_up",
    )

    if batch_files:
        if st.button("▶ Run Batch Inference", use_container_width=False):
            import pandas as pd
            results_list = []
            prog = st.progress(0, text="Initialising …")

            for idx, f in enumerate(batch_files):
                img = Image.open(f).convert("RGB")
                img = apply_preprocessing(img, brightness, contrast, sharpen, denoise)
                t0  = time.perf_counter()
                top, probs = predict(model, device, img, temperature=temperature, topk=3)
                lat = (time.perf_counter() - t0) * 1000
                results_list.append({
                    "File":           f.name,
                    "Top-1":          top[0]["label"],
                    "Conf (%)":       f"{top[0]['prob']*100:.2f}",
                    "Top-2":          top[1]["label"],
                    "Top-3":          top[2]["label"],
                    "Entropy":        f"{entropy(probs):.3f}",
                    "Latency (ms)":   f"{lat:.1f}",
                })
                prog.progress((idx+1)/len(batch_files), text=f"Processing {f.name} …")

            prog.empty()
            df = pd.DataFrame(results_list)
            st.success(f"✅ Processed **{len(batch_files)}** images")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Summary metrics
            from collections import Counter
            top1_counts = Counter(df["Top-1"])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Images Processed", len(batch_files))
            mc2.metric("Most Predicted", top1_counts.most_common(1)[0][0])
            mc3.metric("Unique Classes", len(top1_counts))

            # Pie chart
            fig_p, ax_p = plt.subplots(figsize=(5, 4), facecolor=CHART_BG)
            ax_p.set_facecolor(CHART_BG)
            cmap_pie = plt.cm.Blues(np.linspace(0.35, 0.9, len(top1_counts)))
            wedges, texts, autotexts = ax_p.pie(
                top1_counts.values(), labels=top1_counts.keys(),
                colors=cmap_pie, autopct="%1.0f%%", startangle=90,
                wedgeprops=dict(edgecolor="#ffffff", linewidth=1.5)
            )
            for t in texts:     t.set_color(CHART_FG); t.set_fontsize(8)
            for t in autotexts: t.set_color("#ffffff"); t.set_fontsize(8); t.set_fontweight("bold")
            ax_p.set_title("Batch Top-1 Distribution", color=CHART_FG,
                           fontsize=10, fontweight="bold", pad=12)
            st.pyplot(fig_p, use_container_width=False)
    else:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">📂</div>
  <div class="empty-title">Upload multiple images for batch processing</div>
  <div class="empty-sub">Get summary statistics, latency, and distribution charts for your entire dataset</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    ac1, ac2 = st.columns(2, gap="large")

    with ac1:
        st.markdown("""
### Project Overview
**FairVision** is a CNN-based age group classification system developed for the
IJSE Certified AI & ML Engineer programme. It audits demographic fairness across
race and gender groups using the FairFace dataset.

### Architecture — BetterCNN_V2
| Layer | Detail |
|---|---|
| Block 1 | Conv 3→32, BN, ReLU ×2, MaxPool |
| Block 2 | Conv 32→64, BN, ReLU ×2, MaxPool |
| Block 3 | Conv 64→128, BN, ReLU ×2, MaxPool |
| Block 4 | Conv 128→256, BN, ReLU ×2, MaxPool |
| Block 5 | Conv 256→512, BN, ReLU, Dropout2d(0.2) |
| Pooling | AdaptiveAvgPool2d(1×1) |
| Head | 512→256→9, Dropout(0.45) |

### Training Setup
- **Optimizer:** AdamW with weight decay
- **Loss:** CrossEntropyLoss + label smoothing (0.1)
- **Sampler:** WeightedRandomSampler (√-inverse frequency)
- **Augmentation:** Flip, rotation ±5°, colour jitter
- **Input size:** 224 × 224
""")

    with ac2:
        st.markdown("""
### Dataset — FairFace 0.25
| Split | Samples |
|---|---|
| Train | 69,395 |
| Validation | 17,349 |
| Test | 10,954 |
| **Total** | **97,698** |

### Fairness Audit Groups
| Attribute | Role | Classes |
|---|---|---|
| Age | Target | 9 groups |
| Race | Audit | 7 groups |
| Gender | Audit | Male · Female |

### Inference Features
| Feature | Description |
|---|---|
| Temperature | Scale logits before softmax |
| TTA | Average over augmented views |
| Top-K | Show 1–9 ranked predictions |
| Uncertainty | Entropy-based confidence score |
| Preprocessing | Brightness / contrast / sharpen / denoise |
| Batch mode | Multi-image bulk inference |

### Limitations
- Age estimation is ambiguous at group boundaries
- Demographic performance gaps exist
- Not suitable for surveillance or biometric ID
""")

    st.markdown("---")
    st.caption("FairVision · IJSE – Certified AI & ML Engineer · Individual Assignment 2025/2026 · Built with PyTorch & Streamlit")
