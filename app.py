"""
================================================================================
 Universal ADRC Pro Tuner – Linear & Nonlinear Systems
================================================================================
 A high-end engineering application for automatic tuning of Active Disturbance
 Rejection Control (ADRC) covering both Linear ADRC (LADRC) and Nonlinear ADRC
 (NLADRC) for general SISO systems of arbitrary order.

 Author : Eng. HANI NARIMENE
 School : Higher School of Air Defense of the Territory – Martyr Ali Chabati
 Year   : 2026
 Stack  : Python 3 / Streamlit / NumPy / SciPy / Matplotlib

 Run with:
     streamlit run app.py
================================================================================
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import streamlit as st
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.special import comb


# ============================================================================ #
#                              GLOBAL STREAMLIT SETUP                           #
# ============================================================================ #

st.set_page_config(
    page_title="ADRC Pro Tuner | Eng. HANI NARIMENE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium Aerospace-Grade CSS ─────────────────────────────────────────────
PREMIUM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700;800&family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-base:        #04080f;
    --bg-panel:       #080d17;
    --bg-card:        #0c1221;
    --bg-card-hover:  #101929;
    --bg-surface:     #111c2e;
    --border-subtle:  #1a2a42;
    --border-glow:    #1e3a5f;
    --accent-blue:    #1a6fff;
    --accent-cyan:    #00c8ff;
    --accent-teal:    #00e8c8;
    --accent-amber:   #f5a623;
    --accent-red:     #ff4c4c;
    --accent-green:   #00e676;
    --text-primary:   #e8f0fe;
    --text-secondary: #90aac8;
    --text-muted:     #4a6280;
    --text-dim:       #2d4060;
    --grid-line:      rgba(26,111,255,0.07);
    --glow-blue:      rgba(26,111,255,0.25);
    --glow-cyan:      rgba(0,200,255,0.15);
    --shadow-card:    0 8px 32px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.03) inset;
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body { background: var(--bg-base) !important; }

/* ── Remove Streamlit Defaults ── */
.main > div { padding-top: 0 !important; }
.block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-panel); }
::-webkit-scrollbar-thumb {
    background: var(--border-glow);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

/* ── Typography ── */
body, .stMarkdown, .stText, p, span, div {
    font-family: 'Rajdhani', 'Segoe UI', sans-serif !important;
    color: var(--text-primary);
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Exo 2', sans-serif !important;
    letter-spacing: 0.03em;
}
code, pre, .stCode {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

/* ── HEADER BANNER ── */
.adrc-header {
    background: linear-gradient(135deg, #04080f 0%, #060e1f 40%, #080d1a 100%);
    border-bottom: 1px solid var(--border-glow);
    padding: 0;
    margin: 0 -2rem 1.5rem -2rem;
    position: relative;
    overflow: hidden;
}
.adrc-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        repeating-linear-gradient(90deg,
            transparent,
            transparent 80px,
            var(--grid-line) 80px,
            var(--grid-line) 81px),
        repeating-linear-gradient(0deg,
            transparent,
            transparent 40px,
            var(--grid-line) 40px,
            var(--grid-line) 41px);
    pointer-events: none;
}
.adrc-header::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%,
        var(--accent-blue) 20%,
        var(--accent-cyan) 50%,
        var(--accent-blue) 80%,
        transparent 100%);
}
.adrc-header-inner {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 2rem;
    gap: 1.5rem;
}
.adrc-logo-group {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.adrc-logo-icon {
    width: 54px;
    height: 54px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 0 20px var(--glow-blue), 0 0 40px rgba(26,111,255,0.1);
    flex-shrink: 0;
}
.adrc-title-group {}
.adrc-title-main {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0;
    line-height: 1.1;
    text-shadow: 0 0 30px rgba(0,200,255,0.4);
}
.adrc-title-sub {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.78rem;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 2px 0 0;
}
.adrc-badge-row {
    display: flex;
    gap: 6px;
    margin-top: 6px;
    flex-wrap: wrap;
}
.adrc-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: 1px solid;
}
.adrc-badge-blue {
    background: rgba(26,111,255,0.12);
    color: var(--accent-cyan);
    border-color: rgba(0,200,255,0.3);
}
.adrc-badge-teal {
    background: rgba(0,232,200,0.08);
    color: var(--accent-teal);
    border-color: rgba(0,232,200,0.25);
}
.adrc-badge-amber {
    background: rgba(245,166,35,0.1);
    color: var(--accent-amber);
    border-color: rgba(245,166,35,0.25);
}
.adrc-header-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
    min-width: 220px;
}
.adrc-institution {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-align: right;
    line-height: 1.4;
    max-width: 240px;
}
.adrc-institution strong {
    color: var(--accent-cyan);
    font-family: 'Exo 2', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
}
.adrc-status-row {
    display: flex;
    align-items: center;
    gap: 8px;
}
.adrc-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--accent-green); }
    50% { opacity: 0.6; box-shadow: 0 0 16px var(--accent-green); }
}
.adrc-status-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: var(--accent-green);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.adrc-year-badge {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    border: 1px solid var(--border-subtle);
    padding: 1px 8px;
    border-radius: 3px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050b18 0%, #06101c 100%) !important;
    border-right: 1px solid var(--border-glow) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
.sidebar-header-block {
    background: linear-gradient(135deg, #060e1f, #080d17);
    border-bottom: 1px solid var(--border-glow);
    padding: 1rem 1rem 0.8rem;
    margin: 0 -1rem 1rem;
    position: relative;
}
.sidebar-header-block::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
}
.sidebar-brand {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 2px;
}
.sidebar-subbrand {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sidebar-section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.55rem 0 0.4rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
}
.sidebar-section-header .icon {
    font-size: 0.9rem;
    width: 20px;
    text-align: center;
    opacity: 0.9;
}
.sidebar-section-header .label {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Sidebar footer */
.sidebar-footer {
    border-top: 1px solid var(--border-subtle);
    padding: 0.8rem 0 0;
    margin-top: 1rem;
    text-align: center;
}
.sidebar-footer-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    line-height: 1.6;
}

/* Streamlit widget overrides */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 5px !important;
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 5px !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(26,111,255,0.2) !important;
}

/* ── METRIC CARDS ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.2rem;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    box-shadow: var(--shadow-card);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--card-accent, linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)));
}
.kpi-card:hover {
    border-color: var(--accent-blue);
    box-shadow: 0 12px 40px rgba(0,0,0,0.7), 0 0 20px var(--glow-blue);
    transform: translateY(-2px);
}
.kpi-card-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.kpi-card-value {
    font-family: 'Exo 2', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.0;
    letter-spacing: -0.01em;
}
.kpi-card-unit {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 2px;
}
.kpi-card-icon {
    position: absolute;
    top: 0.8rem; right: 0.9rem;
    font-size: 1.1rem;
    opacity: 0.25;
}
.kpi-card-good { --card-accent: linear-gradient(90deg, #00e676, #00c853); }
.kpi-card-warn { --card-accent: linear-gradient(90deg, var(--accent-amber), #ff8c00); }
.kpi-card-info { --card-accent: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.kpi-card-neutral { --card-accent: linear-gradient(90deg, #546e7a, #78909c); }

.kpi-grid-2 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.5rem;
}

/* ── SECTION HEADERS ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 0 0.5rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border-subtle);
}
.section-header .sh-icon {
    font-size: 1rem;
    padding: 4px 8px;
    background: rgba(26,111,255,0.12);
    border-radius: 4px;
    border: 1px solid rgba(26,111,255,0.2);
}
.section-header .sh-label {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.section-header .sh-sub {
    margin-left: auto;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 8px;
    border: 1px solid var(--border-subtle);
    border-radius: 3px;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px !important;
    background: var(--bg-panel) !important;
    border-radius: 8px 8px 0 0;
    padding: 4px 4px 0;
    border: 1px solid var(--border-glow);
    border-bottom: none;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.45rem 0.9rem !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(26,111,255,0.06) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(26,111,255,0.2), rgba(0,200,255,0.1)) !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 0 8px 8px 8px !important;
    padding: 1.2rem !important;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 5px !important;
    border: 1px solid var(--accent-blue) !important;
    background: linear-gradient(135deg, rgba(26,111,255,0.15), rgba(0,200,255,0.05)) !important;
    color: var(--accent-cyan) !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 0 0 var(--glow-blue) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(26,111,255,0.3), rgba(0,200,255,0.15)) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 16px var(--glow-blue) !important;
    transform: translateY(-1px) !important;
    color: #fff !important;
}
.stDownloadButton > button {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 5px !important;
    border: 1px solid rgba(0,232,200,0.4) !important;
    background: rgba(0,232,200,0.08) !important;
    color: var(--accent-teal) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(0,232,200,0.18) !important;
    box-shadow: 0 0 14px rgba(0,232,200,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── SLIDERS ── */
.stSlider [data-baseweb="slider"] {
    margin-top: 0.2rem;
}
.stSlider [data-baseweb="thumb"] {
    background: var(--accent-blue) !important;
    border: 2px solid var(--accent-cyan) !important;
    width: 18px !important;
    height: 18px !important;
    box-shadow: 0 0 10px var(--glow-blue) !important;
}
.stSlider [data-baseweb="track"] {
    height: 4px !important;
    background: var(--border-glow) !important;
}
.stSlider [data-baseweb="track-fill"] {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}

/* ── METRICS (native Streamlit) ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    padding: 0.85rem 1rem !important;
    border-top: 2px solid var(--accent-blue) !important;
    box-shadow: var(--shadow-card) !important;
    transition: all 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 20px var(--glow-blue) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.62rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: var(--accent-cyan) !important;
}

/* ── CODE BLOCKS ── */
.stCode, [data-testid="stCode"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 6px !important;
}
pre, code {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #a8d8ea !important;
}

/* ── INFO / WARNING / ERROR ── */
.stAlert {
    border-radius: 6px !important;
    border-left: 3px solid !important;
    background: var(--bg-card) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.88rem !important;
}
[data-testid="stAlertWarning"] {
    border-color: var(--accent-amber) !important;
}
[data-testid="stAlertError"] {
    border-color: var(--accent-red) !important;
}
[data-testid="stAlertInfo"] {
    border-color: var(--accent-blue) !important;
}
[data-testid="stAlertSuccess"] {
    border-color: var(--accent-green) !important;
}

/* ── EXPANDERS ── */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 6px !important;
    margin-bottom: 0.5rem !important;
}
summary {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1rem !important;
    cursor: pointer !important;
}
details[open] summary {
    color: var(--accent-cyan) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* ── DIVIDERS ── */
hr, .stDivider {
    border-color: var(--border-glow) !important;
    margin: 0.8rem 0 !important;
}

/* ── CAPTIONS ── */
.stCaption, [data-testid="caption"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em !important;
}

/* ── SPINNERS ── */
.stSpinner > div > div {
    border-top-color: var(--accent-blue) !important;
    border-right-color: var(--accent-cyan) !important;
}

/* ── FOOTER ── */
.adrc-footer {
    background: linear-gradient(135deg, #04080f, #060e1f);
    border-top: 1px solid var(--border-glow);
    padding: 1rem 2rem;
    margin: 2rem -2rem -3rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    position: relative;
}
.adrc-footer::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), var(--accent-blue), transparent);
}
.adrc-footer-left {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    line-height: 1.7;
}
.adrc-footer-left strong {
    color: var(--text-muted);
}
.adrc-footer-center {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: center;
}
.adrc-footer-right {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-align: right;
    line-height: 1.7;
}

/* ── STABILITY INDICATOR ── */
.stability-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0.4rem 1rem;
    border-radius: 5px;
    font-family: 'Exo 2', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.stability-stable {
    background: rgba(0,230,118,0.1);
    border: 1px solid rgba(0,230,118,0.35);
    color: var(--accent-green);
    box-shadow: 0 0 16px rgba(0,230,118,0.1);
}
.stability-unstable {
    background: rgba(255,76,76,0.1);
    border: 1px solid rgba(255,76,76,0.35);
    color: var(--accent-red);
    box-shadow: 0 0 16px rgba(255,76,76,0.1);
}

/* ── WELCOME CARD ── */
.welcome-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-surface) 100%);
    border: 1px solid var(--border-glow);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.welcome-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent-blue), var(--accent-cyan));
}
.welcome-card::after {
    content: 'ADRC';
    position: absolute;
    top: 50%; right: 1.5rem;
    transform: translateY(-50%);
    font-family: 'Exo 2', sans-serif;
    font-size: 4rem;
    font-weight: 900;
    color: rgba(26,111,255,0.04);
    letter-spacing: 0.3em;
    pointer-events: none;
    user-select: none;
}
.welcome-title {
    font-family: 'Exo 2', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.welcome-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.welcome-author {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-cyan);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ── COMPARISON TABLE ── */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
}
.compare-table th {
    background: rgba(26,111,255,0.12);
    color: var(--accent-cyan);
    padding: 0.5rem 0.8rem;
    border: 1px solid var(--border-glow);
    text-align: left;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.72rem;
}
.compare-table td {
    padding: 0.45rem 0.8rem;
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    vertical-align: middle;
}
.compare-table tr:nth-child(even) td {
    background: rgba(255,255,255,0.015);
}
.compare-table tr:hover td {
    background: rgba(26,111,255,0.06);
}
.metric-better { color: var(--accent-green); font-weight: 600; }
.metric-worse  { color: var(--accent-red);   font-weight: 600; }

/* ── RADIO ── */
[data-baseweb="radio"] label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.82rem !important;
}

/* ── SELECT BOX ── */
[data-baseweb="select"] {
    border-radius: 5px !important;
}

/* ── CHECKBOX ── */
[data-baseweb="checkbox"] label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.82rem !important;
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ─── Matplotlib Aerospace Theme ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#04080f",
    "axes.facecolor":    "#080d17",
    "axes.edgecolor":    "#1a2a42",
    "axes.labelcolor":   "#90aac8",
    "axes.titlecolor":   "#e8f0fe",
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#4a6280",
    "ytick.color":       "#4a6280",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "grid.color":        "#0f1e30",
    "grid.linestyle":    "-",
    "grid.alpha":        1.0,
    "text.color":        "#e8f0fe",
    "legend.facecolor":  "#080d17",
    "legend.edgecolor":  "#1a2a42",
    "legend.labelcolor": "#90aac8",
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "font.family":       "DejaVu Sans",
    "lines.linewidth":   1.8,
    "lines.antialiased": True,
    "patch.antialiased": True,
    "figure.dpi":        130,
})


# ============================================================================ #
#                          1.  SYSTEM PARSING UTILITIES                         #
# ============================================================================ #

@dataclass
class PlantModel:
    """
    Generic plant container. Supports linear (TF, SS) and user-defined nonlinear
    dynamics. The unified interface is `dynamics(t, x, u)` which returns x_dot.
    """
    kind: str
    order: int
    b0: float
    dynamics: Callable[[float, np.ndarray, float], np.ndarray]
    output: Callable[[np.ndarray], float]
    description: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


def build_from_tf(num: List[float], den: List[float]) -> PlantModel:
    num = np.array(num, dtype=float).flatten()
    den = np.array(den, dtype=float).flatten()
    if den[0] == 0:
        raise ValueError("Leading denominator coefficient cannot be zero.")
    num = num / den[0]
    den = den / den[0]
    n = len(den) - 1
    if n < 1:
        raise ValueError("System order must be ≥ 1.")
    if len(num) < n + 1:
        num = np.concatenate([np.zeros(n + 1 - len(num)), num])
    A = np.zeros((n, n))
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]
    B = np.zeros((n, 1)); B[-1, 0] = 1.0
    b0_term = num[0]
    C = (num[1:][::-1] - den[1:][::-1] * b0_term).reshape(1, n)
    D = np.array([[b0_term]])
    b0 = num[-1] if abs(num[-1]) > 1e-12 else float(np.max(np.abs(num)) or 1.0)

    def dyn(t, x, u, A=A, B=B):
        return (A @ x + B.flatten() * u)

    def out(x, C=C, D=D):
        return float(np.asarray(C @ x).flatten()[0])

    desc = f"TF order {n}, num={num.tolist()}, den={den.tolist()}"
    return PlantModel(kind="tf", order=n, b0=float(b0),
                      dynamics=dyn, output=out, description=desc,
                      extras={"A": A, "B": B, "C": C, "D": D,
                              "num": num, "den": den})


def build_from_ss(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                  b0_hint: Optional[float] = None) -> PlantModel:
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A must be square.")
    if B.shape[0] != n:
        raise ValueError("B row count must match A.")
    B = B[:, :1]; C = C[:1, :]; D = D[:1, :1]
    if b0_hint is not None and b0_hint != 0:
        b0 = float(b0_hint)
    else:
        b0_est = 0.0
        Ak = np.eye(n)
        for k in range(n):
            val = float((C @ Ak @ B).flatten()[0])
            if abs(val) > 1e-9:
                b0_est = val
                break
            Ak = Ak @ A
        b0 = b0_est if abs(b0_est) > 1e-9 else 1.0

    def dyn(t, x, u, A=A, B=B):
        return (A @ x + B.flatten() * u)

    def out(x, C=C):
        return float(np.asarray(C @ x).flatten()[0])

    desc = f"SS n={n}"
    return PlantModel(kind="ss", order=n, b0=b0,
                      dynamics=dyn, output=out, description=desc,
                      extras={"A": A, "B": B, "C": C, "D": D})


def build_from_nonlinear(f_expr: str, n: int, b0: float,
                         output_index: int = 0) -> PlantModel:
    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "round": round, "pow": pow, "float": float,
        "int": int, "list": list, "tuple": tuple, "True": True, "False": False,
    }
    safe_globals = {
        "__builtins__": safe_builtins,
        "np": np, "math": math,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
        "sign": np.sign, "tanh": np.tanh, "pi": np.pi,
    }
    body = "\n".join("    " + line for line in f_expr.splitlines())
    src = f"def _user_f(x, u, t):\n{body}\n"
    code = compile(src, "<user-nl>", "exec")
    local_ns: Dict[str, Any] = {}
    exec(code, safe_globals, local_ns)
    user_f = local_ns["_user_f"]

    def dyn(t, x, u, _f=user_f):
        out = _f(np.asarray(x, dtype=float), float(u), float(t))
        out = np.asarray(out, dtype=float).flatten()
        if out.shape[0] != n:
            raise ValueError(
                f"User dynamics returned vector of length {out.shape[0]}, expected {n}.")
        return out

    def out_fn(x, idx=output_index):
        return float(np.asarray(x).flatten()[idx])

    return PlantModel(kind="nonlinear", order=n, b0=float(b0),
                      dynamics=dyn, output=out_fn,
                      description=f"User nonlinear ODE, n={n}",
                      extras={"expr": f_expr, "output_index": output_index})


# ============================================================================ #
#                           2.  ADRC DESIGN ENGINE                              #
# ============================================================================ #

@dataclass
class ADRCParams:
    n: int
    b0: float
    omega_c: float
    omega_o: float
    K: np.ndarray
    L: np.ndarray
    A_eso: np.ndarray
    B_eso: np.ndarray
    E_eso: np.ndarray
    is_nonlinear: bool = False
    alpha: List[float] = field(default_factory=list)
    delta: float = 0.01
    r_td: float = 100.0
    h_td: float = 0.01


def _binomial_pole_placement(n_states: int, omega: float) -> np.ndarray:
    coeffs = np.array([comb(n_states, k) * (omega ** k) for k in range(n_states + 1)])
    return coeffs[1:]


def design_ladrc(plant: PlantModel, omega_c: float, k_obs: float = 5.0) -> ADRCParams:
    n = plant.order
    b0 = plant.b0 if abs(plant.b0) > 1e-9 else 1.0
    omega_o = k_obs * omega_c
    Kfb = np.array([comb(n, k) * (omega_c ** (n - k)) for k in range(n)])
    Aeso = np.zeros((n + 1, n + 1))
    Aeso[:-1, 1:] = np.eye(n)
    Beso = np.zeros((n + 1, 1)); Beso[n - 1, 0] = b0
    Ceso = np.zeros((1, n + 1)); Ceso[0, 0] = 1.0
    L = np.array([comb(n + 1, i) * (omega_o ** i) for i in range(1, n + 2)],
                 dtype=float).reshape(-1, 1)
    return ADRCParams(
        n=n, b0=b0, omega_c=omega_c, omega_o=omega_o,
        K=Kfb, L=L, A_eso=Aeso, B_eso=Beso, E_eso=L,
        is_nonlinear=False,
    )


def design_nladrc(plant: PlantModel, omega_c: float, k_obs: float = 5.0,
                  alpha: Optional[List[float]] = None, delta: float = 0.01,
                  r_td: float = 100.0) -> ADRCParams:
    base = design_ladrc(plant, omega_c, k_obs)
    n = plant.order
    if alpha is None:
        alpha = [0.5 ** (i + 1) for i in range(n + 1)]
    base.is_nonlinear = True
    base.alpha = alpha
    base.delta = float(delta)
    base.r_td = float(r_td)
    base.h_td = 1e-3
    return base


def fal(e: float, alpha: float, delta: float) -> float:
    if abs(e) <= delta:
        return e / (delta ** (1.0 - alpha))
    return (abs(e) ** alpha) * np.sign(e)


def auto_select_omega_c(plant: PlantModel, objective: str) -> float:
    try:
        if plant.kind in ("tf", "ss"):
            A = plant.extras["A"]
            eigs = np.linalg.eigvals(A)
            real_parts = np.abs(eigs.real)
            stable = real_parts[eigs.real < 0]
            scale = float(stable.max()) if stable.size else float(np.abs(eigs).max())
            scale = max(scale, 0.5)
        else:
            scale = 1.0
    except Exception:
        scale = 1.0
    base = max(scale, 0.5)
    return {
        "Fast response":            8.0 * base,
        "Balanced":                 4.0 * base,
        "Robust / disturbance rej": 2.5 * base,
        "Noise attenuation":        1.5 * base,
    }.get(objective, 4.0 * base)


def auto_select_kobs(objective: str) -> float:
    return {
        "Fast response":            8.0,
        "Balanced":                 5.0,
        "Robust / disturbance rej": 6.0,
        "Noise attenuation":        3.0,
    }.get(objective, 5.0)


def stability_check(params: ADRCParams) -> Dict[str, Any]:
    n = params.n
    cl_poles_ctrl = np.full(n, -params.omega_c)
    eso_eig = np.linalg.eigvals(params.A_eso - params.E_eso @ np.array([[1] + [0]*n]))
    return {
        "controller_poles": cl_poles_ctrl,
        "observer_poles": eso_eig,
        "stable": bool(np.all(cl_poles_ctrl.real < 0) and np.all(eso_eig.real < 0)),
    }


# ============================================================================ #
#                            3.  SIMULATION ENGINE                              #
# ============================================================================ #

@dataclass
class SimResult:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    r: np.ndarray
    z_hat: np.ndarray
    disturbance: np.ndarray
    label: str = "ADRC"


def make_reference(t: np.ndarray, kind: str = "step",
                   amp: float = 1.0, period: float = 5.0) -> np.ndarray:
    if kind == "step":
        return np.where(t >= 0.0, amp, 0.0)
    if kind == "ramp":
        return amp * np.clip(t / max(period * 0.5, 1e-6), 0, 1)
    if kind == "sine":
        return amp * np.sin(2 * np.pi * t / max(period, 1e-6))
    if kind == "square":
        return amp * np.sign(np.sin(2 * np.pi * t / max(period, 1e-6)))
    return np.where(t >= 0.0, amp, 0.0)


def make_disturbance(t: np.ndarray, kind: str, amp: float, t_on: float) -> np.ndarray:
    if kind == "none":
        return np.zeros_like(t)
    if kind == "step":
        return np.where(t >= t_on, amp, 0.0)
    if kind == "sine":
        return np.where(t >= t_on, amp * np.sin(2 * np.pi * 0.5 * (t - t_on)), 0.0)
    if kind == "pulse":
        return np.where((t >= t_on) & (t < t_on + 0.5), amp, 0.0)
    return np.zeros_like(t)


def simulate_adrc(plant: PlantModel, params: ADRCParams,
                  t_end: float = 10.0, dt: float = 0.001,
                  ref_kind: str = "step", ref_amp: float = 1.0,
                  ref_period: float = 5.0, dist_kind: str = "none",
                  dist_amp: float = 0.0, dist_t_on: float = 3.0,
                  noise_std: float = 0.0,
                  u_min: float = -1e6, u_max: float = 1e6) -> SimResult:
    rng = np.random.default_rng(seed=42)
    t = np.arange(0.0, t_end + dt, dt)
    N = len(t)
    n = plant.order
    x = np.zeros(n)
    z = np.zeros(n + 1)
    r_arr = make_reference(t, ref_kind, ref_amp, ref_period)
    d_arr = make_disturbance(t, dist_kind, dist_amp, dist_t_on)
    y_arr = np.zeros(N)
    u_arr = np.zeros(N)
    z_arr = np.zeros((N, n + 1))
    Kfb = params.K
    L = params.L.flatten()
    b0 = params.b0
    is_nl = params.is_nonlinear
    alpha = params.alpha if is_nl else None
    delta = params.delta

    def control_law(r, z_est):
        e_vec = np.zeros(n)
        e_vec[0] = r - z_est[0]
        for i in range(1, n):
            e_vec[i] = -z_est[i]
        if is_nl:
            u0 = 0.0
            for i in range(n):
                a_i = alpha[i] if i < len(alpha) else 0.5
                u0 += Kfb[i] * fal(e_vec[i], a_i, delta)
        else:
            u0 = float(Kfb @ e_vec)
        u = (u0 - z_est[n]) / b0
        return float(np.clip(u, u_min, u_max))

    def plant_rhs(ti, xi, ui, di):
        return plant.dynamics(ti, xi, ui + di)

    def eso_rhs(zi, ui, yi, A_eso=params.A_eso, B_eso=params.B_eso.flatten(), L=L):
        return A_eso @ zi + B_eso * ui + L * (yi - zi[0])

    for k in range(N):
        y_meas = plant.output(x) + (rng.normal(0.0, noise_std) if noise_std > 0 else 0.0)
        u_k = control_law(r_arr[k], z)
        y_arr[k] = plant.output(x)
        u_arr[k] = u_k
        z_arr[k, :] = z
        if k == N - 1:
            break
        d_k = d_arr[k]
        d_kh = d_arr[min(k + 1, N - 1)] * 0.5 + d_arr[k] * 0.5
        k1 = plant_rhs(t[k],          x,                   u_k, d_k)
        k2 = plant_rhs(t[k] + dt / 2, x + dt / 2 * k1,     u_k, d_kh)
        k3 = plant_rhs(t[k] + dt / 2, x + dt / 2 * k2,     u_k, d_kh)
        k4 = plant_rhs(t[k] + dt,     x + dt * k3,         u_k, d_arr[min(k + 1, N - 1)])
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        y_at_k = y_meas
        m1 = eso_rhs(z,                 u_k, y_at_k)
        m2 = eso_rhs(z + dt / 2 * m1,   u_k, y_at_k)
        m3 = eso_rhs(z + dt / 2 * m2,   u_k, y_at_k)
        m4 = eso_rhs(z + dt * m3,       u_k, y_at_k)
        z = z + dt / 6.0 * (m1 + 2 * m2 + 2 * m3 + m4)

    return SimResult(t=t, y=y_arr, u=u_arr, r=r_arr,
                     z_hat=z_arr, disturbance=d_arr, label="ADRC")


# ============================================================================ #
#                       4.  BASELINE CONTROLLERS                                #
# ============================================================================ #

def simulate_pid(plant: PlantModel, Kp: float, Ki: float, Kd: float,
                 t_end: float, dt: float,
                 ref_kind: str, ref_amp: float, ref_period: float,
                 dist_kind: str, dist_amp: float, dist_t_on: float,
                 noise_std: float = 0.0,
                 u_min: float = -1e6, u_max: float = 1e6) -> SimResult:
    rng = np.random.default_rng(seed=42)
    t = np.arange(0.0, t_end + dt, dt)
    N = len(t)
    n = plant.order
    x = np.zeros(n)
    r_arr = make_reference(t, ref_kind, ref_amp, ref_period)
    d_arr = make_disturbance(t, dist_kind, dist_amp, dist_t_on)
    y_arr = np.zeros(N); u_arr = np.zeros(N)
    integ = 0.0; y_prev = 0.0
    for k in range(N):
        y = plant.output(x) + (rng.normal(0.0, noise_std) if noise_std > 0 else 0.0)
        e = r_arr[k] - y
        deriv = -(y - y_prev) / dt
        u_unclipped = Kp * e + Ki * integ + Kd * deriv
        u = float(np.clip(u_unclipped, u_min, u_max))
        if u == u_unclipped:
            integ += e * dt
        y_arr[k] = plant.output(x); u_arr[k] = u
        if k == N - 1: break
        d_k = d_arr[k]
        k1 = plant.dynamics(t[k],          x,              u + d_k)
        k2 = plant.dynamics(t[k] + dt / 2, x + dt/2 * k1,  u + d_k)
        k3 = plant.dynamics(t[k] + dt / 2, x + dt/2 * k2,  u + d_k)
        k4 = plant.dynamics(t[k] + dt,     x + dt * k3,    u + d_arr[min(k+1, N-1)])
        x = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        y_prev = y
    z_dummy = np.zeros((N, plant.order + 1))
    return SimResult(t=t, y=y_arr, u=u_arr, r=r_arr,
                     z_hat=z_dummy, disturbance=d_arr, label="PID")


def auto_tune_pid(plant: PlantModel, omega_c: float) -> Tuple[float, float, float]:
    if plant.kind in ("tf", "ss"):
        try:
            A = plant.extras["A"]
            eigs = np.linalg.eigvals(A)
            tau = 1.0 / max(np.abs(eigs.real).min(), 0.1)
        except Exception:
            tau = 1.0
    else:
        tau = 1.0
    Kp = 1.5 * omega_c * tau / max(plant.b0, 1e-3)
    Ki = 0.4 * Kp / tau
    Kd = 0.1 * Kp * tau
    return Kp, Ki, Kd


# ============================================================================ #
#                       5.  PERFORMANCE METRICS                                 #
# ============================================================================ #

def performance_metrics(res: SimResult, settle_band: float = 0.02) -> Dict[str, float]:
    t, y, r, u = res.t, res.y, res.r, res.u
    r_final = r[-1] if abs(r[-1]) > 1e-9 else (np.max(r) if np.max(np.abs(r)) > 0 else 1.0)
    sign = np.sign(r_final) if r_final != 0 else 1.0
    target = abs(r_final)
    try:
        idx10 = np.argmax(sign * y >= 0.1 * target)
        idx90 = np.argmax(sign * y >= 0.9 * target)
        rise = float(t[idx90] - t[idx10]) if idx90 > idx10 else float("nan")
    except Exception:
        rise = float("nan")
    band = settle_band * max(abs(r_final), 1e-9)
    outside = np.where(np.abs(y - r) > band)[0]
    settle = float(t[outside[-1]]) if outside.size else float(t[0])
    if abs(r_final) > 1e-9:
        peak = np.max(sign * y)
        overshoot = max(0.0, (peak - target) / target) * 100.0
    else:
        overshoot = 0.0
    tail = max(int(0.1 * len(t)), 1)
    sse = float(np.mean(r[-tail:] - y[-tail:]))
    e = r - y
    ise = float(np.trapezoid(e ** 2, t))
    iae = float(np.trapezoid(np.abs(e), t))
    itae = float(np.trapezoid(t * np.abs(e), t))
    ctrl_energy = float(np.trapezoid(u ** 2, t))
    return {
        "rise_time_s": rise,
        "settling_time_s": settle,
        "overshoot_pct": overshoot,
        "steady_state_error": sse,
        "ISE": ise,
        "IAE": iae,
        "ITAE": itae,
        "control_energy": ctrl_energy,
    }


# ============================================================================ #
#                            6.  EXPORT UTILITIES                               #
# ============================================================================ #

def export_matlab(params: ADRCParams) -> str:
    lines = [
        "%% Universal ADRC Pro Tuner — MATLAB Export",
        "%% Author : Eng. HANI NARIMENE",
        "%% School : Higher School of Air Defense of the Territory – Martyr Ali Chabati",
        "%% Year   : 2026",
        "",
        f"n        = {params.n};",
        f"b0       = {params.b0:.10g};",
        f"omega_c  = {params.omega_c:.10g};",
        f"omega_o  = {params.omega_o:.10g};",
        f"K_fb     = {params.K.tolist()};   % state feedback gains",
        f"L        = {params.L.flatten().tolist()};   % ESO observer gains",
        "",
        "% Build ESO matrices (canonical chain of integrators + extended state)",
        "A_eso = diag(ones(n,1),1); A_eso(end+1,:) = 0; A_eso(end,end) = 0;",
        "B_eso = zeros(n+1,1); B_eso(n) = b0;",
        "C_eso = [1, zeros(1,n)];",
        "",
        "% Closed-loop simulation: implement in Simulink or with ode45.",
    ]
    if params.is_nonlinear:
        lines += [
            "% Nonlinear ADRC parameters:",
            f"alpha    = {params.alpha};",
            f"delta    = {params.delta};",
            f"r_td     = {params.r_td};",
        ]
    return "\n".join(lines)


def export_json(params: ADRCParams) -> str:
    payload = {
        "metadata": {
            "author": "Eng. HANI NARIMENE",
            "institution": "Higher School of Air Defense of the Territory – Martyr Ali Chabati",
            "year": 2026,
            "tool": "Universal ADRC Pro Tuner",
        },
        "n": params.n,
        "b0": params.b0,
        "omega_c": params.omega_c,
        "omega_o": params.omega_o,
        "K": params.K.tolist(),
        "L": params.L.flatten().tolist(),
        "is_nonlinear": params.is_nonlinear,
    }
    if params.is_nonlinear:
        payload.update({"alpha": params.alpha, "delta": params.delta, "r_td": params.r_td})
    return json.dumps(payload, indent=2)


def export_report(params: ADRCParams, plant: PlantModel, metrics: Dict[str, float]) -> str:
    lines = [
        "=" * 72,
        " UNIVERSAL ADRC PRO TUNER  —  ENGINEERING REPORT",
        " Author : Eng. HANI NARIMENE",
        " School : Higher School of Air Defense of the Territory – Martyr Ali Chabati",
        " Year   : 2026",
        "=" * 72,
        "",
        "[1] PLANT",
        f"    Type        : {plant.kind}",
        f"    Order       : {plant.order}",
        f"    b0 estimate : {plant.b0:.6g}",
        f"    Description : {plant.description}",
        "",
        "[2] CONTROLLER",
        f"    Mode             : {'Nonlinear ADRC' if params.is_nonlinear else 'Linear ADRC'}",
        f"    Controller BW ωc : {params.omega_c:.4g} rad/s",
        f"    Observer  BW  ωo : {params.omega_o:.4g} rad/s   (ratio = {params.omega_o/params.omega_c:.2f})",
        f"    State FB gains K : {np.array2string(params.K, precision=4)}",
        f"    Observer gains L : {np.array2string(params.L.flatten(), precision=4)}",
    ]
    if params.is_nonlinear:
        lines += [
            f"    α  exponents     : {params.alpha}",
            f"    δ  boundary      : {params.delta:.4g}",
        ]
    lines += ["", "[3] PERFORMANCE"]
    for k, v in metrics.items():
        lines.append(f"    {k:25s}: {v:.6g}")
    lines += ["", "=" * 72]
    return "\n".join(lines)


# ============================================================================ #
#                        7.  PREMIUM PLOTTING ENGINE                            #
# ============================================================================ #

_BLUE    = "#1a6fff"
_CYAN    = "#00c8ff"
_TEAL    = "#00e8c8"
_AMBER   = "#f5a623"
_RED     = "#ff4c4c"
_GREEN   = "#00e676"
_MUTED   = "#4a6280"
_GRAY    = "#2d4060"


def _apply_axis_style(ax, xlabel="", ylabel="", title="", grid=True):
    """Apply uniform aerospace styling to a matplotlib axis."""
    ax.set_facecolor("#060e1c")
    for spine in ax.spines.values():
        spine.set_color("#0f1e30")
    if xlabel:
        ax.set_xlabel(xlabel, color=_MUTED, fontsize=8.5, labelpad=6,
                      fontfamily="DejaVu Sans")
    if ylabel:
        ax.set_ylabel(ylabel, color=_MUTED, fontsize=8.5, labelpad=6,
                      fontfamily="DejaVu Sans")
    if title:
        ax.set_title(title, color="#c8d8f0", fontsize=10, fontweight="bold",
                     pad=10, fontfamily="DejaVu Sans")
    if grid:
        ax.grid(True, color="#0d1c2e", linewidth=0.7, alpha=1.0)
        ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelsize=7.5)


def fig_response(res: SimResult, title: str = "Closed-loop Response") -> plt.Figure:
    fig = plt.figure(figsize=(10, 5.8), facecolor="#04080f")
    fig.subplots_adjust(hspace=0.08, top=0.92, bottom=0.10, left=0.09, right=0.97)
    gs = GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.06, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # --- Output subplot ---
    ax0.fill_between(res.t, res.r, alpha=0.04, color=_BLUE)
    ax0.plot(res.t, res.r, color=_MUTED, lw=1.1, ls="--", label="Reference r(t)", dashes=(5, 3))
    ax0.plot(res.t, res.y, color=_CYAN, lw=2.0, label="Output  y(t)",
             solid_capstyle="round")
    # Fill between output and reference
    ax0.fill_between(res.t, res.r, res.y,
                     where=(res.y < res.r), alpha=0.08, color=_AMBER, interpolate=True)
    ax0.fill_between(res.t, res.r, res.y,
                     where=(res.y >= res.r), alpha=0.06, color=_CYAN, interpolate=True)
    _apply_axis_style(ax0, ylabel="Output  y(t)", title=title)
    leg0 = ax0.legend(loc="lower right", fontsize=8, framealpha=0.85,
                      edgecolor=_GRAY, facecolor="#060e1c")
    for text in leg0.get_texts():
        text.set_color("#90aac8")
    plt.setp(ax0.get_xticklabels(), visible=False)

    # --- Control subplot ---
    ax1.plot(res.t, res.u, color=_AMBER, lw=1.6, label="Control  u(t)",
             solid_capstyle="round")
    if np.any(res.disturbance != 0):
        ax1.plot(res.t, res.disturbance, color=_RED, lw=1.0, ls=":",
                 label="Disturbance d(t)", alpha=0.85)
    ax1.axhline(0, color=_GRAY, lw=0.5, alpha=0.5)
    _apply_axis_style(ax1, xlabel="Time  [s]", ylabel="Control  u(t)")
    leg1 = ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.85,
                      edgecolor=_GRAY, facecolor="#060e1c")
    for text in leg1.get_texts():
        text.set_color("#90aac8")

    # Top accent line
    fig.add_artist(plt.Line2D([0.09, 0.97], [0.96, 0.96],
                              transform=fig.transFigure,
                              color=_BLUE, lw=1.5, alpha=0.5))
    return fig


def fig_eso(res: SimResult, params: ADRCParams) -> plt.Figure:
    fig = plt.figure(figsize=(10, 5.8), facecolor="#04080f")
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.10, left=0.09, right=0.97)
    gs = GridSpec(2, 1, hspace=0.08, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # --- ESO tracking ---
    ax0.plot(res.t, res.y,           color=_MUTED,  lw=1.2, label="Plant output  y(t)")
    ax0.plot(res.t, res.z_hat[:, 0], color=_CYAN,   lw=1.8, ls="--",
             label="ESO estimate  ẑ₁(t)", dashes=(6, 2))
    _apply_axis_style(ax0, ylabel="Output / ESO Estimate",
                      title="Extended State Observer — Output Tracking")
    leg0 = ax0.legend(loc="lower right", fontsize=8, framealpha=0.85,
                      edgecolor=_GRAY, facecolor="#060e1c")
    for text in leg0.get_texts():
        text.set_color("#90aac8")
    plt.setp(ax0.get_xticklabels(), visible=False)

    # --- Disturbance estimation ---
    ax1.fill_between(res.t, res.z_hat[:, -1], alpha=0.07, color=_AMBER)
    ax1.plot(res.t, res.disturbance, color=_MUTED,  lw=1.0, ls="--",
             label="True disturbance  d(t)", dashes=(5, 3))
    ax1.plot(res.t, res.z_hat[:, -1], color=_AMBER, lw=1.8,
             label="ESO estimate  f̂(t)", solid_capstyle="round")
    ax1.axhline(0, color=_GRAY, lw=0.5, alpha=0.5)
    _apply_axis_style(ax1, xlabel="Time  [s]", ylabel="Disturbance")
    _ax1_title = ax1.set_title("Total Disturbance Estimation  f̂(t)",
                               color="#c8d8f0", fontsize=9.5, fontweight="bold",
                               pad=4, fontfamily="DejaVu Sans")
    leg1 = ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.85,
                      edgecolor=_GRAY, facecolor="#060e1c")
    for text in leg1.get_texts():
        text.set_color("#90aac8")
    return fig


def fig_compare(res_a: SimResult, res_b: SimResult) -> plt.Figure:
    fig = plt.figure(figsize=(10, 5.8), facecolor="#04080f")
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.10, left=0.09, right=0.97)
    gs = GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.08, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    ax0.plot(res_a.t, res_a.r,  color=_MUTED, lw=1.0, ls="--",
             label="Reference", dashes=(5, 3))
    ax0.plot(res_a.t, res_a.y,  color=_CYAN,  lw=2.0,  label=res_a.label,
             solid_capstyle="round")
    ax0.plot(res_b.t, res_b.y,  color=_AMBER, lw=1.8,  ls="-.",
             label=res_b.label, dash_capstyle="round")
    _apply_axis_style(ax0, ylabel="Output  y(t)", title="Controller Comparison — ADRC vs PID")
    leg0 = ax0.legend(fontsize=8, framealpha=0.85, edgecolor=_GRAY, facecolor="#060e1c")
    for text in leg0.get_texts():
        text.set_color("#90aac8")
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax1.plot(res_a.t, res_a.u, color=_CYAN,  lw=1.6, label=f"u — {res_a.label}",
             solid_capstyle="round")
    ax1.plot(res_b.t, res_b.u, color=_AMBER, lw=1.4, ls="-.",
             label=f"u — {res_b.label}")
    ax1.axhline(0, color=_GRAY, lw=0.5, alpha=0.5)
    _apply_axis_style(ax1, xlabel="Time  [s]", ylabel="Control  u(t)")
    leg1 = ax1.legend(fontsize=7.5, framealpha=0.85, edgecolor=_GRAY, facecolor="#060e1c")
    for text in leg1.get_texts():
        text.set_color("#90aac8")
    return fig


def fig_pole_map(st_info: Dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="#04080f")
    ax.set_facecolor("#060e1c")
    ax.axhline(0, color=_GRAY, lw=0.8, alpha=0.7)
    ax.axvline(0, color=_GRAY, lw=0.8, alpha=0.7)
    # Left half-plane shade
    xlim_left = min(np.concatenate([st_info["controller_poles"].real,
                                     st_info["observer_poles"].real])) * 1.3
    ax.axvspan(xlim_left * 1.5, 0, alpha=0.04, color=_GREEN)
    ax.axvspan(0, abs(xlim_left) * 0.2, alpha=0.04, color=_RED)

    ax.scatter(st_info["controller_poles"].real, st_info["controller_poles"].imag,
               marker="x", s=100, color=_CYAN, linewidths=2.0,
               label="Controller poles", zorder=5)
    ax.scatter(st_info["observer_poles"].real, st_info["observer_poles"].imag,
               marker="o", s=70, facecolors="none", edgecolors=_AMBER,
               linewidths=1.8, label="Observer (ESO) poles", zorder=5)
    _apply_axis_style(ax, xlabel="Real Axis  [Re(s)]", ylabel="Imaginary Axis  [Im(s)]",
                      title="Closed-Loop Pole Map")
    leg = ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=_GRAY, facecolor="#060e1c")
    for text in leg.get_texts():
        text.set_color("#90aac8")
    fig.tight_layout(pad=1.2)
    return fig


# ============================================================================ #
#                               8.  INPUT PARSERS                               #
# ============================================================================ #

def parse_vector(text: str) -> List[float]:
    text = text.strip().replace(",", " ").replace(";", " ")
    return [float(tok) for tok in text.split() if tok]


def parse_matrix(text: str, n: int) -> np.ndarray:
    rows = [r for r in text.strip().splitlines() if r.strip()]
    if len(rows) != n:
        raise ValueError(f"Expected {n} rows, got {len(rows)}.")
    M = []
    for r in rows:
        toks = r.replace(",", " ").replace(";", " ").split()
        if len(toks) != n:
            raise ValueError(f"Each row must have {n} columns.")
        M.append([float(x) for x in toks])
    return np.array(M, dtype=float)


# ============================================================================ #
#                            9.  STREAMLIT UI                                   #
# ============================================================================ #

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="adrc-header">
  <div class="adrc-header-inner">
    <div class="adrc-logo-group">
      <div class="adrc-logo-icon">🛡️</div>
      <div class="adrc-title-group">
        <div class="adrc-title-main">Universal ADRC Pro Tuner</div>
        <div class="adrc-title-sub">Active Disturbance Rejection Control — Engineering Platform</div>
        <div class="adrc-badge-row">
          <span class="adrc-badge adrc-badge-blue">LADRC</span>
          <span class="adrc-badge adrc-badge-blue">NLADRC</span>
          <span class="adrc-badge adrc-badge-teal">Auto-Tuning</span>
          <span class="adrc-badge adrc-badge-amber">ESO Observer</span>
        </div>
      </div>
    </div>
    <div class="adrc-header-right">
      <div class="adrc-institution">
        <strong>Eng. HANI NARIMENE</strong><br>
        Higher School of Air Defense<br>of the Territory<br>
        Martyr Ali Chabati
      </div>
      <div class="adrc-status-row">
        <span class="adrc-status-dot"></span>
        <span class="adrc-status-text">System Online</span>
        <span class="adrc-year-badge">2026</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="sidebar-header-block">
      <div class="sidebar-brand">🛡️ ADRC Pro Tuner</div>
      <div class="sidebar-subbrand">Eng. HANI NARIMENE · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Plant Definition ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-section-header">
      <span class="icon">📐</span>
      <span class="label">Plant Model</span>
    </div>
    """, unsafe_allow_html=True)

    sys_type = st.selectbox(
        "Model type",
        ["Linear — Transfer Function", "Linear — State-Space",
         "Nonlinear — Custom ODE"],
        help="Select the mathematical representation of the controlled plant."
    )

    plant: Optional[PlantModel] = None
    parse_error: Optional[str] = None

    try:
        if sys_type == "Linear — Transfer Function":
            num_str = st.text_input("Numerator  (high → low order)", "1",
                                    help="Polynomial coefficients of the numerator, space-separated.")
            den_str = st.text_input("Denominator  (high → low order)", "1 2 1",
                                    help="Polynomial coefficients of the denominator, space-separated.")
            num = parse_vector(num_str)
            den = parse_vector(den_str)
            plant = build_from_tf(num, den)

        elif sys_type == "Linear — State-Space":
            n_user = st.number_input("State dimension  n", 1, 10, 2, 1)
            A_str = st.text_area("Matrix  A  (n×n)", "0 1\n-1 -2", height=80,
                                 help="Rows separated by newlines, columns by spaces.")
            B_str = st.text_area("Matrix  B  (n×1)", "0\n1", height=65)
            C_str = st.text_input("Matrix  C  (1×n)", "1 0")
            D_str = st.text_input("Matrix  D  (scalar)", "0")
            b0_hint = st.number_input("b₀ hint  (0 = auto)", value=0.0, format="%.6f",
                                      help="Direct estimate of input-to-output gain. Leave 0 for automatic estimation.")
            A = parse_matrix(A_str, int(n_user))
            B = np.array(parse_vector(B_str)).reshape(-1, 1)
            C = np.array(parse_vector(C_str)).reshape(1, -1)
            D = np.array([[float(D_str.strip() or 0.0)]])
            plant = build_from_ss(A, B, C, D, b0_hint or None)

        else:
            n_user = st.number_input("State dimension  n", 1, 10, 2, 1)
            b0_user = st.number_input("Input gain estimate  b₀", value=1.0, format="%.6f",
                                      help="Approximate input-to-highest-derivative gain.")
            f_default = (
                "# x: ndarray[n], u: scalar, t: scalar → return list[n]\n"
                "# Example: Van der Pol oscillator (mu=1)\n"
                "mu = 1.0\n"
                "dx0 = x[1]\n"
                "dx1 = mu*(1 - x[0]**2)*x[1] - x[0] + u\n"
                "return [dx0, dx1]"
            )
            f_expr = st.text_area("Define  ẋ = f(x, u, t)", f_default, height=160,
                                  help="Python expression. Use np for NumPy. Return list of length n.")
            out_idx = st.number_input("Output state index  (y = xₖ)",
                                      0, int(n_user) - 1, 0, 1)
            plant = build_from_nonlinear(f_expr, int(n_user), float(b0_user), int(out_idx))
    except Exception as e:
        parse_error = str(e)

    st.markdown("<hr style='border-color:#1a2a42; margin:0.6rem 0;'>", unsafe_allow_html=True)

    # ── ADRC Settings ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-section-header">
      <span class="icon">🎛️</span>
      <span class="label">ADRC Configuration</span>
    </div>
    """, unsafe_allow_html=True)

    adrc_type = st.radio("Controller mode",
                         ["Linear ADRC (LADRC)", "Nonlinear ADRC (NLADRC)"], index=0)
    objective = st.selectbox(
        "Tuning objective",
        ["Fast response", "Balanced", "Robust / disturbance rej", "Noise attenuation"],
        index=1,
        help="Drives the automatic bandwidth selection heuristic."
    )
    auto_tune = st.checkbox("🤖 Auto-tune bandwidths", value=True,
                             help="Automatically select ωc and k from plant eigenvalues and objective.")

    if plant and not parse_error:
        wc_default = auto_select_omega_c(plant, objective)
        kobs_default = auto_select_kobs(objective)
    else:
        wc_default, kobs_default = 5.0, 5.0

    omega_c = st.slider("Controller bandwidth  ωc  [rad/s]",
                        0.1, 200.0, float(np.clip(wc_default, 0.1, 200.0)),
                        step=0.1, disabled=auto_tune)
    k_obs = st.slider("Bandwidth ratio  k = ωo / ωc",
                      2.0, 12.0, float(np.clip(kobs_default, 2.0, 12.0)),
                      step=0.5, disabled=auto_tune)

    if auto_tune and plant and not parse_error:
        omega_c = wc_default
        k_obs   = kobs_default
        st.caption(f"AUTO › ωc = **{omega_c:.3g}** rad/s   k = **{k_obs:.2g}**")

    with st.expander("🔬 NLADRC Parameters", expanded=adrc_type.startswith("Non")):
        alpha_str = st.text_input("α exponents  (comma-separated, length n+1)", "",
                                  help="Fractional exponents for fal(). Defaults to Han's geometric ladder.")
        delta_nl = st.number_input("δ  boundary layer", 1e-4, 1.0, 0.01, format="%.4f",
                                   help="Transition point between linear and fractional zones.")
        r_td = st.number_input("TD speed factor  r", 1.0, 10000.0, 200.0, 10.0,
                               help="Tracking differentiator speed parameter.")

    st.markdown("<hr style='border-color:#1a2a42; margin:0.6rem 0;'>", unsafe_allow_html=True)

    # ── Scenario ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-section-header">
      <span class="icon">🌐</span>
      <span class="label">Scenario</span>
    </div>
    """, unsafe_allow_html=True)

    ref_kind   = st.selectbox("Reference signal", ["step", "ramp", "sine", "square"], index=0)
    ref_amp    = st.number_input("Reference amplitude", value=1.0, format="%.3f")
    ref_period = st.number_input("Period (sine/square/ramp)  [s]", 0.1, 100.0, 5.0, 0.1)

    enable_dist = st.checkbox("Inject external disturbance", value=True)
    dist_kind, dist_amp, dist_t_on = "none", 0.0, 0.0
    if enable_dist:
        dist_kind  = st.selectbox("Disturbance type", ["step", "sine", "pulse"], 0)
        dist_amp   = st.number_input("Disturbance amplitude", value=0.5, format="%.3f")
        dist_t_on  = st.number_input("Disturbance onset  [s]", 0.0, 1000.0, 3.0, 0.1)

    enable_noise = st.checkbox("Measurement noise", value=False)
    noise_std = 0.0
    if enable_noise:
        noise_std = st.number_input("Sensor noise  σ", 0.0, 10.0, 0.01, 0.001, format="%.4f")

    st.markdown("<hr style='border-color:#1a2a42; margin:0.6rem 0;'>", unsafe_allow_html=True)

    # ── Simulation ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-section-header">
      <span class="icon">⏱️</span>
      <span class="label">Simulation</span>
    </div>
    """, unsafe_allow_html=True)

    t_end = st.number_input("Time horizon  [s]", 0.1, 10000.0, 10.0, 0.5, format="%.2f")
    dt    = st.select_slider("Integration step  [s]",
                             options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3,
                             format_func=lambda v: f"{v:.4f}")

    st.markdown("<hr style='border-color:#1a2a42; margin:0.6rem 0;'>", unsafe_allow_html=True)
    compare_pid = st.checkbox("⚖️ Compare against PID baseline", value=True)

    st.markdown("""
    <div class="sidebar-footer">
      <div class="sidebar-footer-text">
        ADRC Pro Tuner · v2.0<br>
        Eng. HANI NARIMENE · 2026<br>
        LADRC · NLADRC · ESO
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN PANEL ──────────────────────────────────────────────────────────────

if parse_error:
    st.error(f"⚠️  Plant parsing error: **{parse_error}**")
    st.stop()

if plant is None:
    # Welcome card
    st.markdown("""
    <div class="welcome-card">
      <div class="welcome-title">Active Disturbance Rejection Control — Engineering Platform</div>
      <div class="welcome-sub">
        Configure the plant model in the left sidebar to begin. This tool supports
        Transfer Function (TF), State-Space (SS), and user-defined Nonlinear ODE plants
        for both Linear ADRC (LADRC) and Nonlinear ADRC (NLADRC) design.
      </div>
      <div class="welcome-author">Eng. HANI NARIMENE · Higher School of Air Defense · 2026</div>
    </div>
    """, unsafe_allow_html=True)
    st.info("📐  Define a plant model in the sidebar to launch the simulation engine.")
    st.stop()

# ── Controller design ─────────────────────────────────────────────────────────
try:
    alphas: Optional[List[float]] = None
    if alpha_str.strip():
        alphas = [float(x) for x in alpha_str.replace(",", " ").split()]
    if adrc_type.startswith("Nonlinear"):
        params = design_nladrc(plant, omega_c, k_obs, alpha=alphas,
                               delta=delta_nl, r_td=r_td)
    else:
        params = design_ladrc(plant, omega_c, k_obs)
except Exception as e:
    st.error(f"Controller design failed: {e}")
    st.stop()

# ── Simulation ────────────────────────────────────────────────────────────────
with st.spinner("⚙️  Running closed-loop simulation…"):
    res = simulate_adrc(plant, params,
                        t_end=t_end, dt=dt,
                        ref_kind=ref_kind, ref_amp=ref_amp, ref_period=ref_period,
                        dist_kind=dist_kind, dist_amp=dist_amp,
                        dist_t_on=dist_t_on, noise_std=noise_std)
    res.label = "Nonlinear ADRC" if params.is_nonlinear else "Linear ADRC"

metrics = performance_metrics(res)

if compare_pid:
    Kp, Ki, Kd = auto_tune_pid(plant, omega_c)
    res_pid    = simulate_pid(plant, Kp, Ki, Kd, t_end, dt,
                              ref_kind, ref_amp, ref_period,
                              dist_kind, dist_amp, dist_t_on, noise_std)
    metrics_pid = performance_metrics(res_pid)
else:
    res_pid, metrics_pid, (Kp, Ki, Kd) = None, None, (None, None, None)

# ── KPI Dashboard ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="sh-icon">📊</span>
  <span class="sh-label">Performance Snapshot</span>
  <span class="sh-sub">Real-Time KPIs</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rise Time",          f"{metrics['rise_time_s']:.3g} s")
c2.metric("Settling Time",      f"{metrics['settling_time_s']:.3g} s")
c3.metric("Overshoot",          f"{metrics['overshoot_pct']:.2f} %")
c4.metric("Steady-State Error", f"{metrics['steady_state_error']:.4g}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("ISE",            f"{metrics['ISE']:.4g}")
c6.metric("IAE",            f"{metrics['IAE']:.4g}")
c7.metric("ITAE",           f"{metrics['ITAE']:.4g}")
c8.metric("Control Energy", f"{metrics['control_energy']:.4g}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯  Response",
    "🔭  ESO & Disturbance",
    "⚖️  ADRC vs PID",
    "🧮  Parameters",
    "🛡️  Stability",
    "📤  Export",
    "📚  Theory",
])

# ── Tab 1: Closed-loop response ──────────────────────────────────────────────
with tabs[0]:
    st.markdown("""
    <div class="section-header">
      <span class="sh-icon">🎯</span>
      <span class="sh-label">Closed-Loop Response</span>
    </div>
    """, unsafe_allow_html=True)
    st.pyplot(fig_response(res, title=f"{res.label} — Closed-Loop Response"),
              use_container_width=True)
    plant_info = (
        f"**Plant:** `{plant.kind}` · Order `{plant.order}` · b₀ ≈ `{plant.b0:.4g}`   "
        f"| **Reference:** `{ref_kind}` (amp = {ref_amp})"
    )
    if enable_dist:
        plant_info += f"   | **Disturbance:** `{dist_kind}` (amp = {dist_amp}, t_on = {dist_t_on} s)"
    if enable_noise:
        plant_info += f"   | **Noise** σ = {noise_std}"
    st.caption(plant_info)

# ── Tab 2: ESO & Disturbance ─────────────────────────────────────────────────
with tabs[1]:
    st.markdown("""
    <div class="section-header">
      <span class="sh-icon">🔭</span>
      <span class="sh-label">Extended State Observer</span>
    </div>
    """, unsafe_allow_html=True)
    st.pyplot(fig_eso(res, params), use_container_width=True)
    st.info(
        "The lower panel shows the **total disturbance estimate** f̂(t) = ẑₙ₊₁(t). "
        "ADRC's core principle: the ESO lumps all modelling uncertainty and external "
        "disturbances into a single state, which the control law cancels in real-time "
        "via  **u = (u₀ − ẑₙ₊₁) / b₀**."
    )

# ── Tab 3: ADRC vs PID Comparison ────────────────────────────────────────────
with tabs[2]:
    if compare_pid and res_pid is not None:
        st.markdown("""
        <div class="section-header">
          <span class="sh-icon">⚖️</span>
          <span class="sh-label">ADRC vs PID Comparison</span>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(fig_compare(res, res_pid), use_container_width=True)

        # ── Comparison table ──
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown(f"""
            <div class="section-header">
              <span class="sh-icon">📈</span>
              <span class="sh-label">{res.label}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="section-header">
              <span class="sh-icon">📉</span>
              <span class="sh-label">PID (Kp={Kp:.3g}, Ki={Ki:.3g}, Kd={Kd:.3g})</span>
            </div>
            """, unsafe_allow_html=True)

        metric_labels = {
            "rise_time_s":       "Rise Time [s]",
            "settling_time_s":   "Settling Time [s]",
            "overshoot_pct":     "Overshoot [%]",
            "steady_state_error":"Steady-State Error",
            "ISE":               "ISE",
            "IAE":               "IAE",
            "ITAE":              "ITAE",
            "control_energy":    "Control Energy",
        }
        rows_html = ""
        for key, label in metric_labels.items():
            va = metrics.get(key, float("nan"))
            vb = metrics_pid.get(key, float("nan")) if metrics_pid else float("nan")
            fa = f"{va:.4g}" if not (isinstance(va, float) and math.isnan(va)) else "N/A"
            fb = f"{vb:.4g}" if not (isinstance(vb, float) and math.isnan(vb)) else "N/A"
            # Lower is generally better for all metrics
            if not math.isnan(va) and not math.isnan(vb):
                ca = "metric-better" if va <= vb else "metric-worse"
                cb = "metric-better" if vb <= va else "metric-worse"
            else:
                ca = cb = ""
            rows_html += f"""
            <tr>
              <td>{label}</td>
              <td class="{ca}">{fa}</td>
              <td class="{cb}">{fb}</td>
            </tr>"""

        st.markdown(f"""
        <table class="compare-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>{res.label}</th>
              <th>PID</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.info("Enable **Compare against PID baseline** in the sidebar to populate this tab.")

# ── Tab 4: Controller Parameters ─────────────────────────────────────────────
with tabs[3]:
    st.markdown("""
    <div class="section-header">
      <span class="sh-icon">🧮</span>
      <span class="sh-label">Controller Parameters</span>
    </div>
    """, unsafe_allow_html=True)
    colp1, colp2 = st.columns(2)
    with colp1:
        st.markdown("##### Bandwidth Configuration")
        st.code(
            f"omega_c  = {params.omega_c:.6g} rad/s\n"
            f"omega_o  = {params.omega_o:.6g} rad/s\n"
            f"k ratio  = {params.omega_o / params.omega_c:.4g}",
            language="text"
        )
        st.markdown("##### State Feedback Gains  K")
        st.code(np.array2string(params.K, precision=6), language="text")
        st.markdown("##### Estimated Input Gain  b₀")
        st.code(f"b0 = {params.b0:.6g}", language="text")
    with colp2:
        st.markdown("##### ESO Observer Gains  L")
        st.code(np.array2string(params.L.flatten(), precision=6), language="text")
        st.markdown("##### Plant Summary")
        st.code(
            f"kind     = {plant.kind}\n"
            f"order    = {plant.order}\n"
            f"b0       = {plant.b0:.6g}\n"
            f"mode     = {'NLADRC' if params.is_nonlinear else 'LADRC'}",
            language="text"
        )
        if params.is_nonlinear:
            st.markdown("##### NLADRC  fal( ) Parameters")
            st.code(
                f"alpha    = {params.alpha}\n"
                f"delta    = {params.delta}\n"
                f"r_td     = {params.r_td}",
                language="text"
            )

# ── Tab 5: Stability ──────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("""
    <div class="section-header">
      <span class="sh-icon">🛡️</span>
      <span class="sh-label">Stability Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    st_info = stability_check(params)
    if st_info["stable"]:
        st.markdown("""
        <div class="stability-badge stability-stable">
          ✅  Linear Closed-Loop Nominal Stability: STABLE
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="stability-badge stability-unstable">
          ❌  Linear Closed-Loop Nominal Stability: UNSTABLE
        </div>
        """, unsafe_allow_html=True)

    cp_a, cp_b = st.columns(2)
    with cp_a:
        st.markdown("##### Controller Poles  (target)")
        st.code(np.array2string(st_info["controller_poles"], precision=4), language="text")
    with cp_b:
        st.markdown("##### Observer Poles  (achieved)")
        st.code(np.array2string(st_info["observer_poles"], precision=4), language="text")

    st.pyplot(fig_pole_map(st_info), use_container_width=True)

    if params.is_nonlinear:
        st.warning(
            "For NLADRC the eigenvalue assessment reflects the **linearization** at equilibrium. "
            "Global stability requires Lyapunov analysis or extensive simulation under representative scenarios."
        )

# ── Tab 6: Export ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("""
    <div class="section-header">
      <span class="sh-icon">📤</span>
      <span class="sh-label">Export & Documentation</span>
    </div>
    """, unsafe_allow_html=True)

    matlab_txt = export_matlab(params)
    json_txt   = export_json(params)
    report_txt = export_report(params, plant, metrics)

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.markdown("#### MATLAB / Simulink")
        st.code(matlab_txt, language="matlab")
        st.download_button(
            "⬇️  Download  .m",
            matlab_txt,
            file_name="adrc_params.m",
            mime="text/x-matlab",
            use_container_width=True,
        )
    with exp_col2:
        st.markdown("#### JSON Configuration")
        st.code(json_txt, language="json")
        st.download_button(
            "⬇️  Download  .json",
            json_txt,
            file_name="adrc_params.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp_col3:
        st.markdown("#### Engineering Report")
        st.code(report_txt, language="text")
        st.download_button(
            "⬇️  Download  .txt",
            report_txt,
            file_name="adrc_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ── Tab 7: Theory ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("""
    <div class="section-header">
      <span class="sh-icon">📚</span>
      <span class="sh-label">Theoretical Background</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(r"""
### Bandwidth-Parameterized ADRC — Quick Reference

For a SISO plant of order *n*, ADRC treats the dynamics as
$$
y^{(n)}(t) = f\bigl(y, \dot y, \dots, y^{(n-1)}, w, t\bigr) + b_0\, u(t),
$$
where **f** lumps together known dynamics, model uncertainty and external
disturbance *w(t)*. The Extended State Observer (ESO) estimates the augmented
state $z = [y, \dot y, \dots, y^{(n-1)}, f]^\top$ from $u$ and $y$ alone:

$$\dot{\hat z} = A_{\text{eso}}\hat z + B_{\text{eso}}u + L\,(y-\hat z_1).$$

Choosing all observer poles at $-\omega_o$ gives
$L_i = \binom{n+1}{i}\omega_o^{\,i}$.  The control law

$$
u = \frac{u_0 - \hat z_{n+1}}{b_0},\qquad
u_0 = \sum_{i=1}^{n} k_i\bigl(r_i - \hat z_i\bigr),
$$

with state-feedback gains $k_i = \binom{n}{i-1}\omega_c^{\,n-i+1}$ places the
nominal closed-loop poles at $-\omega_c$. Two knobs — $\omega_c$ and
$\omega_o = k\,\omega_c$ — fully parameterize the LADRC.

---

### Nonlinear ADRC (Han's Method)

Replace the linear gains with $\operatorname{fal}(e,\alpha,\delta)$:
$$
\operatorname{fal}(e,\alpha,\delta) =
\begin{cases}
e/\delta^{1-\alpha},               & |e|\le\delta,\\
|e|^{\alpha}\operatorname{sign}(e),& |e|>\delta.
\end{cases}
$$
This yields a **larger gain on small errors** (improved steady-state precision)
and a **smaller gain on large errors** (reduced control aggressiveness, better
robustness to measurement noise).

---

### Auto-Tuning Heuristic

* $\omega_c$ scales with the dominant time constant of the plant (or its
  linearization), modulated by the user-selected objective.
* $\omega_o = k\,\omega_c$ with $k\in[3,8]$; lower $k$ for noisy environments.
* For NLADRC, default exponents follow Han's geometric ladder
  $\alpha_i = 0.5^{\,i}$.

---

### References

* **Gao, Z.** (2003). *Scaling and bandwidth-parameterization based controller tuning.* ACC.
* **Han, J.** (1998). *From PID to Active Disturbance Rejection Control.* IEEE Trans. Ind. Electron.
* **Huang, Y. & Xue, W.** (2014). *Active disturbance rejection control: Methodology and theoretical analysis.* ISA Trans.
""")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="adrc-footer">
  <div class="adrc-footer-left">
    <strong>Universal ADRC Pro Tuner</strong><br>
    Linear &amp; Nonlinear ADRC · ESO Observer · Auto-Tuning
  </div>
  <div class="adrc-footer-center">
    🛡️ ADRC Pro Tuner · 2026
  </div>
  <div class="adrc-footer-right">
    <strong>Eng. HANI NARIMENE</strong><br>
    Higher School of Air Defense of the Territory<br>
    Martyr Ali Chabati · 2026
  </div>
</div>
""", unsafe_allow_html=True)
