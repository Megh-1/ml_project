"""
Social Media Bot Detection — Interactive Dashboard
====================================================
Upload your own social media data or explore the built-in 10,000-account
sample to see how the bot detection engine works.

Launch: streamlit run app/main.py
"""

import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import cross_val_score, cross_val_predict

from src.features.account_features import AccountFeatureExtractor
from src.models.clustering import BehavioralClusterAnalyzer
from src.models.scoring import CoordinationScorer

# ======================================================================
# Page Config
# ======================================================================

st.set_page_config(
    page_title="Bot Detector — Social Media Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Custom CSS — Premium Dark Theme
# ======================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #8892b0;
        font-size: 1.05rem;
        margin-top: 0;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(0,210,255,0.08), rgba(123,47,247,0.08));
        border: 1px solid rgba(0,210,255,0.2);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-card .number {
        font-size: 2rem;
        font-weight: 700;
        color: #00d2ff;
    }
    .stat-card .label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 2px;
    }

    /* Alert banners */
    .alert-danger {
        background: linear-gradient(135deg, rgba(255,65,108,0.15), rgba(255,75,43,0.1));
        border-left: 4px solid #ff416c;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 1rem 0;
        color: #ff8a8a;
        font-size: 0.95rem;
    }
    .alert-success {
        background: linear-gradient(135deg, rgba(0,176,155,0.15), rgba(150,201,61,0.1));
        border-left: 4px solid #00b09b;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 1rem 0;
        color: #7dcea0;
        font-size: 0.95rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e0e0e0;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(0,210,255,0.2);
    }

    /* Bot reason box */
    .reason-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.9rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Helper Functions
# ======================================================================

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_social_data.csv")
REQUIRED_COLS = ["user_id", "followers", "following", "account_age_days", "total_posts", "total_retweets"]


def validate_uploaded_csv(df: pd.DataFrame) -> tuple:
    """Check if uploaded CSV has the required columns. Returns (is_valid, message)."""
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    if len(df) < 10:
        return False, "Need at least 10 rows to analyze."
    return True, "✅ Data looks good!"


def explain_why_bot(row: pd.Series) -> list:
    """Generate plain-English reasons why this account looks like a bot."""
    reasons = []
    if row.get("follow_ratio", 999) < 0.02:
        reasons.append(
            f"📊 **Follows {int(row['following']):,} accounts but only has "
            f"{int(row['followers']):,} followers** — real users usually have "
            f"a balanced ratio."
        )
    if row.get("amplification_ratio", 0) > 5:
        reasons.append(
            f"🔁 **Retweets {row['amplification_ratio']:.1f}× more than they post** "
            f"— this account mostly amplifies others instead of creating content."
        )
    if row.get("posting_velocity", 0) > 10:
        reasons.append(
            f"⚡ **Posts {row['posting_velocity']:.1f} times per day** — "
            f"this posting speed suggests automation."
        )
    if row.get("account_age_days", 9999) < 30:
        reasons.append(
            f"🆕 **Account is only {int(row['account_age_days'])} days old** — "
            f"many bot accounts are created recently."
        )
    if not reasons:
        reasons.append("🔍 **Multiple behavioral signals combined** indicate this account "
                       "matches known bot patterns.")
    return reasons


def run_analysis(df: pd.DataFrame):
    """Run the full detection pipeline on a DataFrame."""
    # Feature extraction
    extractor = AccountFeatureExtractor()
    enriched = extractor.transform(df)

    # Clustering
    clusterer = BehavioralClusterAnalyzer(n_clusters=4, random_state=42)
    cluster_labels = clusterer.fit_predict(enriched)
    enriched["cluster"] = cluster_labels

    has_labels = "is_bot" in df.columns
    scorer = None
    scores = None
    metrics = None

    if has_labels:
        # Train model and score
        scorer = CoordinationScorer(max_depth=5, random_state=42)
        scorer.fit(enriched, enriched["is_bot"], run_cv=True)
        scores = scorer.predict_proba_batch(enriched)
        enriched["bot_score"] = scores
        enriched["flagged_as_bot"] = scores >= 0.5

        # Cross-validated predictions for fair accuracy
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import StandardScaler
        X = enriched[AccountFeatureExtractor.FEATURE_COLUMNS].values
        y = enriched["is_bot"].astype(int).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv_preds = cross_val_predict(
            DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
            X_scaled, y, cv=5,
        )
        cv_scores = cross_val_score(
            DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
            X_scaled, y, cv=5, scoring="accuracy",
        )

        metrics = {
            "accuracy": accuracy_score(y, cv_preds),
            "precision": precision_score(y, cv_preds),
            "recall": recall_score(y, cv_preds),
            "f1": f1_score(y, cv_preds),
            "cv_scores": cv_scores,
            "confusion": confusion_matrix(y, cv_preds),
            "cv_preds": cv_preds,
        }
    else:
        # No labels — use unsupervised anomaly scoring
        # Flag the most anomalous cluster (highest amplification, lowest follow_ratio)
        cluster_summary = clusterer.get_cluster_summary(enriched, cluster_labels)
        # Find the most suspicious cluster
        if "bot_ratio" not in cluster_summary.columns:
            cluster_summary["suspicion"] = (
                cluster_summary["amplification_ratio"] / (cluster_summary["follow_ratio"] + 0.01)
            )
            suspicious_cluster = cluster_summary.loc[cluster_summary["suspicion"].idxmax(), "cluster_id"]
        else:
            suspicious_cluster = cluster_summary.loc[cluster_summary["bot_ratio"].idxmax(), "cluster_id"]

        enriched["bot_score"] = 0.0
        enriched.loc[enriched["cluster"] == suspicious_cluster, "bot_score"] = 0.85
        enriched["flagged_as_bot"] = enriched["cluster"] == suspicious_cluster

    return enriched, clusterer, scorer, metrics


# ======================================================================
# UI Sections
# ======================================================================

def render_header():
    st.markdown("""
    <div class="hero">
        <h1>🤖 Social Media Bot Detector</h1>
        <p>Upload your data or explore the built-in sample — see which accounts are bots and why</p>
    </div>
    """, unsafe_allow_html=True)


def render_data_input():
    """Data input section — returns a DataFrame or None."""
    st.markdown('<div class="section-title">📁 Your Data</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        mode = st.radio(
            "Choose data source:",
            ["🎯 Use built-in sample (10,000 accounts)", "📤 Upload your own CSV"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with col2:
        if os.path.exists(SAMPLE_DATA_PATH):
            with open(SAMPLE_DATA_PATH, "rb") as f:
                st.download_button(
                    "⬇️ Download Sample CSV",
                    f,
                    "sample_social_data.csv",
                    "text/csv",
                    help="Download to see the expected CSV format",
                )

    if "Upload" in mode:
        uploaded = st.file_uploader(
            "Upload a CSV file with account data",
            type=["csv"],
            help=f"Required columns: {', '.join(REQUIRED_COLS)}",
        )
        if uploaded:
            df = pd.read_csv(uploaded)
            valid, msg = validate_uploaded_csv(df)
            if not valid:
                st.error(msg)
                return None
            st.success(f"{msg} — Loaded **{len(df):,}** accounts.")
            return df
        else:
            st.info(f"📋 Your CSV needs these columns: `{', '.join(REQUIRED_COLS)}`")
            st.caption("Optionally include an `is_bot` column (True/False) to measure model accuracy.")
            return None
    else:
        if os.path.exists(SAMPLE_DATA_PATH):
            df = pd.read_csv(SAMPLE_DATA_PATH)
            st.success(f"Loaded built-in sample — **{len(df):,}** accounts ({df['is_bot'].sum():,} known bots).")
            return df
        else:
            st.error("Sample data file not found. Please upload your own CSV.")
            return None


def render_overview(enriched: pd.DataFrame, metrics: dict = None):
    """Overview stats row."""
    st.markdown('<div class="section-title">📊 Overview</div>', unsafe_allow_html=True)

    total = len(enriched)
    flagged = enriched["flagged_as_bot"].sum()
    clean = total - flagged
    pct = (flagged / total * 100) if total > 0 else 0

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="stat-card"><div class="number">{total:,}</div>'
                     f'<div class="label">Total Accounts</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="stat-card"><div class="number" style="color:#ff416c">{flagged:,}</div>'
                     f'<div class="label">Suspected Bots</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="stat-card"><div class="number" style="color:#00b09b">{clean:,}</div>'
                     f'<div class="label">Clean Accounts</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="stat-card"><div class="number">{pct:.1f}%</div>'
                     f'<div class="label">Bot Rate</div></div>', unsafe_allow_html=True)


def render_3d_scatter(enriched: pd.DataFrame):
    """Interactive 3D scatter plot."""
    st.markdown('<div class="section-title">🌐 3D Behavioral Map</div>', unsafe_allow_html=True)
    st.caption("Each dot is an account. Rotate, zoom, and hover to explore. "
               "Bots tend to cluster in specific regions of this space.")

    # Limit data for performance
    plot_df = enriched.copy()
    if len(plot_df) > 5000:
        plot_df = pd.concat([
            plot_df[plot_df["flagged_as_bot"]],  # Keep all bots
            plot_df[~plot_df["flagged_as_bot"]].sample(
                min(4000, len(plot_df[~plot_df["flagged_as_bot"]])),
                random_state=42
            ),
        ])

    # Cap extreme values for compact visualization
    for col in ["follow_ratio", "amplification_ratio", "posting_velocity"]:
        q95 = plot_df[col].quantile(0.95)
        plot_df[col] = plot_df[col].clip(upper=q95)

    plot_df["Account Type"] = plot_df["flagged_as_bot"].map({True: "🤖 Suspected Bot", False: "✅ Normal"})
    plot_df["Score"] = plot_df["bot_score"].round(3)

    # Compute tight axis limits
    fr_max = plot_df["follow_ratio"].max() * 1.05
    ar_max = plot_df["amplification_ratio"].max() * 1.05
    pv_max = plot_df["posting_velocity"].max() * 1.05

    fig = px.scatter_3d(
        plot_df,
        x="follow_ratio",
        y="amplification_ratio",
        z="posting_velocity",
        color="Account Type",
        color_discrete_map={"🤖 Suspected Bot": "#ff416c", "✅ Normal": "#00d2ff"},
        hover_data={"user_id": True, "Score": True, "followers": True,
                     "following": True, "Account Type": False},
        opacity=0.6,
        size_max=6,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="Follow Ratio", range=[0, fr_max]),
            yaxis=dict(title="Amplification Ratio", range=[0, ar_max]),
            zaxis=dict(title="Posting Velocity", range=[0, pv_max]),
            bgcolor="rgba(15,12,41,0.8)",
            aspectmode="cube",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        height=480,
        margin=dict(l=0, r=0, t=20, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_bot_table(enriched: pd.DataFrame):
    """Table of identified bots with reasons."""
    st.markdown('<div class="section-title">🚨 Identified Bots — Who They Are & Why</div>',
                unsafe_allow_html=True)

    bots = enriched[enriched["flagged_as_bot"]].sort_values("bot_score", ascending=False).copy()

    if len(bots) == 0:
        st.markdown('<div class="alert-success">✅ No suspicious accounts found in this dataset!</div>',
                     unsafe_allow_html=True)
        return

    st.markdown(f'<div class="alert-danger">⚠️ Found <strong>{len(bots):,}</strong> suspicious accounts. '
                f'Scroll the table and expand any row to see the exact reason.</div>',
                unsafe_allow_html=True)

    # Display columns
    display_cols = ["user_id", "bot_score", "followers", "following",
                     "follow_ratio", "amplification_ratio", "posting_velocity",
                     "account_age_days", "cluster"]
    available = [c for c in display_cols if c in bots.columns]

    # Rename for clarity
    rename_map = {
        "bot_score": "Suspicion Score",
        "follow_ratio": "Follow Ratio",
        "amplification_ratio": "Amplification",
        "posting_velocity": "Posts/Day",
        "account_age_days": "Account Age (days)",
        "user_id": "Account ID",
        "followers": "Followers",
        "following": "Following",
        "cluster": "Cluster",
    }

    # Show top 50 in table
    show_df = bots[available].head(50).rename(columns=rename_map)

    def highlight_bots(row):
        score = row.get("Suspicion Score", 0)
        if score >= 0.85:
            return ["background-color: rgba(255,65,108,0.25); color: #ff8a8a"] * len(row)
        elif score >= 0.65:
            return ["background-color: rgba(255,165,0,0.15); color: #ffa500"] * len(row)
        return ["background-color: rgba(255,255,255,0.03)"] * len(row)

    styled = show_df.style.apply(highlight_bots, axis=1).format(
        {"Suspicion Score": "{:.3f}", "Follow Ratio": "{:.4f}",
         "Amplification": "{:.1f}", "Posts/Day": "{:.2f}"}
    )
    st.dataframe(styled, use_container_width=True, height=400)

    if len(bots) > 50:
        st.caption(f"Showing top 50 of {len(bots):,} flagged accounts (sorted by score).")

    # Expandable reasons for top 10
    st.markdown("#### 🔎 Why These Accounts Were Flagged")
    st.caption("Click any account below to see the detailed explanation.")

    for _, row in bots.head(10).iterrows():
        reasons = explain_why_bot(row)
        with st.expander(f"🤖 {row['user_id']} — Score: {row['bot_score']:.3f}"):
            for reason in reasons:
                st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)


def render_model_accuracy(metrics: dict):
    """Model accuracy panel with anti-overfitting evidence."""
    st.markdown('<div class="section-title">📈 Model Accuracy — How Good Is It?</div>',
                unsafe_allow_html=True)

    st.caption("These numbers come from **5-fold cross-validation** — the model is tested on "
               "data it has never seen during training. This proves it's not just memorizing the data.")

    # Metrics row
    cols = st.columns(4)
    with cols[0]:
        st.metric("🎯 Accuracy", f"{metrics['accuracy']:.1%}")
    with cols[1]:
        st.metric("🔍 Precision", f"{metrics['precision']:.1%}",
                   help="Of all accounts flagged as bots, how many actually are?")
    with cols[2]:
        st.metric("📡 Recall", f"{metrics['recall']:.1%}",
                   help="Of all actual bots, how many did we catch?")
    with cols[3]:
        st.metric("⚖️ F1 Score", f"{metrics['f1']:.1%}",
                   help="Balance between precision and recall")

    # Cross-validation consistency (anti-overfitting proof)
    cv = metrics["cv_scores"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cross-Validation Scores (5 Folds)")
        st.caption("If these scores are consistent, the model is **NOT overfitting**.")

        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv))],
            "Accuracy": cv,
        })
        fig = px.bar(
            cv_df, x="Fold", y="Accuracy",
            text=cv_df["Accuracy"].apply(lambda x: f"{x:.1%}"),
            color_discrete_sequence=["#00d2ff"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            showlegend=False,
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        spread = cv.max() - cv.min()
        if spread < 0.05:
            st.success(f"✅ Scores are very consistent (spread: {spread:.1%}) — **no overfitting detected.**")
        elif spread < 0.10:
            st.warning(f"⚠️ Scores have moderate spread ({spread:.1%}) — consider more data.")
        else:
            st.error(f"❌ Scores vary significantly ({spread:.1%}) — possible instability.")

    with col2:
        st.markdown("#### Confusion Matrix")
        st.caption("Rows = actual labels, Columns = predicted labels")

        cm = metrics["confusion"]
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Normal", "Bot"],
            y=["Normal", "Bot"],
            text_auto=True,
            color_continuous_scale=["#0f0c29", "#00d2ff", "#7b2ff7"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_sidebar_info():
    """Sidebar with instructions."""
    with st.sidebar:
        st.markdown("## 🤖 Bot Detector")
        st.markdown("---")
        st.markdown("""
        **How it works:**

        1. Upload a CSV with account data
        2. The engine extracts behavioral patterns
        3. Machine learning identifies suspicious accounts
        4. You see exactly **who** is a bot and **why**
        """)
        st.markdown("---")
        st.markdown("**What makes a bot?**")
        st.markdown("""
        - 📊 Follows thousands, but nobody follows back
        - 🔁 Only retweets, never posts original content
        - ⚡ Posts at inhuman speeds
        - 🆕 Account was just created
        """)
        st.markdown("---")
        st.markdown("""
        **CSV Format:**
        ```
        user_id,followers,following,
        account_age_days,total_posts,
        total_retweets
        ```
        Optional: add `is_bot` column for
        accuracy measurement.
        """)
        st.markdown("---")
        st.caption("Built with Python, scikit-learn, NetworkX & Streamlit")


# ======================================================================
# Main App
# ======================================================================

def main():
    render_sidebar_info()
    render_header()

    # Data input
    df = render_data_input()
    if df is None:
        st.stop()

    # Run analysis
    with st.spinner("🔬 Analyzing accounts... This takes a few seconds."):
        enriched, clusterer, scorer, metrics = run_analysis(df)

    # Overview
    render_overview(enriched, metrics)

    st.markdown("---")

    # 3D Scatter
    render_3d_scatter(enriched)

    st.markdown("---")

    # Bot table with reasons
    render_bot_table(enriched)

    # Model accuracy (only if labels exist)
    if metrics:
        st.markdown("---")
        render_model_accuracy(metrics)


if __name__ == "__main__":
    main()
