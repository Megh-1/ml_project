"""
Social Media Bot Detection — Two-Mode Dashboard
=================================================
Mode 1: Batch CSV Analysis
Mode 2: Single Account Check (Instagram / X)

Launch: streamlit run app/main.py
"""

import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

from src.features.account_features import AccountFeatureExtractor
from src.models.clustering import BehavioralClusterAnalyzer
from src.models.scoring import CoordinationScorer

# ======================================================================
# Page Config
# ======================================================================
st.set_page_config(
    page_title="Bot Detector — Multi-Platform Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CSS
# ======================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .hero { text-align: center; padding: 1.5rem 0 1rem; }
    .hero h1 {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p { color: #8892b0; font-size: 1.05rem; margin-top: 0; }
    .stat-card {
        background: linear-gradient(135deg, rgba(0,210,255,0.08), rgba(123,47,247,0.08));
        border: 1px solid rgba(0,210,255,0.2); border-radius: 14px;
        padding: 1.2rem 1.5rem; text-align: center; margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-4px); }
    .stat-card .number { font-size: 2rem; font-weight: 700; color: #00d2ff; }
    .stat-card .label { font-size: 0.85rem; color: #8892b0; margin-top: 2px; }
    .alert-danger {
        background: linear-gradient(135deg, rgba(255,65,108,0.15), rgba(255,75,43,0.1));
        border-left: 4px solid #ff416c; border-radius: 8px;
        padding: 14px 18px; margin: 1rem 0; color: #ff8a8a; font-size: 0.95rem;
    }
    .alert-success {
        background: linear-gradient(135deg, rgba(0,176,155,0.15), rgba(150,201,61,0.1));
        border-left: 4px solid #00b09b; border-radius: 8px;
        padding: 14px 18px; margin: 1rem 0; color: #7dcea0; font-size: 0.95rem;
    }
    .section-title {
        font-size: 1.4rem; font-weight: 600; color: #e0e0e0;
        margin: 1.5rem 0 0.8rem; padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(0,210,255,0.2);
    }
    .reason-box {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 12px 16px; margin: 6px 0; font-size: 0.9rem;
    }
    .verdict-card {
        border-radius: 16px; padding: 2rem; text-align: center;
        margin: 1.5rem 0; font-size: 1.3rem; font-weight: 600;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# Constants & Helpers
# ======================================================================
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_social_data.csv")
TRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "models", "trained_scorer.pkl")
TRAINED_METRICS_PATH = TRAINED_MODEL_PATH.replace(".pkl", "_metrics.pkl")
REQUIRED_COLS = ["user_id", "followers", "following", "account_age_days", "total_posts", "total_retweets"]


def _load_trained_scorer():
    """Load the pre-trained scorer from disk, or fall back to synthetic training."""
    if os.path.exists(TRAINED_MODEL_PATH):
        return joblib.load(TRAINED_MODEL_PATH), "real"
    # Fallback: train on synthetic data
    from src.data.config import SimulationConfig
    from src.data.simulator import SocialDataSimulator
    config = SimulationConfig(seed=42)
    sim = SocialDataSimulator(config)
    train_users = sim.generate_users()
    extractor = AccountFeatureExtractor()
    train_enriched = extractor.transform(train_users)
    scorer = CoordinationScorer(max_depth=5, random_state=42)
    scorer.fit(train_enriched, train_enriched["is_bot"], run_cv=False)
    return scorer, "synthetic"


def explain_why_bot(row):
    reasons = []
    if row.get("follow_ratio", 999) < 0.02:
        reasons.append(f"📊 **Follows {int(row['following']):,} but only {int(row['followers']):,} followers** — unbalanced ratio.")
    if row.get("amplification_ratio", 0) > 5:
        reasons.append(f"🔁 **Retweets {row['amplification_ratio']:.1f}× more than posts** — amplification bot signal.")
    if row.get("posting_velocity", 0) > 10:
        reasons.append(f"⚡ **Posts {row['posting_velocity']:.1f} times/day** — suggests automation.")
    if row.get("account_age_days", 9999) < 30:
        reasons.append(f"🆕 **Account only {int(row['account_age_days'])} days old** — freshly created.")
    if not reasons:
        reasons.append("🔍 **Multiple behavioral signals combined** match known bot patterns.")
    return reasons


def run_batch_analysis(df, scorer):
    extractor = AccountFeatureExtractor()
    enriched = extractor.transform(df)
    clusterer = BehavioralClusterAnalyzer(n_clusters=4, random_state=42)
    enriched["cluster"] = clusterer.fit_predict(enriched)
    scores = scorer.predict_proba_batch(enriched)
    enriched["bot_score"] = scores
    enriched["flagged_as_bot"] = scores >= 0.5

    metrics = None
    if "is_bot" in df.columns:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import StandardScaler
        y_true = enriched["is_bot"].astype(int).values
        y_pred = (scores >= 0.5).astype(int)
        X = enriched[AccountFeatureExtractor.FEATURE_COLUMNS].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cv_scores = cross_val_score(
            DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
            X_scaled, y_true, cv=5, scoring="accuracy",
        )
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "cv_scores": cv_scores,
            "confusion": confusion_matrix(y_true, y_pred),
        }
    return enriched, metrics


# ======================================================================
# Sidebar
# ======================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 Bot Detector")
        st.markdown("---")
        st.markdown("""
        **Two Modes:**
        - 📤 **Batch Upload** — Score a CSV of accounts
        - 👤 **Single Check** — Enter stats for one account
        """)
        st.markdown("---")
        st.markdown("**Platforms:** 📸 Instagram · 🐦 X (Twitter)")
        st.markdown("---")
        st.markdown("**Model:** DecisionTree (max_depth=5)")
        st.markdown("**Features:** follow_ratio, amplification_ratio, posting_velocity")
        st.markdown("---")

        # Show training info if available
        if os.path.exists(TRAINED_METRICS_PATH):
            metrics = joblib.load(TRAINED_METRICS_PATH)
            st.markdown("**Training Metrics (Real Data):**")
            for split, m in metrics.items():
                st.caption(f"{split}: {m['accuracy']:.1%} accuracy")
        st.markdown("---")
        st.caption("**Trained on 5 datasets:**")
        st.caption("• bot_detection_data.csv (50k)")
        st.caption("• fake_social_media.csv (3k)")
        st.caption("• instafake_training_data.csv (2.6k)")
        st.caption("• Instagram_fake_profile.csv (5k)")
        st.caption("• twitter_human_bots.csv (37k)")


# ======================================================================
# Mode 1: Batch CSV
# ======================================================================
def render_batch_mode(scorer):
    st.markdown('<div class="section-title">📁 Batch CSV Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        source = st.radio("Data source:", ["🎯 Built-in sample (10k accounts)", "📤 Upload CSV"],
                          horizontal=True, label_visibility="collapsed")
    with col2:
        if os.path.exists(SAMPLE_DATA_PATH):
            with open(SAMPLE_DATA_PATH, "rb") as f:
                st.download_button("⬇️ Sample CSV", f, "sample_social_data.csv", "text/csv")

    df = None
    if "Upload" in source:
        uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                     help=f"Required: {', '.join(REQUIRED_COLS)}")
        if uploaded:
            df = pd.read_csv(uploaded)
            missing = set(REQUIRED_COLS) - set(df.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                return
            if len(df) < 10:
                st.error("Need at least 10 rows.")
                return
            st.success(f"✅ Loaded **{len(df):,}** accounts.")
    else:
        if os.path.exists(SAMPLE_DATA_PATH):
            df = pd.read_csv(SAMPLE_DATA_PATH)
            st.success(f"Loaded built-in sample — **{len(df):,}** accounts.")
        else:
            st.error("Sample data not found. Upload your own CSV.")
            return

    if df is None:
        st.info(f"📋 Required columns: `{', '.join(REQUIRED_COLS)}`")
        st.caption("Optional: add `is_bot` column for accuracy measurement.")
        return

    with st.spinner("🔬 Analyzing accounts..."):
        enriched, metrics = run_batch_analysis(df, scorer)

    # Overview cards
    total = len(enriched)
    flagged = int(enriched["flagged_as_bot"].sum())
    clean = total - flagged
    pct = flagged / total * 100 if total > 0 else 0

    cols = st.columns(4)
    for i, (val, lbl, color) in enumerate([
        (f"{total:,}", "Total Accounts", "#00d2ff"),
        (f"{flagged:,}", "Suspected Bots", "#ff416c"),
        (f"{clean:,}", "Clean Accounts", "#00b09b"),
        (f"{pct:.1f}%", "Bot Rate", "#00d2ff"),
    ]):
        with cols[i]:
            st.markdown(f'<div class="stat-card"><div class="number" style="color:{color}">{val}</div>'
                        f'<div class="label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # 3D scatter
    st.markdown('<div class="section-title">🌐 3D Behavioral Map</div>', unsafe_allow_html=True)
    plot_df = enriched.copy()
    if len(plot_df) > 5000:
        plot_df = pd.concat([plot_df[plot_df["flagged_as_bot"]],
                             plot_df[~plot_df["flagged_as_bot"]].sample(min(4000, (~plot_df["flagged_as_bot"]).sum()), random_state=42)])
    for c in ["follow_ratio", "amplification_ratio", "posting_velocity"]:
        plot_df[c] = plot_df[c].clip(upper=plot_df[c].quantile(0.95))
    plot_df["Type"] = plot_df["flagged_as_bot"].map({True: "🤖 Bot", False: "✅ Normal"})
    fig = px.scatter_3d(plot_df, x="follow_ratio", y="amplification_ratio", z="posting_velocity",
                        color="Type", color_discrete_map={"🤖 Bot": "#ff416c", "✅ Normal": "#00d2ff"}, opacity=0.6)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      scene=dict(bgcolor="rgba(15,12,41,0.8)", aspectmode="cube"),
                      height=480, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Bot table
    st.markdown('<div class="section-title">🚨 Identified Bots</div>', unsafe_allow_html=True)
    bots = enriched[enriched["flagged_as_bot"]].sort_values("bot_score", ascending=False)
    if len(bots) == 0:
        st.markdown('<div class="alert-success">✅ No suspicious accounts found!</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-danger">⚠️ Found <strong>{len(bots):,}</strong> suspicious accounts.</div>', unsafe_allow_html=True)
        display = ["user_id", "bot_score", "followers", "following", "follow_ratio", "amplification_ratio", "posting_velocity", "account_age_days", "cluster"]
        avail = [c for c in display if c in bots.columns]
        st.dataframe(bots[avail].head(50), use_container_width=True, height=400)
        st.markdown("#### 🔎 Why Flagged")
        for _, row in bots.head(10).iterrows():
            with st.expander(f"🤖 {row['user_id']} — Score: {row['bot_score']:.3f}"):
                for r in explain_why_bot(row):
                    st.markdown(f'<div class="reason-box">{r}</div>', unsafe_allow_html=True)

    # Accuracy (if labels exist)
    if metrics:
        st.markdown("---")
        st.markdown('<div class="section-title">📈 Model Accuracy</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("🔍 Precision", f"{metrics['precision']:.1%}")
        m3.metric("📡 Recall", f"{metrics['recall']:.1%}")
        m4.metric("⚖️ F1", f"{metrics['f1']:.1%}")

        cv = metrics["cv_scores"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Cross-Validation (5 Folds)")
            cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv))], "Accuracy": cv})
            fig = px.bar(cv_df, x="Fold", y="Accuracy", text=cv_df["Accuracy"].apply(lambda x: f"{x:.1%}"),
                         color_discrete_sequence=["#00d2ff"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0,1.05], tickformat=".0%"),
                              showlegend=False, height=300, margin=dict(l=40,r=20,t=20,b=40))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Confusion Matrix")
            cm = metrics["confusion"]
            fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=["Normal","Bot"], y=["Normal","Bot"], text_auto=True,
                            color_continuous_scale=["#0f0c29","#00d2ff","#7b2ff7"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=300, margin=dict(l=40,r=20,t=20,b=40))
            st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# Mode 2: Single Account Check
# ======================================================================
def render_single_mode(scorer):
    st.markdown('<div class="section-title">👤 Single Account Check</div>', unsafe_allow_html=True)
    st.write("Enter an account's stats to get an instant bot/human verdict.")

    col1, col2 = st.columns(2)
    with col1:
        platform = st.selectbox("Platform", ["📸 Instagram", "🐦 X (Twitter)"])
        followers = st.number_input("Followers", min_value=0, value=100, step=1)
        following = st.number_input("Following", min_value=0, value=500, step=1)
        account_age = st.number_input("Account Age (days)", min_value=1, value=365, step=1)
    with col2:
        total_posts = st.number_input("Total Posts / Tweets", min_value=0, value=50, step=1)
        if "Instagram" in platform:
            total_retweets = 0
            st.info("ℹ️ Instagram has no retweets — set to 0 automatically.")
        else:
            total_retweets = st.number_input("Total Retweets", min_value=0, value=10, step=1)

    if st.button("🔍 Analyze Account", type="primary", use_container_width=True):
        # Build a single-row DataFrame matching the existing schema
        user_df = pd.DataFrame([{
            "user_id": "manual_check",
            "followers": followers,
            "following": following,
            "account_age_days": account_age,
            "total_posts": total_posts,
            "total_retweets": total_retweets,
            "is_bot": 0,  # placeholder
        }])

        extractor = AccountFeatureExtractor()
        enriched = extractor.transform(user_df)
        result = scorer.predict_coordination_score(enriched)
        prob = result["coordination_likelihood"]
        tier = result["risk_tier"]

        st.markdown("---")

        # Verdict card
        if prob >= 0.85:
            st.markdown(f'<div class="verdict-card alert-danger" style="font-size:1.4rem;">🤖 <strong>Bot Detected</strong> — {prob:.1%} confidence<br><small>Risk Tier: {tier}</small></div>', unsafe_allow_html=True)
        elif prob >= 0.5:
            st.markdown(f'<div class="verdict-card alert-danger" style="border-left-color:#ffa500; font-size:1.4rem;">⚠️ <strong>Likely Bot</strong> — {prob:.1%} confidence<br><small>Risk Tier: {tier}</small></div>', unsafe_allow_html=True)
        elif prob >= 0.3:
            st.markdown(f'<div class="verdict-card alert-success" style="border-left-color:#ffd700; font-size:1.4rem;">👤 <strong>Probably Human</strong> — {1-prob:.1%} confidence<br><small>Risk Tier: {tier}</small></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-card alert-success" style="font-size:1.4rem;">✅ <strong>Human</strong> — {1-prob:.1%} confidence<br><small>Risk Tier: {tier}</small></div>', unsafe_allow_html=True)

        # Feature breakdown
        st.markdown("#### 📊 Feature Breakdown")
        fr = enriched["follow_ratio"].iloc[0]
        ar = enriched["amplification_ratio"].iloc[0]
        pv = enriched["posting_velocity"].iloc[0]
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Follow Ratio", f"{fr:.4f}", help="followers / (following + 1)")
        fc2.metric("Amplification", f"{ar:.2f}", help="retweets / (posts + 1)")
        fc3.metric("Posts/Day", f"{pv:.2f}", help="posts / (account_age + 1)")

        # Reasons
        if prob >= 0.5:
            st.markdown("#### 🔎 Why this verdict?")
            row = enriched.iloc[0]
            for r in explain_why_bot(row):
                st.markdown(f'<div class="reason-box">{r}</div>', unsafe_allow_html=True)

        # Top features from model
        st.markdown("#### 🧠 Model's Top Decision Features")
        for feat in result["top_features"]:
            st.markdown(f'<div class="reason-box">**{feat["feature_name"]}** = {feat["value"]:.4f} (importance: {feat["importance"]:.3f})</div>', unsafe_allow_html=True)


# ======================================================================
# Main
# ======================================================================
def main():
    render_sidebar()

    st.markdown("""
    <div class="hero">
        <h1>🤖 Multi-Platform Bot Detector</h1>
        <p>Analyze CSV datasets in batch or check a single Instagram / X account instantly</p>
    </div>
    """, unsafe_allow_html=True)

    scorer, source = _load_trained_scorer()
    if source == "real":
        st.caption("✅ Model trained on 96k+ real-world accounts from 5 datasets.")
    else:
        st.caption("⚠️ Using synthetic-data model. Run `python src/api/train_model.py` for real-data model.")

    tab1, tab2 = st.tabs(["📤 Mode 1: Batch CSV Analysis", "👤 Mode 2: Single Account Check"])
    with tab1:
        render_batch_mode(scorer)
    with tab2:
        render_single_mode(scorer)


if __name__ == "__main__":
    main()
