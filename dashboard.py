"""Interactive Real-Time Fraud Operations & Analyst Dashboard.

Built with Streamlit for financial fraud analysts and risk teams.
Features:
  - Real-Time Transaction Scoring & Live Simulator
  - Interactive SHAP Waterfall & Feature Attribution Explorer
  - Dynamic Cost-Matrix & Decision Threshold Simulator
  - Real-Time Feedback Loop for Streaming Online Learning
  - Population Stability Index (PSI) & Concept Drift Telemetry
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from config import CONFIG, CostMatrixConfig
from cost_evaluation import CostAwareThresholdOptimizer
from data_preprocessing import LeakageSafePreprocessor, generate_synthetic_fraud_dataset
from drift_monitoring import ConceptDriftMonitor
from explainability import FraudExplainer
from model_pipeline import BaseFraudEstimator
from online_learning import OnlineFraudLearner

st.set_page_config(
    page_title="FraudGuard AI | Operations Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("Dashboard")


# ---------------------------------------------------------------------------
# Caching & State Initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def load_system_core():
    """Initializes and caches model, preprocessor, explainer, and drift monitors."""
    model_path = CONFIG.artifact_dir / CONFIG.model_artifact_name
    preproc_path = CONFIG.artifact_dir / CONFIG.preprocessor_artifact_name
    thresh_path = CONFIG.artifact_dir / CONFIG.threshold_artifact_name

    if not (model_path.exists() and preproc_path.exists() and thresh_path.exists()):
        from train import run_training_pipeline
        run_training_pipeline(model_type="xgboost", imbalance_strategy="hybrid", n_samples=10000)

    import joblib
    model = BaseFraudEstimator.load(model_path)
    preprocessor = joblib.load(preproc_path)

    with open(thresh_path, "r") as f:
        meta = json.load(f)

    # Initialize SHAP & Drift
    sample_df = generate_synthetic_fraud_dataset(n_samples=300, random_state=42)
    sample_features = sample_df.drop(columns=["Class"])
    bg_transformed = preprocessor.transform(sample_features)

    explainer = FraudExplainer(model_wrapper=model, background_data=bg_transformed)
    drift_monitor = ConceptDriftMonitor(reference_data=bg_transformed, buffer_size=300)
    online_learner = OnlineFraudLearner()
    online_learner.initialize_from_batch(bg_transformed.iloc[:100], sample_df["Class"].iloc[:100], epochs=2)

    return model, preprocessor, meta, explainer, drift_monitor, online_learner


model, preprocessor, metadata, explainer, drift_monitor, online_learner = load_system_core()

if "recent_transactions" not in st.session_state:
    st.session_state.recent_transactions = []


# ---------------------------------------------------------------------------
# Sidebar Controls & Global Metrics
# ---------------------------------------------------------------------------

st.sidebar.title("🛡️ FraudGuard AI")
st.sidebar.caption("Production Financial Risk & MLOps Engine")
st.sidebar.divider()

st.sidebar.subheader("System Health")
st.sidebar.success(f"**Model Backend:** {model.model_name}")
st.sidebar.metric("Validation PR-AUC", f"{metadata.get('pr_auc', 0.85):.4f}")
st.sidebar.metric("Operational Decision Threshold", f"{metadata.get('optimal_threshold', 0.5):.4f}")
st.sidebar.metric("Max Review Workload Cap", f"{metadata.get('max_review_rate_cap', 0.015):.2%}")


# ---------------------------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------------------------

tab_live, tab_inspect, tab_cost, tab_drift, tab_online = st.tabs([
    "⚡ Real-Time Live Feed",
    "🔍 Transaction Inspector & SHAP",
    "💰 Cost-Aware Threshold Tuner",
    "📊 Concept Drift Monitoring",
    "🔄 Streaming Online Learning",
])


# ---------------------------------------------------------------------------
# TAB 1: Live Feed & Simulator
# ---------------------------------------------------------------------------

with tab_live:
    st.header("Real-Time Transaction Stream Simulator")
    st.markdown("Simulate high-throughput incoming transaction traffic and view real-time risk triage decisions.")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        sim_speed = st.slider("Stream Speed (Txs / Batch)", min_value=1, max_value=20, value=5)
    with col2:
        fraud_bias = st.slider("Simulated Fraud Injection Rate", min_value=0.0, max_value=0.10, value=0.02, step=0.01)
    with col3:
        st.write("")
        st.write("")
        run_sim = st.button("🚀 Stream Next Batch of Transactions", use_container_width=True)

    if run_sim:
        new_batch = generate_synthetic_fraud_dataset(n_samples=sim_speed, fraud_rate=fraud_bias)
        raw_feats = new_batch.drop(columns=["Class"])
        trans_feats = preprocessor.transform(raw_feats)
        probas = model.predict_proba(trans_feats)

        threshold = metadata.get("optimal_threshold", 0.50)

        for i in range(len(new_batch)):
            prob = float(probas[i])
            amt = float(new_batch.iloc[i]["Amount"])
            t_sec = float(new_batch.iloc[i]["Time"])

            if prob >= 0.85:
                action = "DECLINE ⛔"
                action_badge = "🔴 Decline"
            elif prob >= threshold:
                action = "MANUAL REVIEW ⚠️"
                action_badge = "🟡 Review"
            else:
                action = "APPROVE ✅"
                action_badge = "🟢 Approve"

            # Ingest to drift monitor
            drift_monitor.ingest_streaming_transaction(trans_feats.iloc[i].to_dict())

            st.session_state.recent_transactions.insert(
                0,
                {
                    "Timestamp (s)": round(t_sec, 1),
                    "Amount ($)": f"${amt:,.2f}",
                    "Fraud Probability": f"{prob:.4%}",
                    "Triage Action": action_badge,
                    "Raw Score": prob,
                    "Actual Label": int(new_batch.iloc[i]["Class"]),
                },
            )

        if len(st.session_state.recent_transactions) > 50:
            st.session_state.recent_transactions = st.session_state.recent_transactions[:50]

    if st.session_state.recent_transactions:
        df_feed = pd.DataFrame(st.session_state.recent_transactions)
        st.dataframe(df_feed, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Stream Next Batch of Transactions' to start the live transaction stream.")


# ---------------------------------------------------------------------------
# TAB 2: Transaction Inspector & SHAP Explainability
# ---------------------------------------------------------------------------

with tab_inspect:
    st.header("Transaction Risk Analyzer & Local SHAP Explainability")
    st.markdown("Deep dive into specific transaction feature vectors and view exact risk drivers.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        preset_type = st.radio("Load Preset Sample:", ["High-Risk Fraud Profile", "Legitimate Baseline", "Custom Manual Entry"])

    # Prepare sample inputs
    if preset_type == "High-Risk Fraud Profile":
        amt_val = 890.0
        time_val = 43200.0
        v14_val = -5.2
        v17_val = -3.8
        v4_val = 3.5
    elif preset_type == "Legitimate Baseline":
        amt_val = 45.20
        time_val = 36000.0
        v14_val = 0.1
        v17_val = -0.2
        v4_val = 0.05
    else:
        amt_val = st.number_input("Transaction Amount ($)", value=120.0)
        time_val = st.number_input("Time Elapsed (s)", value=25000.0)
        v14_val = st.slider("V14 (Behavioral Pattern Key Signal)", -10.0, 5.0, -1.0)
        v17_val = st.slider("V17 (Historical Fraud Correlation)", -10.0, 5.0, -0.5)
        v4_val = st.slider("V4 (Merchant Risk Profile)", -5.0, 10.0, 1.0)

    # Build input DataFrame
    single_dict = {f"V{i}": 0.0 for i in range(1, 29)}
    single_dict["Time"] = time_val
    single_dict["Amount"] = amt_val
    single_dict["V4"] = v4_val
    single_dict["V14"] = v14_val
    single_dict["V17"] = v17_val

    df_single = pd.DataFrame([single_dict])
    df_single_trans = preprocessor.transform(df_single)
    single_prob = float(model.predict_proba(df_single_trans)[0])
    threshold = metadata.get("optimal_threshold", 0.50)

    st.subheader("Decision Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Fraud Probability", f"{single_prob:.3%}")
    m2.metric("Decision Threshold", f"{threshold:.4f}")
    m3.metric("Expected Financial Loss", f"${single_prob * CONFIG.cost.cost_false_negative:,.2f}")
    if single_prob >= 0.85:
        m4.error("Decision: DECLINE ⛔")
    elif single_prob >= threshold:
        m4.warning("Decision: MANUAL REVIEW ⚠️")
    else:
        m4.success("Decision: APPROVE ✅")

    # SHAP Decomposition
    st.subheader("Top Contributing Risk Reasons (SHAP Values)")
    explanation = explainer.explain_transaction(df_single_trans, top_k=7)

    reasons_data = []
    for c in explanation.top_contributions:
        reasons_data.append({
            "Rank": c.importance_rank,
            "Feature": c.feature_name,
            "Impact Direction": c.direction,
            "SHAP Attribution": c.shap_value,
            "Business Reason": c.reason_code,
        })
    df_reasons = pd.DataFrame(reasons_data)

    st.dataframe(df_reasons, use_container_width=True, hide_index=True)

    # Visual Bar Chart of SHAP Contributions
    st.bar_chart(
        df_reasons.set_index("Feature")["SHAP Attribution"],
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# TAB 3: Cost Matrix & Threshold Tuner
# ---------------------------------------------------------------------------

with tab_cost:
    st.header("Cost-Aware Decision Threshold Simulator")
    st.markdown("Optimize probability decision boundaries against custom financial losses and review team operational capacity.")

    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        cost_fn = st.slider("False Negative Cost ($ Fraud Loss)", 20.0, 500.0, float(CONFIG.cost.cost_false_negative), step=10.0)
    with c_col2:
        cost_fp = st.slider("False Positive Cost ($ Manual Review)", 1.0, 50.0, float(CONFIG.cost.cost_false_positive), step=1.0)
    with c_col3:
        max_workload = st.slider("Max Review Workload Cap (%)", 0.5, 5.0, float(CONFIG.cost.max_review_rate * 100.0), step=0.1) / 100.0

    # Run optimizer on validation split
    dyn_cost_cfg = CostMatrixConfig(
        cost_false_negative=cost_fn,
        cost_false_positive=cost_fp,
        max_review_rate=max_workload,
    )
    optimizer = CostAwareThresholdOptimizer(cost_config=dyn_cost_cfg)

    val_df_sim = generate_synthetic_fraud_dataset(n_samples=2000, fraud_rate=0.002, random_state=99)
    val_trans = preprocessor.transform(val_df_sim.drop(columns=["Class"]))
    val_probs = model.predict_proba(val_trans)

    best_thresh_res, all_thresh_df = optimizer.find_optimal_threshold(val_df_sim["Class"], val_probs)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Optimized Threshold", f"{best_thresh_res.threshold:.4f}")
    t2.metric("Total Loss at Optimal", f"${best_thresh_res.total_cost:,.2f}")
    t3.metric("Review Rate", f"{best_thresh_res.review_rate:.3%}", delta=f"Cap: {max_workload:.2%}")
    t4.metric("Precision / Recall", f"{best_thresh_res.precision:.1%} / {best_thresh_res.recall:.1%}")

    st.subheader("Expected Financial Loss vs. Decision Threshold")
    st.line_chart(all_thresh_df.set_index("threshold")[["total_cost"]], use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 4: Concept Drift Telemetry
# ---------------------------------------------------------------------------

with tab_drift:
    st.header("Real-Time Data & Concept Drift Monitoring")
    st.markdown("Track distribution shifts across transaction attributes using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.")

    drift_report = drift_monitor.evaluate_system_drift()

    d1, d2, d3 = st.columns(3)
    if drift_report.overall_system_status == "HEALTHY":
        d1.success(f"**System Status:** {drift_report.overall_system_status}")
    elif drift_report.overall_system_status == "WARNING":
        d1.warning(f"**System Status:** {drift_report.overall_system_status}")
    else:
        d1.error(f"**System Status:** {drift_report.overall_system_status}")

    d2.metric("Drifted Features Count", f"{drift_report.drifted_feature_count} / {drift_report.total_features_monitored}")
    d3.metric("Highest PSI Feature", f"{drift_report.max_psi_feature} ({drift_report.max_psi_value:.4f})")

    st.info(f"**Automated Recommendation:** {drift_report.recommendation}")

    if drift_report.feature_reports:
        df_drift = pd.DataFrame([
            {
                "Feature": r.feature_name,
                "PSI Score": r.psi_score,
                "KS Statistic": r.ks_statistic,
                "p-value": r.ks_p_value,
                "Drift Status": r.drift_status,
            }
            for r in drift_report.feature_reports
        ])
        st.dataframe(df_drift, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 5: Online Learning & Continuous Feedback
# ---------------------------------------------------------------------------

with tab_online:
    st.header("Streaming Online Learning & Incremental Updates")
    st.markdown("Provide instant ground-truth feedback on disputed charges to update the model in near real-time without taking it offline.")

    on_col1, on_col2 = st.columns([1, 1])

    with on_col1:
        st.subheader("Analyst Feedback Action")
        confirmed_type = st.radio("Analyst Verification Result:", ["Confirmed Fraud (Chargeback)", "Verified Legitimate Transaction"])
        feedback_amount = st.number_input("Transaction Monetary Amount ($)", value=340.0)

        if st.button("⚡ Submit Feedback & Perform Incremental Model Update"):
            label = 1 if "Fraud" in confirmed_type else 0
            fb_dict = {f"V{i}": np.random.randn() * (2.0 if label == 1 else 0.5) for i in range(1, 29)}
            fb_dict["Time"] = 50000.0
            fb_dict["Amount"] = feedback_amount
            fb_df = preprocessor.transform(pd.DataFrame([fb_dict]))

            update_res = online_learner.update_incremental(fb_df, [label])
            st.success(f"Model updated successfully! Cumulative transactions learned: {update_res['cumulative_samples_seen']}")
            st.json(update_res)

    with on_col2:
        st.subheader("Online Streaming Telemetry")
        st.metric("Total Cumulative Samples Ingested", online_learner.stats.total_samples_ingested)
        st.metric("Total Frauds Learned", online_learner.stats.total_frauds_learned)
        st.metric("Current Running Log Loss", f"{online_learner.stats.running_loss:.4f}")

        if online_learner.stats.recent_losses:
            st.line_chart(pd.DataFrame({"Running Log Loss": online_learner.stats.recent_losses}), use_container_width=True)
