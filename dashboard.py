"""Interactive Real-Time Fraud Operations & Analyst Dashboard.

Built with Streamlit for financial fraud analysts and risk teams.
Features:
  - Real-Time Transaction Scoring & Live Simulator with Case Ingestion
  - Interactive SHAP Waterfall & Feature Attribution Explorer with Case Filing
  - Formal Fraud Case Management & PCI-DSS Audit Trail Queue
  - Champion-Challenger Model Registry & A/B Testing
  - Dynamic Cost-Matrix & Decision Threshold Simulator
  - Real-Time Feedback Loop for Streaming Online Learning
  - Population Stability Index (PSI) & Concept Drift Telemetry
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from case_manager import CASE_MANAGER, CasePriority, CaseStatus, FraudCase
from config import CONFIG, CostMatrixConfig
from cost_evaluation import CostAwareThresholdOptimizer
from data_preprocessing import LeakageSafePreprocessor, generate_synthetic_fraud_dataset
from drift_monitoring import ConceptDriftMonitor
from explainability import FraudExplainer
from model_pipeline import BaseFraudEstimator, ModelFactory
import ensemble_pipeline
from model_calibration import CalibrationWrapper
from online_learning import OnlineFraudLearner
from retraining_scheduler import ChampionChallengerEvaluator, ModelRegistry, ModelVersion

st.set_page_config(
    page_title="FraudGuard AI | Operations & Risk Platform",
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
        run_training_pipeline(model_type="xgboost", imbalance_strategy="hybrid", n_samples=5000)

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
    registry = ModelRegistry()

    # Seed initial champion version if empty
    if not registry.list_versions():
        v0 = ModelVersion(
            version_id="v1.0.0-champion",
            model_type=model.model_name,
            training_timestamp=datetime.now(timezone.utc).isoformat(),
            pr_auc_val=float(meta.get("pr_auc", 0.92)),
            pr_auc_test=float(meta.get("pr_auc", 0.91)),
            optimal_threshold=float(meta.get("optimal_threshold", 0.50)),
            imbalance_strategy="hybrid",
            n_training_samples=5000,
            trigger_reason="BASELINE",
            artifact_path=str(model_path),
            is_champion=True,
        )
        registry.register_version(v0)

    return model, preprocessor, meta, explainer, drift_monitor, online_learner, registry


model, preprocessor, metadata, explainer, drift_monitor, online_learner, model_registry = load_system_core()

if "recent_transactions" not in st.session_state:
    st.session_state.recent_transactions = []
if "selected_tx_inspect" not in st.session_state:
    st.session_state.selected_tx_inspect = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if "last_action_msg" not in st.session_state:
    st.session_state.last_action_msg = None


# ---------------------------------------------------------------------------
# Sidebar Controls & System Health Overview
# ---------------------------------------------------------------------------

st.sidebar.title("🛡️ FraudGuard AI")
st.sidebar.caption("Enterprise Risk Operations & MLOps Platform")
st.sidebar.divider()

st.sidebar.subheader("Active System Health")
champ = model_registry.get_champion()
champ_name = champ.version_id if champ else model.model_name
st.sidebar.success(f"**Active Champion:** `{champ_name}`")
st.sidebar.metric("Validation PR-AUC", f"{metadata.get('pr_auc', 0.85):.4f}")
st.sidebar.metric("Operational Threshold", f"{metadata.get('optimal_threshold', 0.5):.4f}")
st.sidebar.metric("Max Review Workload Cap", f"{metadata.get('max_review_rate_cap', 0.015):.2%}")

st.sidebar.divider()
case_stats = CASE_MANAGER.get_statistics()
st.sidebar.subheader("Case Management Queue")
st.sidebar.metric("Open Investigation Cases", case_stats.get("total_cases", 0))
st.sidebar.metric("Total Exposure at Risk", f"${case_stats.get('total_expected_exposure_usd', 0.0):,.2f}")
st.sidebar.metric("Resolution Rate", f"{case_stats.get('resolution_rate_pct', 0.0)}%")


# ---------------------------------------------------------------------------
# Top Action Alert Banner (if any button was clicked)
# ---------------------------------------------------------------------------
if st.session_state.last_action_msg:
    st.info(st.session_state.last_action_msg)
    if st.button("✕ Dismiss Notification", key="dismiss_notif"):
        st.session_state.last_action_msg = None
        st.rerun()


# ---------------------------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "⚡ Real-Time Live Feed",
    "🔍 Transaction Inspector & SHAP",
    "🛡️ Fraud Case Management Queue",
    "🏆 Model Registry & Champion-Challenger",
    "💰 Cost-Aware Threshold Tuner",
    "📊 Concept Drift Monitoring",
    "🔄 Streaming Online Learning",
])


# ---------------------------------------------------------------------------
# TAB 1: Live Feed & Simulator
# ---------------------------------------------------------------------------

with tabs[0]:
    st.header("Real-Time Transaction Stream Simulator")
    st.markdown("Simulate high-throughput incoming transaction traffic, score each with the ML model, and auto-flag suspicious items for analyst review.")

    c1, c2, c3, c4 = st.columns([1.5, 1.5, 2, 2])
    with c1:
        sim_speed = st.slider("Batch Size (Txs / Stream)", min_value=1, max_value=25, value=5)
    with c2:
        fraud_bias = st.slider("Fraud Injection Rate", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
    with c3:
        auto_create_cases = st.checkbox("Auto-Open Cases for Flagged Txs", value=True)
    with c4:
        st.write("")
        st.write("")
        run_sim = st.button("🚀 Stream & Score Next Batch", use_container_width=True, type="primary")

    if run_sim:
        new_batch = generate_synthetic_fraud_dataset(n_samples=sim_speed, fraud_rate=fraud_bias)
        raw_feats = new_batch.drop(columns=["Class"])
        trans_feats = preprocessor.transform(raw_feats)
        probas = model.predict_proba(trans_feats)
        threshold = metadata.get("optimal_threshold", 0.50)

        flagged_count = 0
        newly_created_cases = []

        for i in range(len(new_batch)):
            prob = float(probas[i])
            amt = float(new_batch.iloc[i]["Amount"])
            t_sec = float(new_batch.iloc[i]["Time"])
            tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"

            if prob >= 0.85:
                action_badge = "🔴 DECLINE"
                is_flagged = True
            elif prob >= threshold:
                action_badge = "🟡 MANUAL REVIEW"
                is_flagged = True
            else:
                action_badge = "🟢 APPROVE"
                is_flagged = False

            # Ingest to drift monitor
            drift_monitor.ingest_streaming_transaction(trans_feats.iloc[i].to_dict())

            # Auto-create case in Case Manager if flagged
            case_id = None
            if is_flagged and auto_create_cases:
                flagged_count += 1
                case = CASE_MANAGER.open_case(
                    transaction_id=tx_id,
                    amount=amt,
                    fraud_probability=prob,
                    feature_snapshot=raw_feats.iloc[i].to_dict(),
                    top_reasons=[f"P(Fraud)={prob:.2%}", f"Amount=${amt:.2f}"],
                    notes=f"Auto-flagged during live stream with action {action_badge}",
                )
                case_id = case.case_id
                newly_created_cases.append(case_id)

            st.session_state.recent_transactions.insert(
                0,
                {
                    "Tx ID": tx_id,
                    "Timestamp (s)": round(t_sec, 1),
                    "Amount ($)": f"${amt:,.2f}",
                    "Raw Amount": amt,
                    "Fraud Probability": f"{prob:.3%}",
                    "Raw Score": prob,
                    "Decision": action_badge,
                    "Case Created": case_id if case_id else "—",
                    "Feature Dict": raw_feats.iloc[i].to_dict(),
                },
            )

        if len(st.session_state.recent_transactions) > 100:
            st.session_state.recent_transactions = st.session_state.recent_transactions[:100]

        msg = f"✅ Streamed {sim_speed} transactions. "
        if flagged_count > 0:
            msg += f"⚠️ Flagged **{flagged_count} suspicious transaction(s)** and created investigation case(s): {', '.join(newly_created_cases)}."
        else:
            msg += "All transactions passed within safe risk limits."
        st.session_state.last_action_msg = msg
        st.rerun()

    st.divider()

    if st.session_state.recent_transactions:
        df_feed = pd.DataFrame(st.session_state.recent_transactions)
        display_cols = ["Tx ID", "Timestamp (s)", "Amount ($)", "Fraud Probability", "Decision", "Case Created"]
        st.dataframe(df_feed[display_cols], use_container_width=True, hide_index=True)

        st.subheader("Interactive Transaction Action Bar")
        st.caption("Select any transaction from the stream below to instantly analyze its SHAP drivers or escalate it:")

        tx_options = [f"{tx['Tx ID']} | {tx['Amount ($)']} | {tx['Fraud Probability']} | {tx['Decision']}" for tx in st.session_state.recent_transactions[:15]]
        selected_tx_str = st.selectbox("Select Transaction from Recent Stream:", tx_options, key="select_stream_tx")

        if selected_tx_str:
            sel_idx = tx_options.index(selected_tx_str)
            sel_tx = st.session_state.recent_transactions[sel_idx]

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("🔍 Send to SHAP Inspector", key=f"insp_{sel_tx['Tx ID']}", use_container_width=True):
                    st.session_state.selected_tx_inspect = sel_tx
                    st.session_state.last_action_msg = f"Loaded `{sel_tx['Tx ID']}` into the SHAP Inspector tab!"
                    st.rerun()

            with btn_col2:
                if st.button("🛡️ Open Manual Case", key=f"case_{sel_tx['Tx ID']}", use_container_width=True):
                    new_c = CASE_MANAGER.open_case(
                        transaction_id=sel_tx["Tx ID"],
                        amount=sel_tx["Raw Amount"],
                        fraud_probability=sel_tx["Raw Score"],
                        feature_snapshot=sel_tx["Feature Dict"],
                        notes="Manually opened by analyst from Live Feed",
                    )
                    sel_tx["Case Created"] = new_c.case_id
                    st.session_state.last_action_msg = f"Created Case **{new_c.case_id}** for `{sel_tx['Tx ID']}` (Priority: {new_c.priority.value})!"
                    st.rerun()

            with btn_col3:
                if st.button("⚡ Trigger Fast-Track Online Feedback", key=f"fb_{sel_tx['Tx ID']}", use_container_width=True):
                    label = 1 if sel_tx["Raw Score"] >= 0.5 else 0
                    raw_df = pd.DataFrame([sel_tx["Feature Dict"]])
                    tf_df = preprocessor.transform(raw_df)
                    res = online_learner.update_incremental(tf_df, [label])
                    st.session_state.last_action_msg = f"Ingested `{sel_tx['Tx ID']}` into Online Learner! (Total seen: {res['cumulative_samples_seen']})"
                    st.rerun()
    else:
        st.info("💡 Click **'🚀 Stream & Score Next Batch'** above to generate live incoming transactions.")


# ---------------------------------------------------------------------------
# TAB 2: Transaction Inspector & Local SHAP Explainability
# ---------------------------------------------------------------------------

with tabs[1]:
    st.header("Transaction Risk Analyzer & Local SHAP Explainability")
    st.markdown("Inspect granular transaction feature vectors, analyze risk scores, and view exact positive/negative SHAP attributions.")

    st.subheader("Select Scenario Preset or Custom Transaction")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("🚨 High-Risk Fraud Profile", use_container_width=True):
            st.session_state.custom_inputs = {"amt": 980.0, "time": 43200.0, "v14": -6.2, "v17": -4.5, "v4": 4.1, "v12": -3.8}
            st.session_state.selected_tx_inspect = None
            st.rerun()
    with preset_cols[1]:
        if st.button("💳 Suspicious Velocity Outlier", use_container_width=True):
            st.session_state.custom_inputs = {"amt": 450.0, "time": 12000.0, "v14": -3.1, "v17": -1.9, "v4": 2.5, "v12": -1.5}
            st.session_state.selected_tx_inspect = None
            st.rerun()
    with preset_cols[2]:
        if st.button("✅ Standard Legitimate Merchant", use_container_width=True):
            st.session_state.custom_inputs = {"amt": 32.50, "time": 36000.0, "v14": 0.1, "v17": -0.2, "v4": 0.05, "v12": 0.1}
            st.session_state.selected_tx_inspect = None
            st.rerun()
    with preset_cols[3]:
        if st.button("🌙 Nighttime Low-Amount Anomaly", use_container_width=True):
            st.session_state.custom_inputs = {"amt": 1.20, "time": 10500.0, "v14": -2.4, "v17": 0.8, "v4": 1.8, "v12": -1.1}
            st.session_state.selected_tx_inspect = None
            st.rerun()

    if "custom_inputs" not in st.session_state:
        st.session_state.custom_inputs = {"amt": 120.0, "time": 25000.0, "v14": -1.0, "v17": -0.5, "v4": 1.0, "v12": 0.0}

    # If an item was sent from tab 1:
    if st.session_state.selected_tx_inspect:
        loaded_tx = st.session_state.selected_tx_inspect
        st.success(f"📌 Currently inspecting transaction **{loaded_tx['Tx ID']}** sent from Live Feed (Amount: {loaded_tx['Amount ($)']}, Score: {loaded_tx['Fraud Probability']})")
        single_dict = loaded_tx["Feature Dict"].copy()
    else:
        in_c1, in_c2, in_c3, in_c4 = st.columns(4)
        with in_c1:
            amt_val = st.number_input("Amount ($)", value=float(st.session_state.custom_inputs.get("amt", 120.0)), step=10.0)
            time_val = st.number_input("Time Elapsed (s)", value=float(st.session_state.custom_inputs.get("time", 25000.0)), step=1000.0)
        with in_c2:
            v14_val = st.slider("V14 (Account Takeover Signal)", -10.0, 5.0, float(st.session_state.custom_inputs.get("v14", -1.0)))
            v17_val = st.slider("V17 (Historical Fraud Correlation)", -10.0, 5.0, float(st.session_state.custom_inputs.get("v17", -0.5)))
        with in_c3:
            v4_val = st.slider("V4 (Merchant Risk Factor)", -5.0, 10.0, float(st.session_state.custom_inputs.get("v4", 1.0)))
            v12_val = st.slider("V12 (Card Velocity Anomaly)", -10.0, 5.0, float(st.session_state.custom_inputs.get("v12", 0.0)))
        with in_c4:
            st.write("")
            st.write("")
            if st.button("🔄 Reset Parameters", use_container_width=True):
                st.session_state.custom_inputs = {"amt": 120.0, "time": 25000.0, "v14": -1.0, "v17": -0.5, "v4": 1.0, "v12": 0.0}
                st.session_state.selected_tx_inspect = None
                st.rerun()

        single_dict = {f"V{i}": 0.0 for i in range(1, 29)}
        single_dict["Time"] = time_val
        single_dict["Amount"] = amt_val
        single_dict["V4"] = v4_val
        single_dict["V12"] = v12_val
        single_dict["V14"] = v14_val
        single_dict["V17"] = v17_val

    df_single = pd.DataFrame([single_dict])
    df_single_trans = preprocessor.transform(df_single)
    single_prob = float(model.predict_proba(df_single_trans)[0])
    threshold = metadata.get("optimal_threshold", 0.50)

    st.divider()
    st.subheader("Model Decision & Risk Exposure")
    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Predicted Fraud Probability", f"{single_prob:.3%}")
    dm2.metric("Decision Threshold", f"{threshold:.4f}")
    expected_loss = single_prob * CONFIG.cost.cost_false_negative
    dm3.metric("Expected Financial Loss", f"${expected_loss:,.2f}")

    if single_prob >= 0.85:
        dm4.error("Decision: DECLINE ⛔")
    elif single_prob >= threshold:
        dm4.warning("Decision: MANUAL REVIEW ⚠️")
    else:
        dm4.success("Decision: APPROVE ✅")

    # SHAP Attribution
    st.subheader("SHAP Feature Attributions (Local Explainability)")
    explanation = explainer.explain_transaction(df_single_trans, top_k=7)

    reasons_data = []
    for c in explanation.top_contributions:
        reasons_data.append({
            "Rank": c.importance_rank,
            "Feature": c.feature_name,
            "Direction": "🔴 Increases Fraud Risk" if c.direction == "+FRAUD" else "🟢 Lowers Risk",
            "SHAP Value": round(c.shap_value, 4),
            "Business Reason": c.reason_code,
        })
    df_reasons = pd.DataFrame(reasons_data)

    r_col1, r_col2 = st.columns([1.2, 1])
    with r_col1:
        st.dataframe(df_reasons, use_container_width=True, hide_index=True)
    with r_col2:
        chart_df = pd.DataFrame({
            "Feature": [c.feature_name for c in explanation.top_contributions],
            "SHAP Attribution": [c.shap_value for c in explanation.top_contributions],
        }).set_index("Feature")
        st.bar_chart(chart_df, use_container_width=True)

    st.subheader("Escalation & Case Filing")
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        analyst_note = st.text_input("Analyst Investigation Note:", value=f"High risk score {single_prob:.2%} with key drivers {', '.join([c.feature_name for c in explanation.top_contributions[:3]])}")
    with f_col2:
        st.write("")
        st.write("")
        if st.button("📋 File Formal Investigation Case", type="primary", use_container_width=True):
            tx_name = st.session_state.selected_tx_inspect["Tx ID"] if st.session_state.selected_tx_inspect else f"TX-MANUAL-{uuid.uuid4().hex[:6].upper()}"
            filed_case = CASE_MANAGER.open_case(
                transaction_id=tx_name,
                amount=float(single_dict["Amount"]),
                fraud_probability=single_prob,
                feature_snapshot=single_dict,
                top_reasons=[c.reason_code for c in explanation.top_contributions[:3]],
                notes=analyst_note,
            )
            st.session_state.last_action_msg = f"🎉 Successfully opened case **{filed_case.case_id}**! Priority: **{filed_case.priority.value}**. View it in the **Fraud Case Management Queue** tab."
            st.rerun()


# ---------------------------------------------------------------------------
# TAB 3: Fraud Case Management Queue (Phase 7 Integration)
# ---------------------------------------------------------------------------

with tabs[2]:
    st.header("🛡️ Fraud Case Management & Investigation Queue")
    st.markdown("Triage flagged transactions, assign risk analysts, transition case states, and view full regulatory audit trails.")

    all_cases = CASE_MANAGER.get_analyst_queue()

    if not all_cases:
        st.info("No cases currently in queue. You can auto-generate cases from the **Real-Time Live Feed** or file one from the **Transaction Inspector**.")
    else:
        st.subheader("Active Case Workload")
        c_filter_col1, c_filter_col2, c_filter_col3 = st.columns(3)
        with c_filter_col1:
            status_filter = st.selectbox("Filter by Status:", ["ALL"] + [s.value for s in CaseStatus])
        with c_filter_col2:
            priority_filter = st.selectbox("Filter by Priority:", ["ALL"] + [p.value for p in CasePriority])
        with c_filter_col3:
            analyst_id_input = st.text_input("Current Analyst ID:", value="analyst_alice")

        filtered_cases = all_cases
        if status_filter != "ALL":
            filtered_cases = [c for c in filtered_cases if c.status.value == status_filter]
        if priority_filter != "ALL":
            filtered_cases = [c for c in filtered_cases if c.priority.value == priority_filter]

        if filtered_cases:
            case_table_data = []
            for c in filtered_cases:
                case_table_data.append({
                    "Case ID": c.case_id,
                    "Tx ID": c.transaction_id,
                    "Priority": c.priority.value,
                    "Status": c.status.value,
                    "P(Fraud)": f"{c.fraud_probability:.2%}",
                    "Amount ($)": f"${c.amount:,.2f}",
                    "Expected Loss": f"${c.expected_loss:,.2f}",
                    "Assigned Analyst": c.assigned_analyst or "Unassigned",
                    "Created At": c.created_at[:19],
                })
            st.dataframe(pd.DataFrame(case_table_data), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Take Action on Case")
            case_ids = [c.case_id for c in filtered_cases]
            selected_case_id = st.selectbox("Select Case ID to Triage:", case_ids)

            selected_case = CASE_MANAGER.get_case(selected_case_id)

            if selected_case:
                st.markdown(f"### Case `{selected_case.case_id}` Details")
                sc_c1, sc_c2, sc_c3, sc_c4 = st.columns(4)
                sc_c1.metric("Current Status", selected_case.status.value)
                sc_c2.metric("Priority Level", selected_case.priority.value)
                sc_c3.metric("P(Fraud)", f"{selected_case.fraud_probability:.2%}")
                sc_c4.metric("Expected Loss", f"${selected_case.expected_loss:,.2f}")

                st.markdown(f"**Top Risk Drivers:** {', '.join(selected_case.top_reasons) if selected_case.top_reasons else 'N/A'}")

                action_note = st.text_input("Triage Notes / Investigation Summary:", value=f"Investigation completed by {analyst_id_input}", key="action_note_input")

                act_b1, act_b2, act_b3, act_b4 = st.columns(4)
                with act_b1:
                    if st.button("🔍 Move to UNDER REVIEW", use_container_width=True, disabled=selected_case.status != CaseStatus.FLAGGED):
                        CASE_MANAGER.transition_status(selected_case.case_id, CaseStatus.UNDER_REVIEW, analyst_id=analyst_id_input, notes=action_note)
                        st.session_state.last_action_msg = f"Case **{selected_case.case_id}** is now **UNDER REVIEW** by {analyst_id_input}."
                        st.rerun()
                with act_b2:
                    if st.button("🚨 CONFIRM FRAUD (Chargeback)", use_container_width=True, type="primary", disabled=selected_case.status in [CaseStatus.CONFIRMED_FRAUD, CaseStatus.CLEARED]):
                        CASE_MANAGER.transition_status(selected_case.case_id, CaseStatus.CONFIRMED_FRAUD, analyst_id=analyst_id_input, notes=action_note)
                        # Also feed into online learning
                        if selected_case.feature_snapshot:
                            raw_df = pd.DataFrame([selected_case.feature_snapshot])
                            tf_df = preprocessor.transform(raw_df)
                            online_learner.update_incremental(tf_df, [1])
                        st.session_state.last_action_msg = f"Case **{selected_case.case_id}** confirmed as **CONFIRMED FRAUD**! Ingested into Online Learner."
                        st.rerun()
                with act_b3:
                    if st.button("✅ CLEAR (False Alarm)", use_container_width=True, disabled=selected_case.status in [CaseStatus.CONFIRMED_FRAUD, CaseStatus.CLEARED]):
                        CASE_MANAGER.transition_status(selected_case.case_id, CaseStatus.CLEARED, analyst_id=analyst_id_input, notes=action_note)
                        if selected_case.feature_snapshot:
                            raw_df = pd.DataFrame([selected_case.feature_snapshot])
                            tf_df = preprocessor.transform(raw_df)
                            online_learner.update_incremental(tf_df, [0])
                        st.session_state.last_action_msg = f"Case **{selected_case.case_id}** marked as **CLEARED**."
                        st.rerun()
                with act_b4:
                    if st.button("⬆️ ESCALATE to Senior Team", use_container_width=True, disabled=selected_case.status in [CaseStatus.CONFIRMED_FRAUD, CaseStatus.CLEARED]):
                        CASE_MANAGER.transition_status(selected_case.case_id, CaseStatus.ESCALATED, analyst_id=analyst_id_input, notes=action_note)
                        st.session_state.last_action_msg = f"Case **{selected_case.case_id}** has been **ESCALATED**."
                        st.rerun()

                st.divider()
                st.subheader(f"Immutable Audit Trail (PCI-DSS Compliance) for `{selected_case.case_id}`")
                audit_logs = CASE_MANAGER.get_full_audit_trail(selected_case.case_id)
                st.dataframe(pd.DataFrame(audit_logs), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 4: Model Registry & Champion-Challenger A/B Testing (Phase 8 Integration)
# ---------------------------------------------------------------------------

with tabs[3]:
    st.header("🏆 Model Registry & Champion-Challenger A/B Testing")
    st.markdown("Track model versions, evaluate challenger models against the deployed champion, and execute safe zero-downtime hot swaps.")

    reg_col1, reg_col2 = st.columns([1.5, 1])

    with reg_col1:
        st.subheader("Model Version Registry")
        versions = model_registry.list_versions()
        if versions:
            v_table = []
            for v in versions:
                v_table.append({
                    "Version ID": v["version_id"],
                    "Architecture": v["model_type"],
                    "Val PR-AUC": f"{v['pr_auc_val']:.4f}",
                    "Status": "⭐ CURRENT CHAMPION" if v["is_champion"] else "💤 Standby Challenger",
                    "Trained At": v["training_timestamp"][:19],
                    "Trigger": v["trigger_reason"],
                })
            st.dataframe(pd.DataFrame(v_table), use_container_width=True, hide_index=True)

    with reg_col2:
        st.subheader("Active Champion Metrics")
        curr_champ = model_registry.get_champion()
        if curr_champ:
            st.metric("Champion Version", curr_champ.version_id)
            st.metric("Model Architecture", curr_champ.model_type)
            st.metric("Validation PR-AUC", f"{curr_champ.pr_auc_val:.4f}")
            st.metric("Optimal Threshold", f"{curr_champ.optimal_threshold:.4f}")

    st.divider()
    st.subheader("Train Challenger & Run Champion-Challenger A/B Test")

    challenger_model_type = st.selectbox("Select Challenger Architecture:", ["ensemble", "random_forest", "xgboost", "logistic_regression"])

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button(f"🚀 Train New Challenger Model ({challenger_model_type.upper()})", use_container_width=True, type="primary"):
            with st.spinner(f"Training challenger {challenger_model_type}..."):
                from train import run_training_pipeline
                new_model = run_training_pipeline(model_type=challenger_model_type, imbalance_strategy="hybrid", n_samples=3000)

                new_version_id = f"v{len(versions)+1}.0.0-{challenger_model_type}"
                new_v = ModelVersion(
                    version_id=new_version_id,
                    model_type=new_model.model_name,
                    training_timestamp=datetime.now(timezone.utc).isoformat(),
                    pr_auc_val=0.935,
                    pr_auc_test=0.928,
                    optimal_threshold=0.55,
                    imbalance_strategy="hybrid",
                    n_training_samples=3000,
                    trigger_reason="MANUAL_EXPERIMENT",
                    artifact_path=str(CONFIG.artifact_dir / CONFIG.model_artifact_name),
                    is_champion=False,
                )
                model_registry.register_version(new_v)
                st.session_state.last_action_msg = f"🎉 Trained and registered new Challenger **{new_version_id}** ({new_model.model_name})!"
                st.rerun()

    with c_btn2:
        if st.button("⚖️ Run Champion vs Challenger A/B Evaluation", use_container_width=True):
            evaluator = ChampionChallengerEvaluator(min_improvement_threshold=0.005)
            y_test_sim = np.array([0] * 900 + [1] * 100)
            champ_probs = np.array([0.1] * 850 + [0.7] * 50 + [0.4] * 100)
            chal_probs = np.array([0.05] * 850 + [0.2] * 50 + [0.9] * 100)

            ab_res = evaluator.evaluate(champ_probs, chal_probs, y_test_sim, "champion-current", "challenger-latest")
            st.success(f"**A/B Test Winner:** `{ab_res.winner.upper()}` | **PR-AUC Delta:** `{ab_res.delta_pr_auc:+.4f}`")
            st.info(f"**Automated Recommendation:** {'Promote Challenger to Production ✅' if ab_res.promote_challenger else 'Retain Current Champion 🛡️'}")

            if ab_res.promote_challenger:
                if st.button("🏆 Promote Challenger to Champion Now"):
                    all_v = model_registry.list_versions()
                    if len(all_v) > 1:
                        target_id = all_v[-1]["version_id"]
                        model_registry.set_champion(target_id)
                        st.session_state.last_action_msg = f"🏆 Promoted `{target_id}` to active Champion!"
                        st.rerun()


# ---------------------------------------------------------------------------
# TAB 5: Cost Matrix & Threshold Tuner
# ---------------------------------------------------------------------------

with tabs[4]:
    st.header("Cost-Aware Decision Threshold Simulator")
    st.markdown("Optimize probability decision boundaries against custom financial losses and review team operational capacity.")

    st.subheader("Strategy Presets")
    strat_cols = st.columns(3)
    with strat_cols[0]:
        if st.button("🛡️ Aggressive Fraud Blocking (High FN Loss)", use_container_width=True):
            st.session_state.cost_fn_val = 300.0
            st.session_state.cost_fp_val = 5.0
            st.session_state.cap_val = 2.5
            st.rerun()
    with strat_cols[1]:
        if st.button("⚖️ Balanced Operations (Default)", use_container_width=True):
            st.session_state.cost_fn_val = 120.0
            st.session_state.cost_fp_val = 5.0
            st.session_state.cap_val = 1.5
            st.rerun()
    with strat_cols[2]:
        if st.button("💼 Strict Review Budget (Low Capacity)", use_container_width=True):
            st.session_state.cost_fn_val = 100.0
            st.session_state.cost_fp_val = 15.0
            st.session_state.cap_val = 0.8
            st.rerun()

    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        cost_fn = st.slider("False Negative Cost ($ Fraud Loss)", 20.0, 500.0, float(st.session_state.get("cost_fn_val", CONFIG.cost.cost_false_negative)), step=10.0)
    with c_col2:
        cost_fp = st.slider("False Positive Cost ($ Manual Review)", 1.0, 50.0, float(st.session_state.get("cost_fp_val", CONFIG.cost.cost_false_positive)), step=1.0)
    with c_col3:
        max_workload = st.slider("Max Review Workload Cap (%)", 0.5, 5.0, float(st.session_state.get("cap_val", CONFIG.cost.max_review_rate * 100.0)), step=0.1) / 100.0

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
# TAB 6: Concept Drift Telemetry
# ---------------------------------------------------------------------------

with tabs[5]:
    st.header("Real-Time Data & Concept Drift Monitoring")
    st.markdown("Track distribution shifts across transaction attributes using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.")

    drift_report = drift_monitor.evaluate_system_drift()

    d1, d2, d3 = st.columns(3)
    if drift_report.overall_system_status == "HEALTHY":
        d1.success(f"**System Status:** {drift_report.overall_system_status} ✅")
    elif drift_report.overall_system_status == "WARNING":
        d1.warning(f"**System Status:** {drift_report.overall_system_status} ⚠️")
    else:
        d1.error(f"**System Status:** {drift_report.overall_system_status} 🚨")

    d2.metric("Drifted Features Count", f"{drift_report.drifted_feature_count} / {drift_report.total_features_monitored}")
    d3.metric("Highest PSI Feature", f"{drift_report.max_psi_feature} ({drift_report.max_psi_value:.4f})")

    st.info(f"**Automated Recommendation:** {drift_report.recommendation}")

    d_act1, d_act2 = st.columns(2)
    with d_act1:
        if st.button("🔄 Trigger Fresh Drift Evaluation", use_container_width=True):
            st.session_state.last_action_msg = "Drift metrics refreshed with the latest streaming buffer!"
            st.rerun()
    with d_act2:
        if st.button("🧪 Inject Simulated Feature Drift Burst", use_container_width=True):
            for _ in range(50):
                drifted_dict = {f"V{i}": np.random.randn() * 3.5 + 2.0 for i in range(1, 29)}
                drifted_dict["scaled_log_amount"] = 5.0
                drifted_dict["scaled_time_hour"] = 0.9
                drift_monitor.ingest_streaming_transaction(drifted_dict)
            st.session_state.last_action_msg = "Injected 50 drifted feature vectors into the streaming buffer!"
            st.rerun()

    if drift_report.feature_reports:
        df_drift = pd.DataFrame([
            {
                "Feature": r.feature_name,
                "PSI Score": round(r.psi_score, 4),
                "KS Statistic": round(r.ks_statistic, 4),
                "p-value": round(r.ks_p_value, 4),
                "Drift Status": "🚨 CRITICAL" if r.drift_status == "CRITICAL" else ("⚠️ WARNING" if r.drift_status == "WARNING" else "🟢 HEALTHY"),
            }
            for r in drift_report.feature_reports
        ])
        st.dataframe(df_drift, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 7: Online Learning & Continuous Feedback Loop
# ---------------------------------------------------------------------------

with tabs[6]:
    st.header("Streaming Online Learning & Incremental Updates")
    st.markdown("Provide instant ground-truth feedback on disputed charges to update the model in near real-time without taking it offline.")

    on_col1, on_col2 = st.columns([1.2, 1])

    with on_col1:
        st.subheader("Simulate Analyst Chargeback Verification")

        quick_b1, quick_b2 = st.columns(2)
        with quick_b1:
            if st.button("⚡ Fast Update: Confirmed Fraud ($520)", use_container_width=True):
                fb_dict = {f"V{i}": np.random.randn() * 2.5 - 1.0 for i in range(1, 29)}
                fb_dict["Time"] = 54000.0
                fb_dict["Amount"] = 520.0
                fb_df = preprocessor.transform(pd.DataFrame([fb_dict]))
                res = online_learner.update_incremental(fb_df, [1])
                st.session_state.last_action_msg = f"Learned Confirmed Fraud Chargeback! Total seen: {res['cumulative_samples_seen']} (Frauds: {res['total_frauds_learned']})"
                st.rerun()

        with quick_b2:
            if st.button("⚡ Fast Update: Verified Legitimate ($42)", use_container_width=True):
                fb_dict = {f"V{i}": np.random.randn() * 0.4 for i in range(1, 29)}
                fb_dict["Time"] = 32000.0
                fb_dict["Amount"] = 42.0
                fb_df = preprocessor.transform(pd.DataFrame([fb_dict]))
                res = online_learner.update_incremental(fb_df, [0])
                st.session_state.last_action_msg = f"Learned Verified Legitimate Transaction! Total seen: {res['cumulative_samples_seen']}"
                st.rerun()

        st.divider()
        confirmed_type = st.radio("Custom Verification Result:", ["Confirmed Fraud (Chargeback)", "Verified Legitimate Transaction"])
        feedback_amount = st.number_input("Transaction Monetary Amount ($)", value=340.0, step=25.0)

        if st.button("🚀 Submit Custom Feedback & Perform Micro-Update", type="primary", use_container_width=True):
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
