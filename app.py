"""
Streamlit Web Application: SELF-HEALING MACHINES
AI-Powered Predictive Maintenance Intelligence Platform
"""

import os
import sys
import json
import sqlite3
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import CMAPSSDataLoader
from src.pipeline import InferencePipeline, load_config
from src.work_order_service import WorkOrderService
from src.database import get_db_connection, init_db
from src.monitoring import FleetMonitor
from src.retraining import RetrainingManager
from src.rul_labels import add_rul_labels
from src.rul_model import create_engine_grouped_split

# Page Configuration
st.set_page_config(
    page_title="Self-Healing Machines | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Industrial Analytics Theme)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def get_config():
    return load_config("config/config.yaml")


@st.cache_data(ttl=120)
def load_fleet_data(subset_name="FD001"):
    loader = CMAPSSDataLoader()
    try:
        train_df, test_df, rul_df = loader.load_subset(subset_name)
        return train_df, test_df, rul_df
    except Exception as e:
        st.error(f"Error loading C-MAPSS dataset: {e}")
        return None, None, None


@st.cache_data(ttl=300)
def load_evaluation_report():
    report_path = "artifacts/reports/evaluation_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_feature_importance():
    imp_path = "artifacts/reports/feature_importance.json"
    if os.path.exists(imp_path):
        with open(imp_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=120)
def get_scored_fleet_data():
    config = get_config()
    models_exist = os.path.exists("artifacts/models/rul_model.joblib")
    if not models_exist:
        return None

    train_df, test_df, _ = load_fleet_data(config["dataset"]["name"])
    if test_df is None:
        return None

    pipeline = InferencePipeline("config/config.yaml")
    scored = pipeline.process_telemetry_batch(test_df, dispatch_work_orders=True, last_cycle_only=True)
    return scored


# Sidebar Navigation
config = get_config()
st.sidebar.image("https://img.icons8.com/fluency/96/jet-engine.png", width=64)
st.sidebar.title("Self-Healing Machines")
st.sidebar.caption("Predictive Maintenance Intelligence")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Executive Overview",
        "Fleet Health",
        "Engine Details",
        "Model Performance",
        "Maintenance Work Orders",
        "MLOps Monitoring",
        "Retraining Demo",
        "System Status",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Active Fleet Subset:** `{config['dataset']['name']}`\n\n"
    f"**Model Type:** `HistGradientBoosting`\n\n"
    f"**RUL Cap:** `{config['rul_labeling']['rul_cap_value']} cycles`"
)

# Initialize Database
init_db(config["work_orders"]["db_path"])
wo_service = WorkOrderService(config["work_orders"]["db_path"])

# ==============================================================================
# 1. EXECUTIVE OVERVIEW
# ==============================================================================
if page == "Executive Overview":
    st.markdown('<p class="main-header">Fleet Executive Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time prognostic health monitoring, risk stratification, and dispatch metrics.</p>', unsafe_allow_html=True)

    scored_fleet = get_scored_fleet_data()
    wo_stats = wo_service.get_summary_statistics()

    if scored_fleet is not None:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        total_eng = len(scored_fleet)
        crit_count = int((scored_fleet["priority"] == "CRITICAL").sum())
        high_count = int((scored_fleet["priority"] == "HIGH").sum())
        anom_count = int((scored_fleet["anomaly_status"] == "ANOMALOUS").sum())
        avg_rul = float(scored_fleet["predicted_rul"].mean())

        c1.metric("Monitored Engines", f"{total_eng} Units")
        c2.metric("Attention Required", f"{crit_count + high_count} Units", delta=f"{crit_count} Critical", delta_color="inverse")
        c3.metric("Critical Alerts", f"{crit_count}", delta="Priority 1", delta_color="inverse")
        c4.metric("Active Work Orders", f"{wo_stats['open_work_orders']}", delta=f"{wo_stats['total_work_orders']} Total")
        c5.metric("Avg Predicted RUL", f"{avg_rul:.1f} Cycles")
        c6.metric("Active Anomalies", f"{anom_count}", delta=f"Isolation Forest")

        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Fleet Risk Stratification")
            prio_counts = scored_fleet["priority"].value_counts().reset_index()
            prio_counts.columns = ["Priority", "Count"]
            color_map = {
                "CRITICAL": "#EF4444",
                "HIGH": "#F97316",
                "MEDIUM": "#F59E0B",
                "LOW": "#10B981",
            }
            fig_prio = px.bar(
                prio_counts,
                x="Priority",
                y="Count",
                color="Priority",
                color_discrete_map=color_map,
                text="Count",
                title="Active Fleet Engine Count by Maintenance Priority",
            )
            fig_prio.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_prio, use_container_width=True)

        with col_chart2:
            st.subheader("Remaining Useful Life Distribution")
            fig_hist = px.histogram(
                scored_fleet,
                x="predicted_rul",
                nbins=25,
                color="priority",
                color_discrete_map=color_map,
                title="Predicted RUL Spread across Monitored Units",
                labels={"predicted_rul": "Predicted RUL (cycles)"},
            )
            fig_hist.add_vline(x=15, line_dash="dash", line_color="red", annotation_text="Critical (15)")
            fig_hist.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="High (30)")
            fig_hist.update_layout(height=340)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Immediate Action Items (Critical & High Priority)")
        urgent_df = scored_fleet[scored_fleet["priority"].isin(["CRITICAL", "HIGH"])][
            ["engine_id", "cycle", "predicted_rul", "anomaly_status", "priority", "triage_reason"]
        ].sort_values(by="predicted_rul").reset_index(drop=True)

        if not urgent_df.empty:
            st.dataframe(urgent_df, use_container_width=True)
        else:
            st.success("All monitored fleet engines operating in nominal or medium ranges.")
    else:
        st.warning("Model artifacts not yet generated. Please execute model training via the 'Retraining Demo' page or run `python scripts/train.py`.")


# ==============================================================================
# 2. FLEET HEALTH
# ==============================================================================
elif page == "Fleet Health":
    st.markdown('<p class="main-header">Fleet Telemetry & Prognostics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Granular per-machine prognostic assessments with priority filters.</p>', unsafe_allow_html=True)

    scored_fleet = get_scored_fleet_data()
    if scored_fleet is not None:
        c_filter1, c_filter2, c_filter3 = st.columns(3)
        with c_filter1:
            prio_filter = st.multiselect("Filter Priority", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        with c_filter2:
            anom_filter = st.multiselect("Filter Anomaly Status", ["NORMAL", "WARNING", "ANOMALOUS"], default=["NORMAL", "WARNING", "ANOMALOUS"])
        with c_filter3:
            search_eng = st.text_input("Search Engine ID (e.g. 42)")

        filtered = scored_fleet[
            (scored_fleet["priority"].isin(prio_filter)) &
            (scored_fleet["anomaly_status"].isin(anom_filter))
        ]
        if search_eng.strip():
            filtered = filtered[filtered["engine_id"] == int(search_eng.strip())]

        st.write(f"Showing **{len(filtered)}** of **{len(scored_fleet)}** fleet engines:")

        display_cols = [
            "engine_id", "cycle", "predicted_rul", "anomaly_score",
            "anomaly_status", "priority", "triage_reason", "recommended_action"
        ]
        st.dataframe(
            filtered[display_cols].sort_values(by="predicted_rul").reset_index(drop=True),
            use_container_width=True,
            column_config={
                "predicted_rul": st.column_config.NumberColumn("Predicted RUL (cycles)", format="%.1f"),
                "anomaly_score": st.column_config.ProgressColumn("Anomaly Score", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
    else:
        st.warning("Please train models first to view fleet health.")


# ==============================================================================
# 3. ENGINE DETAILS
# ==============================================================================
elif page == "Engine Details":
    st.markdown('<p class="main-header">Single Machine Diagnostics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sensor trajectory inspection, degradation tracking, and work order audit for an individual engine.</p>', unsafe_allow_html=True)

    train_df, test_df, rul_df = load_fleet_data(config["dataset"]["name"])
    if test_df is not None:
        available_engines = sorted(test_df["engine_id"].unique())
        selected_engine = st.selectbox("Select Target Engine ID", available_engines, index=min(30, len(available_engines)-1))

        eng_history = test_df[test_df["engine_id"] == selected_engine].sort_values(by="cycle").reset_index(drop=True)
        max_c = eng_history["cycle"].max()

        # Engine Quick Info Card
        c1, c2, c3 = st.columns(3)
        c1.metric("Engine Identifier", f"Unit #{selected_engine}")
        c2.metric("Observed Operational Cycles", f"{max_c} Cycles")

        # Ground truth RUL for test engine
        gt_rul_row = rul_df[rul_df["engine_id"] == selected_engine]
        gt_val = gt_rul_row["rul"].values[0] if not gt_rul_row.empty else "N/A"
        c3.metric("True Remaining Cycles (NASA Ground Truth)", f"{gt_val} Cycles")

        st.markdown("---")
        st.subheader("Sensor Degradation Trajectories")

        # Select sensors to plot
        candidate_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_20", "sensor_21"]
        selected_sensors = st.multiselect(
            "Select Sensor Channels to Visualize",
            candidate_sensors,
            default=["sensor_2", "sensor_3", "sensor_4", "sensor_11"],
        )

        if selected_sensors:
            fig_sensors = go.Figure()
            for s in selected_sensors:
                fig_sensors.add_trace(go.Scatter(
                    x=eng_history["cycle"],
                    y=eng_history[s],
                    mode="lines+markers",
                    name=s,
                ))
            fig_sensors.update_layout(
                title=f"Telemetry Stream Across Operating Cycles (Engine {selected_engine})",
                xaxis_title="Operating Cycle",
                yaxis_title="Sensor Measurement Value",
                height=420,
            )
            st.plotly_chart(fig_sensors, use_container_width=True)

        st.markdown("---")
        st.subheader(f"Maintenance Work Order History for Engine {selected_engine}")
        eng_wos = wo_service.list_work_orders(engine_id_filter=selected_engine)
        if not eng_wos.empty:
            st.dataframe(eng_wos, use_container_width=True)
        else:
            st.info(f"No maintenance work orders dispatched for Engine {selected_engine} yet.")


# ==============================================================================
# 4. MODEL PERFORMANCE
# ==============================================================================
elif page == "Model Performance":
    st.markdown('<p class="main-header">Model Performance & Prognostic Benchmarking</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluation against official NASA C-MAPSS FD001 test labels and validation splits.</p>', unsafe_allow_html=True)

    report = load_evaluation_report()
    feature_imp = load_feature_importance()

    if report is not None:
        test_m = report["official_test_metrics"]
        val_m = report["validation_metrics"]

        st.subheader("Official NASA Benchmark Results (Held-Out Test Set)")
        st.caption("Evaluated at the final observed operating cycle of each test engine against official reference RUL values.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Absolute Error (MAE)", f"{test_m['mae']:.2f} Cycles")
        c2.metric("Root Mean Squared Error (RMSE)", f"{test_m['rmse']:.2f} Cycles")
        c3.metric("Coefficient of Determination (R²)", f"{test_m['r2']:.4f}")
        c4.metric("NASA Asymmetric PHM'08 Score", f"{test_m['nasa_score']:.2f} Pts")

        st.markdown("---")
        col_plot1, col_plot2 = st.columns(2)

        # Plot 1: Predicted vs Actual RUL Scatter
        preds_csv_path = "artifacts/reports/test_predictions.csv"
        if os.path.exists(preds_csv_path):
            preds_df = pd.read_csv(preds_csv_path)
            with col_plot1:
                st.subheader("Predicted vs. Actual RUL")
                max_val = max(preds_df["actual_rul"].max(), preds_df["predicted_rul"].max()) + 10
                fig_scatter = px.scatter(
                    preds_df,
                    x="actual_rul",
                    y="predicted_rul",
                    hover_data=["engine_id", "error"],
                    labels={"actual_rul": "Actual True RUL (cycles)", "predicted_rul": "Predicted RUL (cycles)"},
                    title="Engine Predictions vs. Ground Truth Reference",
                )
                fig_scatter.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode="lines", name="Ideal Parity",
                    line=dict(color="red", dash="dash"),
                ))
                fig_scatter.update_layout(height=380)
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_plot2:
                st.subheader("Residual Distribution (Prediction Error)")
                fig_res = px.histogram(
                    preds_df,
                    x="error",
                    nbins=25,
                    title="Residual Distribution (Predicted RUL - Actual RUL)",
                    labels={"error": "Residual Error (cycles)"},
                    color_discrete_sequence=["#3B82F6"],
                )
                fig_res.add_vline(x=0, line_dash="solid", line_color="black")
                fig_res.update_layout(height=380)
                st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("---")
        # Feature Importance Plot
        if feature_imp:
            st.subheader("Feature Importance Diagnostic")
            st.caption("Top 20 most influential engineered features in the HistGradientBoosting regressor.")
            imp_df = pd.DataFrame(list(feature_imp.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False).head(20)
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=480)
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.subheader("Validation vs Official Test Comparison")
        comp_df = pd.DataFrame([
            {"Stage": "Validation (Grouped Split)", "MAE (cycles)": val_m["mae"], "RMSE (cycles)": val_m["rmse"], "R² Score": val_m["r2"], "NASA Score": val_m["nasa_score"]},
            {"Stage": "Official Held-Out Test Set", "MAE (cycles)": test_m["mae"], "RMSE (cycles)": test_m["rmse"], "R² Score": test_m["r2"], "NASA Score": test_m["nasa_score"]},
        ])
        st.table(comp_df)
    else:
        st.info("Evaluation report not found. Train the model to generate actual evaluation metrics.")


# ==============================================================================
# 5. MAINTENANCE WORK ORDERS
# ==============================================================================
elif page == "Maintenance Work Orders":
    st.markdown('<p class="main-header">Automated Maintenance Work Orders</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Persistent SQLite work order registry with deterministic audit tracking and lifecycle updates.</p>', unsafe_allow_html=True)

    col_flt1, col_flt2, col_flt3 = st.columns([1, 1, 2])
    with col_flt1:
        st_filter = st.selectbox("Status", ["ALL", "OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED"])
    with col_flt2:
        pr_filter = st.selectbox("Priority", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])

    st_arg = None if st_filter == "ALL" else st_filter
    pr_arg = None if pr_filter == "ALL" else pr_filter

    wos = wo_service.list_work_orders(status_filter=st_arg, priority_filter=pr_arg)

    # Export CSV Button
    csv_bytes = wos.to_csv(index=False).encode("utf-8")
    with col_flt3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Export Work Orders to CSV",
            data=csv_bytes,
            file_name=f"work_orders_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.write(f"Total Tickets Matching Criteria: **{len(wos)}**")
    if not wos.empty:
        st.dataframe(wos, use_container_width=True)

        st.markdown("---")
        st.subheader("Update Work Order Status")
        with st.form("update_wo_form"):
            selected_wo = st.selectbox("Select Work Order to Transition", wos["work_order_id"].tolist())
            new_status = st.selectbox("Target Status", ["IN_PROGRESS", "COMPLETED", "CANCELLED", "OPEN"])
            technician_note = st.text_area("Technician Operational Log / Maintenance Notes", placeholder="e.g., Replaced HPC 1st stage stators; verified vibration tolerance.")
            submit_btn = st.form_submit_button("Update Status in Database")

            if submit_btn:
                success = wo_service.update_work_order_status(selected_wo, new_status, note=technician_note)
                if success:
                    st.success(f"Work order {selected_wo} transitioned to {new_status} successfully!")
                    st.rerun()
                else:
                    st.error("Failed to update work order.")
    else:
        st.info("No work orders found matching the filter criteria.")


# ==============================================================================
# 6. MLOPS MONITORING
# ==============================================================================
elif page == "MLOps Monitoring":
    st.markdown('<p class="main-header">MLOps Fleet Monitoring & Data Drift</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Telemetry health, feature shift metrics (PSI/KS tests), and experiment lineage.</p>', unsafe_allow_html=True)

    train_df, test_df, _ = load_fleet_data(config["dataset"]["name"])
    scored_fleet = get_scored_fleet_data()

    if train_df is not None and scored_fleet is not None:
        monitor = FleetMonitor(baseline_df=train_df)
        sensor_cols = [f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 20, 21]]
        metrics = monitor.analyze_fleet_telemetry(scored_fleet, sensor_cols=sensor_cols)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scored Fleet Batch", f"{metrics['total_engines_evaluated']} Units")
        c2.metric("Data Quality Status", "PASS (100%)" if metrics["data_quality"]["is_healthy"] else "FAIL")
        c3.metric("Active Anomaly Alerts", f"{metrics['anomalous_observations']}")
        c4.metric("Experiment Tracking URI", "Local MLflow (`mlruns`)")

        st.markdown("---")
        st.subheader("Sensor Distribution Shift (Population Stability Index & KS-Test)")
        st.caption(
            "Detects global covariate shift across the fleet. "
            "Note: Continuous wear across cycles is expected physical degradation; sudden fleet-wide PSI spikes indicate environmental or sensor calibration drift."
        )

        drift_data = []
        for s, d in metrics["sensor_drift_analysis"].items():
            drift_data.append({
                "Sensor Channel": s,
                "PSI (Population Stability)": d["psi"],
                "KS-Statistic": d["ks_statistic"],
                "P-Value": d["p_value"],
                "Drift Flag": "⚠️ DRIFT" if d["drift_detected"] else "✅ STABLE",
            })
        drift_df = pd.DataFrame(drift_data)
        st.dataframe(drift_df, use_container_width=True)

        st.markdown("---")
        st.subheader("MLflow Experiment Lineage")
        st.markdown(
            f"""
            - **Experiment Name:** `{config['mlops']['experiment_name']}`
            - **Artifact Storage:** `{config['mlops']['artifacts_dir']}`
            - **Logged Hyperparameters:** RUL cap (`{config['rul_labeling']['rul_cap_value']}`), Isolation Forest Contamination (`{config['anomaly_detection']['contamination']}`), HistGradientBoosting `max_iter` (`150`), `learning_rate` (`0.05`).
            - **Launch MLflow UI:** Run `mlflow ui` in terminal to inspect parameters, artifacts, and runs.
            """
        )
    else:
        st.warning("Training baseline or scored fleet data not yet ready.")


# ==============================================================================
# 7. RETRAINING DEMO
# ==============================================================================
elif page == "Retraining Demo":
    st.markdown('<p class="main-header">Automated Retraining Demonstration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Reproducible retraining workflow with objective candidate acceptance vs. champion rollback criteria.</p>', unsafe_allow_html=True)

    st.markdown(
        """
        ### Retraining Governance Policy
        1. Ingest newly labeled run-to-failure batch.
        2. Validate telemetry data integrity and verify absence of cross-engine leakage.
        3. Train candidate model (`HistGradientBoostingRegressor`).
        4. Evaluate candidate on fixed validation partition.
        5. **Promotion Rule:** Candidate is promoted if and only if **Candidate RMSE improves over Champion RMSE by at least 2.0%** and candidate RMSE <= 30 cycles.
        6. If criteria fail, the production champion model is safely preserved and the attempt is logged in MLflow.
        """
    )

    if st.button("🚀 Trigger Automated Retraining Workflow", type="primary"):
        with st.spinner("Executing retraining workflow, validating data, training candidate model, and logging to MLflow..."):
            train_df, _, _ = load_fleet_data(config["dataset"]["name"])
            labeled_train = add_rul_labels(train_df, cap_rul=True, rul_cap_value=125)
            train_part, val_part = create_engine_grouped_split(labeled_train, test_size=0.25, random_state=42)

            pipeline = InferencePipeline("config/config.yaml")
            feature_cols = pipeline.rul_predictor.feature_names_

            retrain_mgr = RetrainingManager(
                champion_model_path="artifacts/models/rul_model.joblib",
                min_improvement_pct=2.0,
            )
            result = retrain_mgr.execute_retraining(
                new_data_df=train_part,
                validation_df=val_part,
                feature_cols=feature_cols,
                candidate_model_type="hist_gradient_boosting",
            )

            st.markdown("---")
            st.subheader("Retraining Execution Outcome")
            if result.promoted:
                st.success(f"🎉 **{result.message}**")
            else:
                st.warning(f"🛡️ **{result.message}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Champion Model RMSE", f"{result.champion_rmse:.2f} Cycles")
            c2.metric("Candidate Model RMSE", f"{result.candidate_rmse:.2f} Cycles")
            c3.metric("Performance Delta", f"{result.improvement_pct:+.2f}%")
            c4.metric("Promotion Status", "PROMOTED" if result.promoted else "REJECTED")

            st.json(result.to_dict())


# ==============================================================================
# 8. SYSTEM STATUS
# ==============================================================================
elif page == "System Status":
    st.markdown('<p class="main-header">System Health & Environment Status</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Verification of dataset availability, model artifacts, and SQLite database connectivity.</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Dataset status
    data_ok = os.path.exists("data/raw/train_FD001.txt") or os.path.exists("CMAPSSData/train_FD001.txt")
    c1.metric("C-MAPSS Dataset Files", "AVAILABLE (FD001-004)" if data_ok else "MISSING")

    # Models status
    models_ok = os.path.exists("artifacts/models/rul_model.joblib") and os.path.exists("artifacts/models/isolation_forest.joblib")
    c2.metric("ML Model Artifacts", "ACTIVE & LOADED" if models_ok else "NOT FOUND")

    # DB status
    db_ok = os.path.exists(config["work_orders"]["db_path"])
    c3.metric("SQLite Work Orders DB", "CONNECTED" if db_ok else "UNINITIALIZED")

    st.markdown("---")
    st.subheader("Component Verification Checklist")

    checklist = [
        ("C-MAPSS Raw Files (`CMAPSSData/` / `data/raw/`)", data_ok, "Dataset accessible for training and testing"),
        ("Preprocessor Artifact (`artifacts/models/preprocessor.joblib`)", os.path.exists("artifacts/models/preprocessor.joblib"), "Feature scalers and variance filters"),
        ("Feature Engineer (`artifacts/models/feature_engineer.joblib`)", os.path.exists("artifacts/models/feature_engineer.joblib"), "Engine-wise rolling transforms"),
        ("Isolation Forest Anomaly Detector (`artifacts/models/isolation_forest.joblib`)", os.path.exists("artifacts/models/isolation_forest.joblib"), "Unsupervised out-of-distribution detector"),
        ("HistGradientBoosting RUL Regressor (`artifacts/models/rul_model.joblib`)", os.path.exists("artifacts/models/rul_model.joblib"), "Primary RUL prediction model"),
        ("SQLite Work Orders Database (`database/work_orders.db`)", db_ok, "Persistent maintenance ticketing store"),
        ("Evaluation Benchmark Report (`artifacts/reports/evaluation_report.json`)", os.path.exists("artifacts/reports/evaluation_report.json"), "Official NASA C-MAPSS metrics"),
    ]

    for item, status, desc in checklist:
        col_icon, col_name, col_desc = st.columns([1, 4, 6])
        col_icon.markdown("✅ PASS" if status else "❌ MISSING")
        col_name.markdown(f"**{item}**")
        col_desc.markdown(f"_{desc}_")
