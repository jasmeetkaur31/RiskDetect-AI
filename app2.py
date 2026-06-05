import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="RiskDetect AI",
    page_icon="💳",
    layout="wide"
)

# ==================================================
# TITLE
# ==================================================

st.title("💳RiskDetect AI ")

st.markdown("""
Upload a transaction CSV and let AI identify suspicious transactions.

### Steps
1. Upload CSV
2. Select amount column
3. Choose detection mode
4. Run risk detection
5. Download anomaly report
""")

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# ==================================================
# LOAD DATA
# ==================================================

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# ==================================================
# DATA PREVIEW
# ==================================================

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# ==================================================
# SELECT AMOUNT COLUMN
# ==================================================

numeric_columns = list(
    df.select_dtypes(include=["number"]).columns
)

if len(numeric_columns) == 0:
    st.error("No numeric columns found.")
    st.stop()

amount_col = st.selectbox(
    "Select Transaction Amount Column",
    numeric_columns
)

# ==================================================
# DETECTION MODE
# ==================================================

risk_mode = st.selectbox(
    "Detection Mode",
    [
        "Conservative",
        "Balanced",
        "Aggressive"
    ]
)

if risk_mode == "Conservative":
    contamination = 0.01
elif risk_mode == "Balanced":
    contamination = 0.02
else:
    contamination = 0.05

# ==================================================
# RUN DETECTION
# ==================================================

run_detection = st.button(
    "🚨 Run Risk Detection"
)

if not run_detection:
    st.stop()

# ==================================================
# FEATURE PREPARATION
# ==================================================

features = df[[amount_col]].copy()

features = features.fillna(0)

# ==================================================
# MODEL
# ==================================================

model = IsolationForest(
    contamination=contamination,
    random_state=42
)

model.fit(features)

# ==================================================
# PREDICTION
# ==================================================

predictions = model.predict(features)

df["RiskFlag"] = (
    predictions == -1
).astype(int)

df["RiskScore"] = model.score_samples(features)

# ==================================================
# RESULTS
# ==================================================

anomaly_df = df[
    df["RiskFlag"] == 1
]

total_transactions = len(df)

total_anomalies = len(anomaly_df)

risk_rate = (
    total_anomalies /
    total_transactions
) * 100

total_spend = df[amount_col].sum()

# ==================================================
# OVERALL RISK LEVEL
# ==================================================

if risk_rate < 1:
    risk_level = "🟢 Low Risk"
elif risk_rate < 3:
    risk_level = "🟠 Medium Risk"
else:
    risk_level = "🔴 High Risk"

# ==================================================
# KPI CARDS
# ==================================================

st.header("Executive Summary")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Total Spend",
    f"${total_spend:,.0f}"
)

c2.metric(
    "Transactions",
    f"{total_transactions:,}"
)

c3.metric(
    "Suspicious",
    f"{total_anomalies:,}"
)

c4.metric(
    "Risk Rate",
    f"{risk_rate:.2f}%"
)

c5.metric(
    "Risk Level",
    risk_level
)

# ==================================================
# RISK STATUS
# ==================================================

if risk_rate < 1:
    st.success("Overall Dataset Risk: LOW")
elif risk_rate < 3:
    st.warning("Overall Dataset Risk: MEDIUM")
else:
    st.error("Overall Dataset Risk: HIGH")

# ==================================================
# VISUALIZATIONS
# ==================================================

st.header("Risk Analysis")

left, right = st.columns(2)

with left:

    fig1 = px.histogram(
        df,
        x=amount_col,
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with right:

    if len(anomaly_df) > 0:

        fig2 = px.scatter(
            anomaly_df,
            x="RiskScore",
            y=amount_col,
            title="Suspicious Transactions",
            color="RiskFlag"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ==================================================
# TOP SUSPICIOUS TRANSACTIONS
# ==================================================

st.header("Top Suspicious Transactions")

top_risky = (
    anomaly_df
    .sort_values(
        amount_col,
        ascending=False
    )
    .head(20)
)

st.dataframe(
    top_risky,
    use_container_width=True
)

# ==================================================
# DOWNLOAD REPORT
# ==================================================

csv = anomaly_df.to_csv(index=False)

st.download_button(
    label="📥 Download Risk Report",
    data=csv,
    file_name="risk_report.csv",
    mime="text/csv"
)

# ==================================================
# SUMMARY
# ==================================================

st.header("Analysis Summary")

st.info(
    f"""
Transactions Analysed: {total_transactions:,}

Suspicious Transactions: {total_anomalies:,}

Risk Rate: {risk_rate:.2f}%

Risk Level: {risk_level}

Detection Mode: {risk_mode}
"""
)