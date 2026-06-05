import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Credit Card Risk Dashboard",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("powerbi.csv")

df = load_data()

# ==================================================
# SIDEBAR FILTERS
# ==================================================

st.sidebar.title("Dashboard Filters")

selected_years = st.sidebar.multiselect(
    "Select Year",
    sorted(df["Year"].unique()),
    default=sorted(df["Year"].unique())
)

selected_departments = st.sidebar.multiselect(
    "Select Department",
    sorted(df["Department"].unique()),
    default=sorted(df["Department"].unique())
)

filtered_df = df[
    (df["Year"].isin(selected_years))
    &
    (df["Department"].isin(selected_departments))
]

# ==================================================
# TITLE
# ==================================================

st.title("💳 Credit Card Transaction Risk Analytics Dashboard")
st.markdown("### Executive Summary")

# ==================================================
# KPI CARDS
# ==================================================

total_spend = filtered_df["TrnxAmount"].sum()
total_transactions = len(filtered_df)
total_anomalies = filtered_df["anomaly_v2"].sum()

anomaly_rate = (
    total_anomalies /
    total_transactions
) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Spend",
    f"${total_spend/1_000_000:.1f}M"
)

col2.metric(
    "Transactions",
    f"{total_transactions:,}"
)

col3.metric(
    "Anomalies",
    f"{total_anomalies:,}"
)

col4.metric(
    "Anomaly Rate",
    f"{anomaly_rate:.2f}%"
)

# ==================================================
# SPENDING ANALYSIS
# ==================================================

st.header("Spending Analysis")

left, right = st.columns(2)

with left:

    yearly_spend = (
        filtered_df
        .groupby("Year")["TrnxAmount"]
        .sum()
        .reset_index()
    )

    fig1 = px.line(
        yearly_spend,
        x="Year",
        y="TrnxAmount",
        markers=True,
        title="Year-wise Spending Trend"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with right:

    dept_spend = (
        filtered_df
        .groupby("Department")["TrnxAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        dept_spend,
        x="Department",
        y="TrnxAmount",
        title="Top 10 Departments by Spending"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==================================================
# DEPARTMENT RISK ANALYSIS
# ==================================================

st.header("Department Risk Analysis")

dept_risk = (
    filtered_df
    .groupby("Department")
    .agg(
        Total_Transactions=("anomaly_v2", "count"),
        Anomalies=("anomaly_v2", "sum")
    )
)

dept_risk["Anomaly_Rate"] = (
    dept_risk["Anomalies"]
    /
    dept_risk["Total_Transactions"]
) * 100

dept_risk = (
    dept_risk
    .sort_values(
        "Anomaly_Rate",
        ascending=False
    )
    .head(15)
    .reset_index()
)

fig3 = px.bar(
    dept_risk,
    x="Department",
    y="Anomaly_Rate",
    title="Top Departments by Anomaly Rate"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ==================================================
# MERCHANT RISK ANALYSIS
# ==================================================

st.header("Merchant Risk Analysis")

merchant_risk = (
    filtered_df[
        filtered_df["anomaly_v2"] == 1
    ]
    .groupby("Merchant")
    .size()
    .sort_values(ascending=False)
    .head(15)
    .reset_index(name="Anomalies")
)

fig4 = px.bar(
    merchant_risk,
    x="Merchant",
    y="Anomalies",
    title="Top Merchants by Number of Anomalies"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ==================================================
# TREEMAP
# ==================================================

st.header("Department Spending Distribution")

fig5 = px.treemap(
    filtered_df,
    path=["Department"],
    values="TrnxAmount",
    title="Department Spending Treemap"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# ==================================================
# ANOMALY VISUALIZATION
# ==================================================

st.header("Anomaly Visualization")

anomaly_df = filtered_df[
    filtered_df["anomaly_v2"] == 1
]

if len(anomaly_df) > 0:

    fig6 = px.scatter(
        anomaly_df,
        x="anomaly_score",
        y="TrnxAmount",
        color="Department",
        hover_data=[
            "Merchant",
            "Year"
        ],
        title="Anomalous Transactions by Risk Score"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

# ==================================================
# ANOMALY TABLE
# ==================================================

st.header("Anomaly Investigation")

columns_to_show = [
    "Department",
    "Merchant",
    "TrnxAmount",
    "Year"
]

if "anomaly_score" in filtered_df.columns:
    columns_to_show.append("anomaly_score")

anomaly_table = filtered_df[
    filtered_df["anomaly_v2"] == 1
][columns_to_show]

st.dataframe(
    anomaly_table,
    use_container_width=True
)

# ==================================================
# DOWNLOAD REPORT
# ==================================================

csv = anomaly_table.to_csv(index=False)

st.download_button(
    label="📥 Download Anomaly Report",
    data=csv,
    file_name="anomaly_report.csv",
    mime="text/csv"
)

# ==================================================
# KEY INSIGHTS
# ==================================================

st.header("Key Insights")

st.markdown(f"""
### Summary

- Total Transactions Analyzed: **{total_transactions:,}**
- Total Anomalies Detected: **{total_anomalies:,}**
- Overall Anomaly Rate: **{anomaly_rate:.2f}%**
- Total Spend: **${total_spend/1_000_000:.1f}M**

### Business Insights

- Department of Corrections contributes the largest spending volume.
- Transportation is the second-largest spending department.
- Only a small percentage of transactions are flagged as anomalous.
- Certain merchants repeatedly appear in anomaly investigations.
- Isolation Forest successfully identifies unusual spending behavior for further review.
""")