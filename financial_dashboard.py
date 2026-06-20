import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


st.set_page_config(layout="wide")

# _________________________________________________
# NUMBER FORMAT
# _________________________________________________
def format_number(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except:
        return "0"

# _________________________________________________
# LOAD DATA
# _________________________________________________

@st.cache_data
def load_data():
    df = pd.read_csv("financial_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["revenue","budget","expense"]:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["gap"] = df["revenue"] - df["budget"]
    return df

df = load_data()

# ________________________________________________
# SIDEBAR
# ________________________________________________
st.sidebar.markdown("## 📊 Finance Dashboard")
st.sidebar.markdown("Budget • Revenue • Forecast")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigation",
    ["Overview","2022","2023","2024","2025 Forecast"]
)

# _________________________________________________
# OVERVIEW
# _________________________________________________
if section == "Overview":
    st.title("Yearly Overview 📊 (2022-2024)")
    st.markdown("General performance summary across all years.")
    yearly = df.groupby("year").agg({
        "budget":"sum",
        "revenue":"sum",
        "gap":"sum"
    }).reset_index()
    for col in ["budget","revenue","gap"]:
        yearly[col] = yearly[col].apply(format_number)
    st.dataframe(yearly,use_container_width=True)

    
# ________________________________________________
# YEAR DASHBOARD 2022-2024
# ________________________________________________
elif section in ["2022","2023","2024"]:
    year = int(section)
    df_year = df[df["year"] == year]

    st.title(f"{year} DETAILED ANALYSIS")

    # KPI
    total_budget = df_year["budget"].sum()
    total_actual = df_year["revenue"].sum()
    total_gap = total_actual - total_budget

    col1,col2,col3 = st.columns(3)
    col1.metric("💰 Total Revenue",format_number(total_actual))
    col2.metric("📊 Total Budget",format_number(total_budget))
    col3.metric("📉 Budget Gap",format_number(total_gap))

    st.markdown("---")

    # TOP 3 VARIANCE MONTHS
    st.subheader("TOP 3 HIGHEST VARIANCE MONTHS")
    top3 = df_year.nlargest(3,"gap")[["month","budget","revenue","gap"]].sort_values("month")
    top3.rename(columns={"month":"MONTH","budget":"BUDGET","revenue":"ACTUAL","gap":"GAP"},inplace=True)
    for col in ["BUDGET","ACTUAL","GAP"]:
        top3[col] = top3[col].apply(format_number)
    st.dataframe(top3.style.highlight_max(subset=["GAP"],color="lightgreen"),use_container_width=True)

    st.markdown("---")

    # MONTHLY BREAKDOWN
    st.subheader("MONTHLY BREAKDOWN")
    monthly = df_year[["month","budget","revenue","gap"]]
    monthly.rename(columns={"month":"MONTH","budget":"BUDGET","revenue":"ACTUAL","gap":"GAP"},inplace=True)
    for col in ["BUDGET","ACTUAL","GAP"]:
        monthly[col] = monthly[col].apply(format_number)
    st.dataframe(monthly.style.highlight_max(subset=["GAP"],color="lightgreen"),use_container_width=True)

    st.markdown("---")

    # QUARTERLY BREAKDOWN
    st.subheader("QUARTERLY BREAKDOWN")
    quarterly = df_year.groupby("quarter").agg({"budget":"sum","revenue":"sum","gap":"sum"}).reset_index()
    quarterly.rename(columns={"quarter":"QUARTER","budget":"BUDGET","revenue":"ACTUAL","gap":"GAP"},inplace=True)
    for col in ["BUDGET","ACTUAL","GAP"]:
        quarterly[col] = quarterly[col].apply(format_number)
    st.dataframe(quarterly.style.highlight_max(subset=["GAP"],color="lightgreen"),use_container_width=True)

    st.markdown("---")

    # MONTHLY TREND
    st.subheader("MONTHLY TREND (BUDGET VS ACTUAL)")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_year["month"],df_year["budget"],marker="o",label="BUDGET")
    ax.plot(df_year["month"],df_year["revenue"],marker="o",label="ACTUAL")
    ax.set_xticks(range(1,13))
    ax.set_xlabel("MONTH")
    ax.set_ylabel("AMOUNT")
    ax.set_title(f"{year} BUDGET VS ACTUAL")
    ax.legend()
    st.pyplot(fig)
    
# ________________________________________________
# 2025 FORECAST
# ________________________________________________
elif section == "2025 Forecast":
    st.title("🔮 2025 Budget & Forecast Analysis")

    # ____________________________
    # Enflasyon oranı   
    # ____________________________
    inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0, 100, 40)
    inflation = inflation_rate / 100

    # ____________________________
    # 2024 actual verileri
    # ____________________________
    df_2024 = df[df["year"] == 2024].groupby("month")["revenue"].sum().reset_index()

    # ____________________________
    # 2025 budget = 2024 actual * (1 + inflation)
    # ____________________________
    df_2025 = df_2024.copy()
    df_2025["budget_2025"] = df_2025["revenue"] * (1 + inflation)

    # ____________________________
    # LINEAR REGRESSION ile Ocak–Mart actual tahmini (trend)
    # 
    X = df_2025["month"].values[:3].reshape(-1,1)  # Ocak-Şubat-Mart
    y = df_2025["budget_2025"].values[:3]

    model = LinearRegression()
    model.fit(X, y)


    df_2025["actual_2025"] = model.predict(df_2025["month"].values.reshape(-1,1))

    # ___________________________
    # GAP
    # ___________________________
    df_2025["gap"] = df_2025["budget_2025"] - df_2025["actual_2025"]

    # ___________________________
    # Nisan forecast (Moving Average)
    # ___________________________                       
    jan_to_mar_actual = df_2025.loc[df_2025["month"].isin([1,2,3]),"actual_2025"]
    forecast_april = jan_to_mar_actual.mean()
    budget_april = df_2025.loc[df_2025["month"]==4,"budget_2025"].values[0]
    gap_april = budget_april - forecast_april

    st.subheader("Next Month (April) Forecast & Gap")
    st.metric("Forecast Actual (April)", format_number(forecast_april))
    st.metric("Budget (April)", format_number(budget_april))
    st.metric("Forecast Gap (April)", format_number(gap_april))

    st.markdown("---")

    # ______________________
    # 2025 Monthly Table
    # ______________________
    df_display = df_2025.copy()
    df_display.rename(columns={
        "month":"MONTH",
        "revenue":"2024 ACTUAL",
        "budget_2025":"2025 BUDGET",
        "actual_2025":"2025 ACTUAL",
        "gap":"GAP"
    }, inplace=True)

    for col in ["2024 ACTUAL","2025 BUDGET","2025 ACTUAL","GAP"]:
        df_display[col] = df_display[col].apply(format_number)

    st.subheader("2025 Monthly Budget vs Actual (LR + Moving Average Forecast)")
    st.dataframe(df_display,use_container_width=True)

    # ______________________
    # 2025 Monthly Trend Graph
    # ______________________
    st.subheader("2025 Monthly Trend (Budget vs Actual)")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_display["MONTH"], df_2025["budget_2025"], marker="o", label="BUDGET")
    ax.plot(df_display["MONTH"], df_2025["actual_2025"], marker="o", label="ACTUAL (Forecast)")
    ax.set_xticks(range(1,13))
    ax.set_xlabel("MONTH")
    ax.set_ylabel("AMOUNT")
    ax.set_title("2025 Budget vs Forecast Actual")
    ax.legend()
    st.pyplot(fig)

    for col in ["2024 ACTUAL","2025 BUDGET","2025 ACTUAL","GAP"]:
        df_display[col] = df_display[col].apply(format_number)

    st.subheader("2025 Monthly Budget vs Actual")
    st.dataframe(df_display, use_container_width=True)
