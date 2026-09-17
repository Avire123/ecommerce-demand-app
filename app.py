import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="E-Commerce Demand Forecaster",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. LOAD ASSETS (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    """Load the trained XGBoost pipeline from disk."""
    return joblib.load("models/demand_xgboost_model.pkl")

@st.cache_data
def load_data():
    """Load the preprocessed dataset for historical exploration."""
    df = pd.read_csv("data/cleaned_ecommerce_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

try:
    model = load_model()
    df_data = load_data()
except Exception as e:
    st.error(f"Error loading model or data artifacts: {e}")
    st.info("Make sure 'train_model.py' has run and generated files in 'models/' and 'data/'.")
    st.stop()

# ==========================================
# 3. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.header("🎯 Input Product Parameters")

categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports"]
category = st.sidebar.selectbox("Product Category", categories)

base_price = st.sidebar.number_input(
    "Base Price ($)",
    min_value=1.0,
    max_value=3000.0,
    value=150.0,
    step=5.0,
)

discount = st.sidebar.slider(
    "Discount Percentage (%)",
    min_value=0,
    max_value=70,
    value=10,
    step=5,
)

competitor_price = st.sidebar.number_input(
    "Competitor Price ($)",
    min_value=1.0,
    max_value=3500.0,
    value=145.0,
    step=5.0,
)

rating = st.sidebar.slider(
    "Customer Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.2,
    step=0.1,
)

forecast_date = st.sidebar.date_input(
    "Forecast Date",
    value=datetime.date.today(),
)

# ==========================================
# 4. FEATURE ENGINEERING FOR INFERENCE
# ==========================================
effective_price = base_price * (1 - (discount / 100))
day_of_week = forecast_date.weekday()
month = forecast_date.month
is_weekend = 1 if day_of_week >= 5 else 0
price_diff = competitor_price - effective_price
price_ratio = effective_price / competitor_price if competitor_price > 0 else 1.0

# Prepare single-row DataFrame matching trained pipeline columns
input_df = pd.DataFrame([{
    "Category": category,
    "BasePrice": base_price,
    "EffectivePrice": effective_price,
    "Discount": discount,
    "Rating": rating,
    "CompetitorPrice": competitor_price,
    "DayOfWeek": day_of_week,
    "Month": month,
    "IsWeekend": is_weekend,
    "PriceDiffvsCompetitor": price_diff,
    "PriceRatio": price_ratio,
}])

# ==========================================
# 5. PREDICTION
# ==========================================
predicted_demand = float(model.predict(input_df)[0])
predicted_demand = max(0, round(predicted_demand))
predicted_revenue = predicted_demand * effective_price

# ==========================================
# 6. MAIN DASHBOARD DISPLAY
# ==========================================
st.title("📦 E-Commerce Demand Prediction & Analytics")
st.markdown("Predict expected sales volume and revenue using our trained **XGBoost** model.")

# Key Performance Indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Predicted Sales Volume", value=f"{predicted_demand:,} units")
with col2:
    st.metric(label="Estimated Revenue", value=f"${predicted_revenue:,.2f}")
with col3:
    st.metric(label="Effective Selling Price", value=f"${effective_price:.2f}", delta=f"-{discount}%")
with col4:
    price_advantage = competitor_price - effective_price
    st.metric(
        label="Price vs Competitor",
        value=f"${abs(price_advantage):.2f}",
        delta="Cheaper" if price_advantage >= 0 else "More Expensive",
        delta_color="normal" if price_advantage >= 0 else "inverse",
    )

st.divider()

# Tabs for in-depth exploration
tab1, tab2, tab3 = st.tabs(["📊 Price Sensitivity Simulator", "📈 Historical Trends", "📋 Raw Data"])

with tab1:
    st.subheader("Price Sensitivity & Demand Curve")
    st.caption("Explore how changing discounts impacts unit sales and total revenue.")

    # Simulate demand across discounts 0% to 60%
    discounts_sim = np.arange(0, 65, 5)
    sim_rows = []
    for d in discounts_sim:
        eff_p = base_price * (1 - (d / 100))
        sim_rows.append({
            "Category": category,
            "BasePrice": base_price,
            "EffectivePrice": eff_p,
            "Discount": d,
            "Rating": rating,
            "CompetitorPrice": competitor_price,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": is_weekend,
            "PriceDiffvsCompetitor": competitor_price - eff_p,
            "PriceRatio": eff_p / competitor_price if competitor_price > 0 else 1.0,
        })

    df_sim = pd.DataFrame(sim_rows)
    df_sim["SimulatedDemand"] = np.maximum(0, np.round(model.predict(df_sim)))
    df_sim["SimulatedRevenue"] = df_sim["SimulatedDemand"] * df_sim["EffectivePrice"]

    fig_sim = px.line(
        df_sim,
        x="Discount",
        y="SimulatedDemand",
        markers=True,
        title=f"Estimated Demand across Discount Rates for {category}",
        labels={"Discount": "Discount (%)", "SimulatedDemand": "Predicted Units"},
    )
    fig_sim.add_scatter(
        x=[discount],
        y=[predicted_demand],
        mode="markers+text",
        name="Current Selection",
        text=["Current"],
        textposition="top center",
        marker=dict(color="red", size=12),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

with tab2:
    st.subheader("Category Distribution & Demand Insights")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_cat = px.box(
            df_data,
            x="Category",
            y="SalesVolume",
            color="Category",
            title="Sales Volume Distribution by Category",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        fig_scatter = px.scatter(
            df_data,
            x="EffectivePrice",
            y="SalesVolume",
            color="Category",
            opacity=0.6,
            title="Effective Price vs Sales Volume",
            labels={"EffectivePrice": "Effective Price ($)", "SalesVolume": "Units Sold"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.subheader("Cleaned Training Dataset")
    st.dataframe(df_data.head(100), use_container_width=True)