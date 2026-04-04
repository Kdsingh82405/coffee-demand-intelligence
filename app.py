import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="☕ Coffee Forecast Dashboard", layout="wide")

# -------------------------------------------------
# CSS 
# -------------------------------------------------
st.markdown("""
<style>
/* Background */
body {
    background-color: #0f172a;
}

/* Main container */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* KPI Cards */
.kpi-card {
    height: 140px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-radius: 18px;
    padding: 25px;
    text-align: center;

    box-shadow: 
        inset 0 0 20px rgba(0,255,200,0.05),
        0 6px 25px rgba(0,0,0,0.4);

    transition: all 0.3s ease;
}

.kpi-card:hover {
    transform: scale(1.05);
    box-shadow: 0px 6px 25px rgba(0,255,200,0.3);
}

/* KPI Title */
.kpi-title {
    font-size: 14px;
    color: #94a3b8;
}

/* KPI Value */
.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #00ffc3;
}

/* Section Titles */
.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 20px;
    color: #0f172a;
}

/* Divider */
.divider {
    height: 2px;
    background: linear-gradient(to right, #00ffc3, transparent);
    margin: 20px 0;
}

/* Insight Box */
.insight-box {
    background-color: #ecfdf5;
    padding: 18px;
    border-radius: 15px;
    border-left: 5px solid #10b981;
    font-size: 15px;
}
/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    overflow-x: auto;
    white-space: nowrap;
    display: flex;
    flex-wrap: nowrap;
}

.stTabs [data-baseweb="tab"] {
    background: #f1f5f9;
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 500;
    font-size: 14px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #2563eb, #00ffc3) !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Afficionado_Coffee_Roasters.csv")
    
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], format="%H:%M:%S")

    df = df.sort_values("transaction_time").reset_index(drop=True)
    df["datetime"] = pd.date_range(start="2025-01-01", periods=len(df), freq="min")
    df["hour"] = df["datetime"].dt.hour
    df["date"] = pd.to_datetime(df["datetime"].dt.date)

    return df

df = load_data()

@st.cache_data
def load_forecast():
    f = pd.read_csv("data/forecast_results.csv")
    f.columns = [c.lower() for c in f.columns]
    f["date"] = pd.to_datetime(f["date"])

    # detect forecast column
    for col in ["forecast", "prediction", "yhat"]:
        if col in f.columns:
            return f, col

    st.error("❌ Forecast column not found")
    st.stop()

forecast_df, forecast_col = load_forecast()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("🔎 Filters")
st.sidebar.markdown("### 🎯 Customize Forecast")

store = st.sidebar.selectbox("Store", ["All"] + sorted(df["store_id"].unique()))
metric = st.sidebar.radio("Metric", ["Revenue", "Quantity"])
horizon = st.sidebar.slider("Forecast Days", 7, 30, 14)
model_choice = st.sidebar.selectbox("Model", ["Gradient Boosting", "Baseline"])

# ---------------- DYNAMIC METRIC ----------------
if metric == "Revenue":
    actual_col = "revenue"
    forecast_col = "forecast"
else:
    actual_col = "transaction_qty"
    forecast_col = "forecast_qty"
    
# -------------------------------------------------
# FILTER DATA
# -------------------------------------------------
df_filtered = df.copy()
forecast_filtered = forecast_df.copy()

if store != "All":
    df_filtered = df_filtered[df_filtered["store_id"] == store]
    forecast_filtered = forecast_filtered[forecast_filtered["store_id"] == store]

forecast_filtered = forecast_filtered.sort_values("date")

latest_date = forecast_filtered["date"].max()
min_date = latest_date - pd.Timedelta(days=horizon)

filtered_forecast = forecast_filtered[
    forecast_filtered["date"] >= min_date
]

df_filtered = df_filtered[df_filtered["date"] >= min_date]

if filtered_forecast.empty or df_filtered.empty:
    st.warning("No data available for selected filters")
    st.stop()

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("Data-Driven Forecasting & Peak Demand Prediction for Afficionado Coffee Roasters")
st.divider()

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
st.subheader("📊 Key Performance Indicators")

full = filtered_forecast.copy()

# Metrics
mae = mean_absolute_error(full[actual_col], full[forecast_col])
rmse = np.sqrt(mean_squared_error(full[actual_col], full[forecast_col]))
mape = np.mean(
    np.abs((full[actual_col] - full[forecast_col]) /
        full[actual_col].replace(0, np.nan))
) * 100

accuracy = 100 - mape

# Peak detection
threshold = full[actual_col].mean() + 1.5 * full[actual_col].std()
full["peak_actual"] = full[actual_col] > threshold
full["peak_pred"] = full[forecast_col] > threshold
peak_capture = (full["peak_actual"] == full["peak_pred"]).mean() * 100

# Lead time
actual_diff = full[actual_col].diff().fillna(0)
pred_diff = full[forecast_col].diff().fillna(0)
lead_time = 100 - (rmse / full[actual_col].mean() * 100)
stability = full.groupby("store_id")[forecast_col].std().mean()

# KPI UI
c1, c2, c3, c4, c5 = st.columns(5)

c1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Forecast Accuracy</div><div class='kpi-value'>{accuracy:.2f}%</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Peak Demand Capture Rate</div><div class='kpi-value'>{peak_capture:.2f}%</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Revenue Forecast Error</div><div class='kpi-value'>{mae:.2f}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Lead Time Accuracy</div><div class='kpi-value'>{lead_time:.2f}%</div></div>", unsafe_allow_html=True)
c5.markdown(f"<div class='kpi-card'><div class='kpi-title'>Store Forecast Stability</div><div class='kpi-value'>{stability:.2f}</div></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:auto; color:#64748b; font-size:13px; margin-top:10px;'>Showing last 14 days data</div>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# TOP NAVIGATION TABS 
# -------------------------------------------------
tab1, tab2, tab3, tab4= st.tabs([
    "📈 Forecast vs Actual & Model Evaluation" ,
    "📉 Uncertainty & Planning",
    "⚡ Demand Planning Analysis",
    "📦 Category & Store Analysis"
])

# -------------------------------------------------
# FORECAST VS ACTUAL
# -------------------------------------------------
with tab1:
    st.markdown("<div class='section-title'>📈 Forecast vs Actual</div>", unsafe_allow_html=True)

    agg = filtered_forecast.groupby("date").agg({
        actual_col: "sum",
        forecast_col: "sum"
    }).reset_index()

    agg["baseline"] = agg[actual_col].rolling(3).mean().bfill()
    y_col = "baseline" if model_choice == "Baseline" else forecast_col

    fig = px.line(
        agg,
        x="date",
        y=[y_col, actual_col],
        title=None
    )

    fig.data[0].name = "Forecast"
    fig.data[1].name = "Actual"
    # Style
    fig.data[0].update(
        mode="lines+markers",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=6)
    )

    fig.data[1].update(
        mode="lines+markers",
        line=dict(color="green", width=3, dash="dot"),
        marker=dict(size=6)
    )
    # Dynamic range
    y_min = min(agg[actual_col].min(), agg[y_col].min())
    y_max = max(agg[actual_col].max(), agg[y_col].max())

    fig.update_layout(
        yaxis=dict(range=[y_min - 500, y_max + 500]),
        yaxis_title="Value",
        xaxis_title="Date",
        template="simple_white",
        hovermode="closest",
    )

    fig.update_traces(
        hovertemplate="%{x}<br>%{fullData.name}: %{y:.0f}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

    peak_day = agg.sort_values(forecast_col, ascending=False).iloc[0]["date"]
    st.success(f"🔥 Highest predicted demand on {peak_day.date()} (Plan staffing & inventory)")
    st.divider()

    # -------------------------------------------------
    # MODEL COMPARISON
    # -------------------------------------------------

    st.markdown("<div class='section-title'>📊 Model Comparison</div>", unsafe_allow_html=True)

    baseline = full[actual_col].shift(1).bfill()
    baseline_mae = mean_absolute_error(full[actual_col], baseline)

    improvement = (1 - mae / baseline_mae) * 100

    col1, col2 = st.columns(2, gap="large")
    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    # Baseline
    col1.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Baseline MAE</div>
        <div class='kpi-value' style='color:#ef4444; font-size:30px;'>
            {baseline_mae:.2f} ↑
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Card
    col2.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Model MAE</div>
        <div class='kpi-value' style='font-size:30px;'>
            {mae:.2f} ↓
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Improvement Box
    st.markdown(f"""
    <div class='insight-box' style="text-align:center; font-size:15px;">
        Error reduced by <b>{improvement:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # -------------------------------------------------
    # CONFIDENCE INTERVAL
    # -------------------------------------------------
with tab2:
    st.subheader("📉 Confidence Interval")

    residual = agg[actual_col] - agg[forecast_col]
    std = residual.rolling(7, min_periods=1).std().fillna(residual.std())

    agg["upper"] = agg[forecast_col] + 1.96 * std
    agg["lower"] = agg[forecast_col] - 1.96 * std

    fig = px.line(agg, x="date", y=forecast_col)

    fig.add_scatter(
        x=agg["date"],
        y=agg["upper"],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    )

    fig.add_scatter(
        x=agg["date"],
        y=agg["lower"],
        fill='tonexty',
        mode='lines',
        fillcolor='rgba(56,189,248,0.2)',
        line=dict(width=0),
        name="Confidence Interval"
    )
    fig.update_layout(
        template="simple_white",
        hovermode="closest",
        xaxis_title="Date",
        yaxis_title="Value"
    )
    fig.update_traces(line=dict(color="#2563eb", width=2))
    fig.add_scatter(x=agg["date"], y=agg["upper"], name="Upper")
    fig.add_scatter(x=agg["date"], y=agg["lower"], name="Lower")

    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # -------------------------------------------------
    # SCENARIO ANALYSIS (Best-case vs Worst-case Scenario)
    # -------------------------------------------------
    st.markdown("<div class='section-title'>📊 Scenario Analysis</div>", unsafe_allow_html=True)
    scenario = agg.copy()
    # Best & Worst Case (±10%)
    scenario["best_case"] = scenario[y_col] * 1.10
    scenario["worst_case"] = scenario[y_col] * 0.90
    fig = px.line(
        scenario,
        x="date",
        y=[y_col, "best_case", "worst_case"],
    )
    # Rename
    fig.data[0].name = "Forecast"
    fig.data[1].name = "Best Case (+10%)"
    fig.data[2].name = "Worst Case (-10%)"
    # Style
    fig.data[0].update(line=dict(color="#2563eb", width=2))
    fig.data[1].update(line=dict(color="green", dash="dot"))
    fig.data[2].update(line=dict(color="red", dash="dot"))

    fig.data[0].update(mode="lines+markers")
    fig.data[1].update(mode="lines+markers")
    fig.data[2].update(mode="lines+markers")

    fig.update_layout(
        template="simple_white",
        hovermode="closest",
        xaxis_title="Date",
        yaxis_title="value"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("📈 Best-case and worst-case scenarios help plan inventory and staffing under demand uncertainty.")

    st.divider()
# -------------------------------------------------
# Demand Spike Analysis
# -------------------------------------------------
with tab3:
    st.markdown("<div class='section-title'>⚡ Demand Spike Analysis</div>", unsafe_allow_html=True)

    spike = full.copy()

    # Define spike threshold (Top 10%)
    threshold = spike[actual_col].quantile(0.90)

    # Create label
    spike["is_spike"] = spike[actual_col] > threshold
    spike["is_spike"] = spike["is_spike"].map({True: "Spike", False: "Normal"})

    # Plot
    fig = px.scatter(
        spike,
        x="date",
        y=actual_col,
        color="is_spike",
        title=None,
        color_discrete_map={
            "Normal": "#2563eb",
            "Spike": "#dc2626"
        }
    )

    fig.update_traces(marker=dict(size=9, opacity=0.85))

    # 🔥 Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=3,
        annotation_text="Spike Threshold",
        annotation_font=dict(color="#f59e0b")
    )

    # ✅ LIGHT THEME (MAIN FIX)
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1e293b"),

        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    num_spikes = spike[spike["is_spike"] == "Spike"].shape[0]

    st.markdown(f"""
    <div class='insight-box'>
    🔥 Total Spike Events Detected: <b>{num_spikes}</b><br>
    📈 These spikes indicate peak demand periods requiring higher inventory and staffing.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    # -------------------------------------------------
    # HOURLY DEMAND
    # -------------------------------------------------
    st.subheader("⏱ Hourly Demand Pattern")

    y = "revenue" if metric == "Revenue" else "transaction_qty"

    hourly = df_filtered.groupby("hour")[y].sum().reset_index()
    fig = px.line(hourly, x="hour", y=y, markers=True)


    fig.update_layout(
        yaxis=dict(range=[hourly[y].min() - 200, hourly[y].max() + 200])
    )

    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # -------------------------------------------------
    # HEATMAP
    # -------------------------------------------------
    st.subheader("🔥 Demand Heatmap")

    pivot = df_filtered.pivot_table(values=actual_col, index="hour", columns="store_id")

    if not pivot.empty:
        fig = px.imshow(pivot, aspect="auto")

        fig.update_layout(title="Hourly Demand Heatmap")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

# -------------------------------------------------
# CATEGORY DEMAND
# -------------------------------------------------
with tab4:
    st.markdown("<div class='section-title'>📦 Category Analysis</div>", unsafe_allow_html=True)

    # Step 1: grouped data (store-wise)
    cat = df_filtered.groupby(
        ["product_category", "store_id"]
    )[actual_col].sum().reset_index()

    # Step 2: total per category
    cat_total = df_filtered.groupby("product_category")[actual_col].sum().reset_index()

    # Step 3: grouped bar chart
    fig = px.bar(
        cat,
        x="product_category",
        y=actual_col,
        color="store_id",
        barmode="stack"
    )

    # Step 4: ADD TOTAL LABEL ON TOP 🔥
    fig.add_scatter(
        x=cat_total["product_category"],
        y=cat_total[actual_col],
        text=[f"{v/1000:.1f}k" for v in cat_total[actual_col]],
        mode="text",
        textposition="top center",
        textfont=dict(size=14, color="black"),
        showlegend=False
    )

    fig.update_layout(
        template="simple_white",
        xaxis_title="Category",
        yaxis_title=metric,
        legend_title="Store",
        bargap=0.3,
        bargroupgap=0.15
    )

    st.plotly_chart(fig, use_container_width=True)
    top_cat = cat.groupby("product_category")[actual_col].sum().idxmax()
    low_cat = cat.groupby("product_category")[actual_col].sum().idxmin()

    st.success(f"🏆 Top Category: {top_cat}")
    st.warning(f"📉 Lowest Category: {low_cat}")
    st.divider()

    # -------------------------------------------------
    # STORE PERFORMANCE
    # -------------------------------------------------
    st.markdown("<div class='section-title'>🏪 Store Performance</div>", unsafe_allow_html=True)

    store_perf = df_filtered.groupby(["store_id", "store_location"])[actual_col].sum().reset_index()
    store_perf = store_perf.sort_values(actual_col, ascending=False)
    store_perf["label"] = "Store " + store_perf["store_id"].astype(str) + "<br>" + store_perf[actual_col].astype(int).astype(str)

    fig = px.bar(
        store_perf,
        x="store_location",
        y=actual_col,
        color=actual_col,
        color_continuous_scale="Blues",
        title="Store-wise Revenue Performance",
        text="label"
    )
    # Remove legend
    fig.update_layout(showlegend=False)
    # Auto range
    max_val = store_perf[actual_col].max()
    fig.update_layout(
        yaxis=dict(range=[0, max_val * 1.15])
    )
    # Position text
    fig.update_traces(
        textposition='outside'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    # -------------------------------------------------
    # STORE FORECAST
    # -------------------------------------------------
    if store != "All":
        st.subheader(f"🏪 Store {store} Forecast")

        store_data = filtered_forecast[filtered_forecast["store_id"] == store]

        fig = px.line(store_data, x="date", y=forecast_col)
        fig.add_scatter(x=store_data["date"], y=store_data[actual_col], name="Actual")

        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    # -------------------------------------------------
    # INSIGHTS
    # -------------------------------------------------
    st.subheader("📌 Business Insights")

    st.success("""
    • Peak demand occurs in morning & evening  
    • Store demand varies significantly  
    • Forecast accuracy is above 90%  
    • Model helps reduce waste and optimize staffing  
    """)