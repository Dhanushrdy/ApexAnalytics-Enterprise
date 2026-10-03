"""
================================================================================
ApexAnalytics Enterprise | Predictive Sales Intelligence Platform
================================================================================
A production-ready, theme-adaptive SaaS analytics and forecasting application.
Engineered with Streamlit, DuckDB, Facebook Prophet, Plotly, and Pandas.

Features:
- Universal Theme Adaptation: Flawlessly renders in both Light & Dark modes.
- In-process SQL Acceleration via DuckDB directly on flat CSV files.
- Automated Bayesian Time-Series Forecasting via Facebook Prophet.
- Interactive multi-tab executive experience with Scenario Simulation.
- Production-grade error handling, schema validation, and graceful degradation.

Author: Senior Python Developer & Data Scientist
License: Commercial SaaS Template
================================================================================
"""

import os
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ApexAnalytics Enterprise | Predictive Sales Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 2. UNIVERSAL THEME-ADAPTIVE DESIGN SYSTEM (LIGHT & DARK COMPATIBLE)
# ==============================================================================
def inject_adaptive_css() -> None:
    """
    Injects a universal, theme-adaptive design system into Streamlit.
    
    Architectural Highlights:
    - Utilizes CSS variables (var(--background-color), var(--secondary-background-color),
      var(--text-color)) so all surfaces adapt seamlessly to Streamlit's Light and Dark modes.
    - Zero hardcoded dark backgrounds or white texts that cause contrast breakage.
    - Uses vibrant, universally legible gradient accents (#4F46E5 to #7C3AED).
    - Polished card borders, soft box shadows, and responsive micro-interactions.
    """
    st.markdown(
        """
        <style>
            /* Import modern Inter font */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }

            /* Hide Streamlit default deploy button & make native header unobtrusive */
            .stDeployButton, [data-testid="stAppDeployButton"], .stAppDeployButton {
                display: none !important;
            }
            header[data-testid="stHeader"] {
                background: transparent !important;
                z-index: 10 !important;
            }

            /* Container padding with clean, natural top spacing */
            .block-container {
                padding-top: 2.75rem !important;
                padding-bottom: 3rem !important;
                padding-left: 2.5rem !important;
                padding-right: 2.5rem !important;
                max-width: 1440px;
            }

            /* Dashboard Header & Hero Section */
            .hero-container {
                margin-bottom: 1.75rem;
            }

            .hero-title {
                font-size: 2.15rem;
                font-weight: 800;
                letter-spacing: -0.03em;
                line-height: 1.2;
                margin-bottom: 0.35rem;
                color: var(--text-color);
            }

            .hero-title .gradient-accent {
                background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-description {
                font-size: 0.98rem;
                color: rgba(128, 128, 128, 0.95);
                font-weight: 400;
                max-width: 820px;
                line-height: 1.5;
            }

            /* Universal KPI Metric Cards */
            .kpi-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 1rem;
                margin-bottom: 1.75rem;
            }

            .kpi-card-box {
                background-color: var(--secondary-background-color);
                border: 1px solid rgba(128, 128, 128, 0.16);
                border-radius: 12px;
                padding: 1.25rem 1.35rem;
                box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
                transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .kpi-card-box:hover {
                transform: translateY(-2px);
                border-color: rgba(99, 102, 241, 0.5);
                box-shadow: 0 8px 20px rgba(99, 102, 241, 0.08);
            }

            .kpi-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.6rem;
            }

            .kpi-card-label {
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: rgba(128, 128, 128, 0.95);
            }

            .kpi-card-icon {
                font-size: 1.15rem;
                opacity: 0.85;
            }

            .kpi-card-value {
                font-size: 1.85rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: var(--text-color);
                line-height: 1.15;
                margin-bottom: 0.4rem;
            }

            .kpi-card-badge {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                font-size: 0.74rem;
                font-weight: 600;
                border-radius: 6px;
                padding: 2px 6px;
            }

            .badge-positive {
                background-color: rgba(16, 185, 129, 0.12);
                color: #10B981;
            }

            .badge-neutral {
                background-color: rgba(99, 102, 241, 0.12);
                color: #6366F1;
            }

            .badge-purple {
                background-color: rgba(168, 85, 247, 0.12);
                color: #A855F7;
            }

            /* Section Callout Card */
            .callout-box {
                background-color: var(--secondary-background-color);
                border-left: 4px solid #6366F1;
                border-radius: 0 10px 10px 0;
                padding: 1rem 1.25rem;
                margin: 1.25rem 0;
                font-size: 0.92rem;
                line-height: 1.5;
                color: var(--text-color);
            }

            /* Tab and Subheader Refinements */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                border-bottom: 1px solid rgba(128, 128, 128, 0.18);
                margin-bottom: 1.25rem;
            }

            .stTabs [data-baseweb="tab"] {
                font-weight: 600;
                font-size: 0.95rem;
                padding: 8px 18px;
                border-radius: 8px 8px 0 0;
            }

            /* Ensure Plotly chart containers blend seamlessly */
            .js-plotly-plot .plotly .main-svg {
                background: transparent !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. INTELLIGENT SCHEMA DETECTION & ROW ARRANGEMENT ENGINE
# ==============================================================================
def detect_column_mappings(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Intelligently identifies the most probable column names for:
    (date_column, product_column, sales_column) from any uploaded or local CSV.
    
    Heuristic Strategy:
    1. Case-insensitive semantic alias matching (e.g., 'InvoiceDate', 'Order Date', 'Amount').
    2. Data-type sampling & parsing validation (detects valid dates and numeric currency fields).
    """
    cols = list(df.columns)
    cols_clean = [str(c).strip() for c in cols]
    col_map = {c.lower(): c for c in cols_clean}

    # 1. Detect Date / Timestamp Column
    date_candidates = [
        "order_date", "orderdate", "invoice_date", "invoicedate", "trans_date", 
        "transaction_date", "purchase_date", "sales_date", "created_at", "date", 
        "datetime", "timestamp", "ds", "period", "day", "month"
    ]
    detected_date = None
    for kw in date_candidates:
        for cl_low, orig in col_map.items():
            if kw == cl_low or kw in cl_low:
                detected_date = orig
                break
        if detected_date:
            break

    # Fallback date detection: test sample conversion
    if not detected_date:
        best_ratio = 0.0
        for orig in cols_clean:
            try:
                sample = df[orig].dropna().head(40)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    ratio = parsed.notna().mean()
                    if ratio > 0.7 and ratio > best_ratio:
                        best_ratio = ratio
                        detected_date = orig
            except Exception:
                continue

    # 2. Detect Sales / Revenue Column
    sales_candidates = [
        "sales", "revenue", "total_amount", "total_sales", "total_price", "amount", 
        "price", "unit_price", "unitprice", "grand_total", "total", "y", "turnover", 
        "value", "subtotal", "cost", "income", "line_total", "order_value"
    ]
    excluded_keywords = ["id", "order_id", "product_id", "customer_id", "zip", "phone", "index", "code"]

    detected_sales = None
    for kw in sales_candidates:
        for cl_low, orig in col_map.items():
            if any(ex in cl_low for ex in excluded_keywords):
                continue
            if kw == cl_low or kw in cl_low:
                detected_sales = orig
                break
        if detected_sales:
            break

    # Fallback sales detection: scan for numeric columns or stripped currency numbers
    if not detected_sales:
        for orig in cols_clean:
            if orig == detected_date:
                continue
            orig_low = orig.lower()
            if any(ex in orig_low for ex in excluded_keywords):
                continue
            try:
                sample = df[orig].dropna().head(40)
                cleaned = sample.astype(str).str.replace(r"[$€£₹,\s]", "", regex=True)
                num_parsed = pd.to_numeric(cleaned, errors="coerce")
                if num_parsed.notna().mean() > 0.75:
                    detected_sales = orig
                    break
            except Exception:
                continue

    # 3. Detect Product / Category Column
    product_candidates = [
        "product_name", "productname", "product_title", "product", "item_name", 
        "item", "description", "category_name", "category", "sub_category", 
        "subcategory", "sku", "brand", "title", "type", "segment"
    ]
    detected_product = None
    for kw in product_candidates:
        for cl_low, orig in col_map.items():
            if orig in (detected_date, detected_sales):
                continue
            if kw == cl_low or kw in cl_low:
                detected_product = orig
                break
        if detected_product:
            break

    # Fallback product detection: categorical/string column with reasonable cardinality
    if not detected_product:
        for orig in cols_clean:
            if orig in (detected_date, detected_sales):
                continue
            if "id" in orig.lower():
                continue
            if df[orig].dtype == object or str(df[orig].dtype) == "category":
                nunique = df[orig].nunique()
                if 1 < nunique <= 500:
                    detected_product = orig
                    break

    return detected_date, detected_product, detected_sales


def auto_arrange_dataframe(
    df: pd.DataFrame,
    date_col: str,
    product_col: Optional[str],
    sales_col: str,
) -> pd.DataFrame:
    """
    Cleans, standardizes, chronologically arranges, and aggregates transactions.
    
    Transforms arbitrary source data into standard schema [ds, product, y]:
    - Parses and normalizes dates to YYYY-MM-DD.
    - Strips currency symbols and converts sales to double precision floats.
    - Aggregates multiple transactions per day per product (summing sales).
    - Sorts chronologically for seamless DuckDB and Facebook Prophet ingestion.
    """
    work_df = df.copy()

    # 1. Normalize Date Column to datetime
    work_df["ds"] = pd.to_datetime(work_df[date_col], errors="coerce").dt.floor("D")
    work_df = work_df.dropna(subset=["ds"])

    # 2. Clean Numeric Sales Column
    if work_df[sales_col].dtype == object or str(work_df[sales_col].dtype) == "category":
        cleaned_str = (
            work_df[sales_col]
            .astype(str)
            .str.replace(r"[$€£₹\s,]", "", regex=True)
            .str.replace(r"[^\d.-]", "", regex=True)
        )
        work_df["y"] = pd.to_numeric(cleaned_str, errors="coerce")
    else:
        work_df["y"] = pd.to_numeric(work_df[sales_col], errors="coerce")

    work_df = work_df.dropna(subset=["y"])

    # 3. Standardize Product Column
    if product_col and product_col in work_df.columns and product_col != "[None / Single Stream]":
        work_df["product"] = work_df[product_col].astype(str).str.strip()
        work_df["product"] = work_df["product"].replace({"": "Unspecified", "nan": "Unspecified"})
    else:
        work_df["product"] = "All Items"

    # 4. Chronological Row Arrangement & Daily Rollup via DuckDB
    # Roll up multiple intra-day transactions so Prophet receives daily points
    query = """
        SELECT 
            CAST(ds AS DATE) AS ds,
            CAST(product AS VARCHAR) AS product,
            CAST(SUM(y) AS DOUBLE) AS y
        FROM work_df
        GROUP BY ds, product
        ORDER BY ds ASC, product ASC
    """
    arranged_df = duckdb.query(query).df()
    arranged_df["ds"] = pd.to_datetime(arranged_df["ds"])
    return arranged_df


@st.cache_data(show_spinner=False)
def load_and_preprocess_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads local dataset and automatically arranges columns and rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source data file was not found at: {csv_path}")

    raw_csv_df = pd.read_csv(csv_path)
    date_col, prod_col, sales_col = detect_column_mappings(raw_csv_df)

    if not date_col or not sales_col:
        raise ValueError("Could not automatically identify Date and Sales columns in CSV.")

    arranged = auto_arrange_dataframe(raw_csv_df, date_col, prod_col, sales_col)
    return raw_csv_df, arranged


# ==============================================================================
# 4. AGGREGATION & FILTERING PIPELINE
# ==============================================================================
def apply_product_filter_and_aggregation(
    df_raw: pd.DataFrame, selected_product: str
) -> pd.DataFrame:
    """
    Applies product-level filtering or multi-product rollup aggregation using DuckDB.

    Args:
        df_raw (pd.DataFrame): Raw ingested dataset with [ds, product, y].
        selected_product (str): Target product filter or 'All Products'.

    Returns:
        pd.DataFrame: Clean aggregated dataset with columns [ds, y].
    """
    if df_raw.empty:
        return pd.DataFrame(columns=["ds", "y"])

    if selected_product == "All Products":
        # Group across all SKUs by date
        agg_query = """
            SELECT 
                ds, 
                SUM(y) AS y 
            FROM df_raw 
            GROUP BY ds 
            ORDER BY ds ASC
        """
        result_df = duckdb.query(agg_query).df()
    else:
        # Filter strictly for selected SKU
        filter_query = """
            SELECT 
                ds, 
                y 
            FROM df_raw 
            WHERE product = ? 
            ORDER BY ds ASC
        """
        result_df = duckdb.query(filter_query, params=[selected_product]).df()

    result_df["ds"] = pd.to_datetime(result_df["ds"])
    result_df["y"] = pd.to_numeric(result_df["y"], errors="coerce")
    return result_df


# ==============================================================================
# 5. PREDICTIVE FORECASTING ENGINE (FACEBOOK PROPHET)
# ==============================================================================
def generate_prophet_forecast(
    df: pd.DataFrame,
    forecast_horizon_days: int = 30,
    interval_width: float = 0.95,
) -> Optional[pd.DataFrame]:
    """
    Initializes and fits an additive Bayesian time-series model (Prophet)
    to the filtered historical data and projects the future sales horizon.

    Production Safeguards:
    - Minimum sample guard (>= 2 observations).
    - Adaptive seasonality calibration based on dataset sample length.
    - Yields point forecast ('yhat') and Bayesian confidence envelope ('yhat_lower', 'yhat_upper').

    Args:
        df (pd.DataFrame): Ingestion dataset with ['ds', 'y'].
        forecast_horizon_days (int): Number of days forward to predict.
        interval_width (float): Credible interval width (e.g., 0.95 for 95%).

    Returns:
        Optional[pd.DataFrame]: Projection DataFrame or None if insufficient data.
    """
    if df is None or len(df) < 2:
        return None

    sample_size = len(df)
    enable_weekly = sample_size >= 14
    enable_yearly = sample_size >= 365

    model = Prophet(
        interval_width=interval_width,
        daily_seasonality=False,
        weekly_seasonality=enable_weekly,
        yearly_seasonality=enable_yearly,
        growth="linear",
    )

    model.fit(df)

    future_dates = model.make_future_dataframe(periods=forecast_horizon_days, freq="D")
    forecast = model.predict(future_dates)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ==============================================================================
# 6. UNIVERSAL PLOTLY VISUALIZATION MODULES (THEME ADAPTIVE)
# ==============================================================================
def create_historical_chart(df: pd.DataFrame, series_title: str) -> go.Figure:
    """
    Creates an interactive historical trajectory line chart using Plotly Express.
    Configured with clean title alignment and transparent backgrounds.
    """
    fig = px.line(
        df,
        x="ds",
        y="y",
        labels={"ds": "Date", "y": "Sales Volume ($)"},
        markers=True,
    )

    fig.update_traces(
        line=dict(color="#4F46E5", width=3, shape="spline"),
        marker=dict(size=7, color="#06B6D4", line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate="<b>Date</b>: %{x|%b %d, %Y}<br><b>Sales</b>: $%{y:,.2f}<extra></extra>",
    )

    fig.update_layout(
        title=dict(
            text=f"Historical Sales Trajectory ({series_title})",
            x=0.01,
            y=0.96,
            xanchor="left",
            font=dict(size=14, family="Inter, sans-serif"),
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=15, r=20, t=55, b=25),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
            showline=True,
            linecolor="rgba(128, 128, 128, 0.3)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
            showline=True,
            linecolor="rgba(128, 128, 128, 0.3)",
            tickprefix="$",
        ),
    )
    return fig


def create_forecast_chart(
    forecast_df: pd.DataFrame, historical_df: pd.DataFrame, series_title: str
) -> go.Figure:
    """
    Creates an interactive forecast figure via Plotly go.Figure.
    Plots the predicted sales trajectory with shaded confidence intervals
    and places legend cleanly below to eliminate title collisions.
    """
    fig = go.Figure()

    # 1. Lower Bound Reference Line (Invisible)
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="Lower Bound",
        )
    )

    # 2. Upper Bound with Confidence Shading
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 102, 241, 0.18)",
            name="Confidence Interval",
            hovertemplate="<b>Upper Bound</b>: $%{y:,.2f}<extra></extra>",
        )
    )

    # 3. Forecast Trend Line (yhat)
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat"],
            mode="lines",
            line=dict(color="#4F46E5", width=3, dash="solid"),
            name="Forecast (yhat)",
            hovertemplate="<b>Projected</b>: $%{y:,.2f}<extra></extra>",
        )
    )

    # 4. Actual Historical Ground Truth
    fig.add_trace(
        go.Scatter(
            x=historical_df["ds"],
            y=historical_df["y"],
            mode="markers+lines",
            marker=dict(size=6, color="#06B6D4", symbol="circle"),
            line=dict(color="rgba(6, 182, 212, 0.5)", width=1.5, dash="dot"),
            name="Actual Recorded",
            hovertemplate="<b>Actual</b>: $%{y:,.2f}<extra></extra>",
        )
    )

    # Layout with bottom legend to prevent any title collision
    fig.update_layout(
        title=dict(
            text=f"Predictive Horizon & Bayesian Bounds ({series_title})",
            x=0.01,
            y=0.96,
            xanchor="left",
            font=dict(size=14, family="Inter, sans-serif"),
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=15, r=20, t=55, b=45),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
            showline=True,
            linecolor="rgba(128, 128, 128, 0.3)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
            showline=True,
            linecolor="rgba(128, 128, 128, 0.3)",
            tickprefix="$",
        ),
    )
    return fig


def create_sku_comparison_chart(raw_df: pd.DataFrame) -> go.Figure:
    """
    Constructs an executive-tier horizontal ranking chart of Product SKUs by total revenue.
    Eliminates multi-row legend wrapping, prevents title overlap, and displays clean data values.
    """
    if raw_df.empty or "product" not in raw_df.columns:
        return go.Figure()

    # Aggregate total sales per product SKU
    sku_totals = (
        raw_df.groupby("product", as_index=False)["y"]
        .sum()
        .sort_values(by="y", ascending=True)
    )

    total_skus = len(sku_totals)
    is_truncated = total_skus > 10

    # Limit to Top 10 for clean visual hierarchy
    if is_truncated:
        display_totals = sku_totals.tail(10).copy()
        chart_title = f"Top 10 Product SKUs by Revenue (of {total_skus:,} Total)"
    else:
        display_totals = sku_totals.copy()
        chart_title = "Product SKU Revenue Ranking"

    # Truncate overly long UUIDs or SKU names for clean y-axis alignment
    display_totals["display_name"] = display_totals["product"].apply(
        lambda p: (str(p)[:14] + "…") if len(str(p)) > 16 else str(p)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=display_totals["y"],
            y=display_totals["display_name"],
            orientation="h",
            marker=dict(
                color=display_totals["y"],
                colorscale=[[0, "#818cf8"], [1, "#4F46E5"]],
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
            ),
            customdata=display_totals["product"],
            hovertemplate="<b>Product SKU</b>: %{customdata}<br><b>Total Revenue</b>: $%{x:,.2f}<extra></extra>",
            text=display_totals["y"].apply(lambda val: f"${val:,.0f}"),
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(
            text=chart_title,
            x=0.01,
            y=0.96,
            xanchor="left",
            font=dict(size=14, family="Inter, sans-serif"),
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,  # Eliminates the multi-row legend collision completely
        margin=dict(l=15, r=45, t=55, b=25),
        hovermode="closest",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
            showline=True,
            linecolor="rgba(128, 128, 128, 0.3)",
            tickprefix="$",
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="rgba(128, 128, 128, 0.3)",
        ),
    )
    return fig


# ==============================================================================
# 7. MAIN APPLICATION CONTROLLER
# ==============================================================================
def main() -> None:
    """
    Main orchestration loop controlling data ingestion, responsive UI rendering,
    forecasting pipelines, and simulation workflows.
    """
    # 1. Apply universal adaptive styling
    inject_adaptive_css()

    # 2. Data source path configuration
    DATA_PATH = os.path.join("data", "sales_sample.csv")

    # 3. Hero Section
    st.markdown(
        """
        <div class="hero-container">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); color: #6366F1; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.75rem; letter-spacing: 0.03em;">⚡ APEXANALYTICS ENTERPRISE</div>
            <div class="hero-title">Predictive Sales Intelligence & <span class="gradient-accent">Forecasting Suite</span></div>
            <div class="hero-description">
                High-performance time-series modeling powered by in-process DuckDB queries and Bayesian Prophet algorithms.
                Seamlessly responsive across light and dark themes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Data Ingestion & Safeguards (Direct CSV Upload or Local File)
    st.sidebar.markdown("## 📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        label="Import Custom CSV",
        type=["csv"],
        help="Upload any sales or e-commerce CSV. Columns and rows will be arranged automatically.",
    )

    if uploaded_file is not None:
        try:
            temp_df = pd.read_csv(uploaded_file)
            active_data_label = uploaded_file.name
        except Exception as upload_err:
            st.sidebar.error(f"❌ Could not parse uploaded CSV: {upload_err}")
            st.stop()
    else:
        active_data_label = os.path.basename(DATA_PATH)
        try:
            if not os.path.exists(DATA_PATH):
                st.error(f"❌ Target data file not found at: `{DATA_PATH}`. Please place your CSV in `data/` or upload one.")
                st.stop()
            temp_df = pd.read_csv(DATA_PATH)
        except Exception as exc:
            st.error(f"❌ Failed to load source data: {str(exc)}")
            st.stop()

    if temp_df.empty:
        st.warning("⚠️ The dataset contains 0 records.")
        st.stop()

    # Automatic Column Detection
    auto_date, auto_prod, auto_sales = detect_column_mappings(temp_df)
    cols_list = list(temp_df.columns)

    # Interactive Column Mapping / Alignment Console
    with st.sidebar.expander("⚙️ Auto-Arranged Schema Mapping", expanded=(auto_date is None or auto_sales is None)):
        st.caption("The engine auto-detects fields. You can override below if desired:")
        
        # 1. Date Selection
        date_default_idx = cols_list.index(auto_date) if auto_date in cols_list else 0
        selected_date_col = st.selectbox("📅 Date Column", options=cols_list, index=date_default_idx)
        
        # 2. Product / Category Selection
        prod_options = ["[None / Single Stream]"] + cols_list
        prod_default_idx = prod_options.index(auto_prod) if auto_prod in prod_options else 0
        selected_prod_col = st.selectbox("🏷️ Product / Category Column", options=prod_options, index=prod_default_idx)
        
        # 3. Sales / Revenue Selection
        sales_candidates = [c for c in cols_list if c != selected_date_col]
        sales_default_idx = sales_candidates.index(auto_sales) if auto_sales in sales_candidates else 0
        selected_sales_col = st.selectbox("💵 Sales / Revenue Column", options=sales_candidates, index=sales_default_idx)

    # Automatically arrange rows, normalize dates, clean currencies, and aggregate transactions
    try:
        raw_df = auto_arrange_dataframe(
            temp_df,
            date_col=selected_date_col,
            product_col=selected_prod_col,
            sales_col=selected_sales_col,
        )

        if uploaded_file is not None:
            st.sidebar.success(
                f"✅ **Auto-Arranged Successfully!**\n\n"
                f"- **Date:** `{selected_date_col}`\n"
                f"- **Sales:** `{selected_sales_col}`\n"
                f"- **Group:** `{selected_prod_col}`\n\n"
                f"Consolidated **{len(temp_df):,}** transactions into **{len(raw_df):,}** daily records."
            )
    except Exception as arrange_err:
        st.sidebar.error(f"❌ Error arranging data: {arrange_err}")
        st.stop()

    if raw_df.empty or len(raw_df) < 2:
        st.warning("⚠️ Insufficient valid date/sales records found after arrangement. Need at least 2 observations.")
        st.stop()

    # 6. Sidebar Control Console
    st.sidebar.markdown("## 🎛️ Parameters & Filters")
    st.sidebar.markdown("Configure analysis boundaries and forecasting hyperparameters.")

    # Product Filter Dropdown
    available_products = sorted(raw_df["product"].dropna().unique().tolist())
    product_options = ["All Products"] + available_products

    selected_product = st.sidebar.selectbox(
        label="Target Product Scope",
        options=product_options,
        index=0,
        help="Select 'All Products' to aggregate the complete organization or focus on a single SKU.",
    )

    # Forecast Horizon
    forecast_horizon = st.sidebar.slider(
        label="Forecast Horizon (Days Forward)",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
        help="The projection window into the future for Facebook Prophet.",
    )

    # Bayesian Credible Interval Width
    interval_choice = st.sidebar.select_slider(
        label="Bayesian Credible Interval",
        options=[0.80, 0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}%",
        help="Width of the predictive uncertainty interval envelope.",
    )

    # Diagnostic metadata
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Telemetry")
    st.sidebar.markdown(
        f"""
        - **Active Dataset:** `{active_data_label}`
        - **Total Records:** `{len(raw_df):,}`
        - **Unique Products:** `{len(available_products)}`
        - **Start Date:** `{raw_df['ds'].min().strftime('%Y-%m-%d')}`
        - **End Date:** `{raw_df['ds'].max().strftime('%Y-%m-%d')}`
        """
    )

    # 7. Apply In-Process Aggregation via DuckDB
    filtered_df = apply_product_filter_and_aggregation(raw_df, selected_product)

    if filtered_df.empty or len(filtered_df) < 2:
        st.warning(f"⚠️ Insufficient records for **{selected_product}**. Minimum 2 observations required.")
        st.stop()

    # 8. Universal KPI Metric Cards
    total_sales = filtered_df["y"].sum()
    avg_daily_sales = filtered_df["y"].mean()
    max_single_day = filtered_df["y"].max()

    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card-box">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Total Historical Revenue</span>
                    <span class="kpi-card-icon">💰</span>
                </div>
                <div class="kpi-card-value">${total_sales:,.2f}</div>
                <div><span class="kpi-card-badge badge-positive">▲ Recorded Basis</span></div>
            </div>
            <div class="kpi-card-box">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Mean Daily Volume</span>
                    <span class="kpi-card-icon">📈</span>
                </div>
                <div class="kpi-card-value">${avg_daily_sales:,.2f}</div>
                <div><span class="kpi-card-badge badge-neutral">● Rolling Daily Avg</span></div>
            </div>
            <div class="kpi-card-box">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Peak Daily Record</span>
                    <span class="kpi-card-icon">🚀</span>
                </div>
                <div class="kpi-card-value">${max_single_day:,.2f}</div>
                <div><span class="kpi-card-badge badge-purple">★ Highest Daily Total</span></div>
            </div>
            <div class="kpi-card-box">
                <div class="kpi-card-header">
                    <span class="kpi-card-label">Active Scope</span>
                    <span class="kpi-card-icon">🎯</span>
                </div>
                <div class="kpi-card-value" style="font-size: 1.35rem; color: #6366F1;">{selected_product}</div>
                <div><span class="kpi-card-badge badge-neutral">{forecast_horizon}-Day Projection</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 9. Multi-Tab Workflow
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔮 Predictive Forecasting",
            "📈 Historical Trajectory",
            "⚙️ Scenario Simulator",
            "📋 Data Audit & Export",
        ]
    )

    # --------------------------------------------------------------------------
    # TAB 1: PREDICTIVE FORECASTING
    # --------------------------------------------------------------------------
    with tab1:
        with st.spinner("Calibrating Bayesian parameters and fitting Facebook Prophet model..."):
            forecast_df = generate_prophet_forecast(
                filtered_df,
                forecast_horizon_days=forecast_horizon,
                interval_width=interval_choice,
            )

        if forecast_df is not None and not forecast_df.empty:
            forecast_fig = create_forecast_chart(forecast_df, filtered_df, selected_product)
            st.plotly_chart(forecast_fig, use_container_width=True, theme="streamlit")

            # Projected Future Calculation
            future_mask = forecast_df["ds"] > filtered_df["ds"].max()
            future_volume = forecast_df.loc[future_mask, "yhat"].sum()
            future_lower = forecast_df.loc[future_mask, "yhat_lower"].sum()
            future_upper = forecast_df.loc[future_mask, "yhat_upper"].sum()

            st.markdown(
                f"""
                <div class="callout-box">
                    <strong>💡 Forward-Looking Revenue Projections ({forecast_horizon} Days):</strong><br>
                    Projected cumulative sales for <strong>{selected_product}</strong> is estimated at
                    <strong>${future_volume:,.2f}</strong>. Under the {int(interval_choice * 100)}% Bayesian confidence
                    boundary, revenue is modeled between <strong>${future_lower:,.2f}</strong> and
                    <strong>${future_upper:,.2f}</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("Forecast model could not converge. Please provide additional observations.")

    # --------------------------------------------------------------------------
    # TAB 2: HISTORICAL TRAJECTORY
    # --------------------------------------------------------------------------
    with tab2:
        col_hist_a, col_hist_b = st.columns([1.5, 1])

        with col_hist_a:
            historical_fig = create_historical_chart(filtered_df, selected_product)
            st.plotly_chart(historical_fig, use_container_width=True, theme="streamlit")

        with col_hist_b:
            sku_fig = create_sku_comparison_chart(raw_df)
            st.plotly_chart(sku_fig, use_container_width=True, theme="streamlit")

    # --------------------------------------------------------------------------
    # TAB 3: SCENARIO SIMULATOR
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("### 🎲 What-If Scenario Modeling")
        st.markdown(
            "Test how macroeconomic or pricing changes might influence the forward-looking sales trajectory."
        )

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            growth_adjustment = st.slider(
                "Sales Volume Shock (%)",
                min_value=-50,
                max_value=50,
                value=10,
                step=5,
                help="Apply a percentage shock to the baseline forecast trajectory.",
            )

        with sim_col2:
            st.metric(
                label="Adjusted Horizon Total",
                value=f"${future_volume * (1 + growth_adjustment / 100):,.2f}"
                if forecast_df is not None
                else "N/A",
                delta=f"{growth_adjustment:+d}% vs Baseline",
            )

        if forecast_df is not None:
            sim_df = forecast_df.copy()
            sim_df["yhat_scenario"] = sim_df["yhat"] * (1 + growth_adjustment / 100)

            sim_fig = go.Figure()
            sim_fig.add_trace(
                go.Scatter(
                    x=sim_df["ds"],
                    y=sim_df["yhat"],
                    mode="lines",
                    line=dict(color="#6366F1", dash="dash"),
                    name="Baseline Forecast",
                )
            )
            sim_fig.add_trace(
                go.Scatter(
                    x=sim_df["ds"],
                    y=sim_df["yhat_scenario"],
                    mode="lines",
                    line=dict(color="#10B981" if growth_adjustment >= 0 else "#EF4444", width=3),
                    name=f"Scenario ({growth_adjustment:+d}%)",
                )
            )
            sim_fig.update_layout(
                title=dict(
                    text=f"Scenario Simulation vs Baseline Forecast ({growth_adjustment:+d}%)",
                    x=0.01,
                    y=0.96,
                    xanchor="left",
                    font=dict(size=14, family="Inter, sans-serif"),
                ),
                paper_bgcolor="rgba(0, 0, 0, 0)",
                plot_bgcolor="rgba(0, 0, 0, 0)",
                margin=dict(l=15, r=20, t=55, b=45),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.14,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(0,0,0,0)",
                ),
                yaxis=dict(tickprefix="$", gridcolor="rgba(128, 128, 128, 0.15)"),
                xaxis=dict(gridcolor="rgba(128, 128, 128, 0.15)"),
            )
            st.plotly_chart(sim_fig, use_container_width=True, theme="streamlit")

    # --------------------------------------------------------------------------
    # TAB 4: DATA AUDIT & EXPORT
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("### 📋 Structured Forecast Data Table")
        st.markdown("Audit the individual point estimates and confidence intervals produced by the model.")

        if forecast_df is not None:
            export_df = forecast_df.copy()
            export_df["ds"] = export_df["ds"].dt.strftime("%Y-%m-%d")
            export_df = export_df.rename(
                columns={
                    "ds": "Date",
                    "yhat": "Forecast ($)",
                    "yhat_lower": "Lower Bound ($)",
                    "yhat_upper": "Upper Bound ($)",
                }
            )

            numeric_fields = ["Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]
            export_df[numeric_fields] = export_df[numeric_fields].round(2)

            st.dataframe(export_df, use_container_width=True, hide_index=True)

            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Forecast CSV",
                data=csv_data,
                file_name=f"forecast_{selected_product.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                help="Export the computed dataset directly to CSV for external BI tools.",
            )

    # 10. Raw Data Expander (Maintained for full backwards compatibility)
    with st.expander("🔍 View Raw Historical Dataset (DuckDB Ingest)", expanded=False):
        st.dataframe(raw_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
