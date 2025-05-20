import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import prepro
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM

# Styling adjustments
st.set_page_config(layout="wide", page_title="Dashboard Group 15", page_icon="📊")
st.markdown(
    """
    <style>
    body {background-color: #f4f4f9;}
    .stApp {background-color: #ffffff; color: #333333; font-family: 'Arial', sans-serif;}
    .css-18e3th9 {padding: 2rem;}
    .block-container {padding-top: 2rem;}
    .stMetric {background: linear-gradient(135deg, #4b79a1, #283e51); color: #ffffff; border-radius: 10px; padding: 15px;}
    .stPlotlyChart {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
    div.stButton > button:hover {
        background-color: #45a049;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📊 ADashboard - Group 15")


# Load Data
def load_data(path: str):
    data = pd.read_csv(path)
    return data


if "page" not in st.session_state:
    st.session_state.page = "upload"
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

if st.session_state.page == "upload":
    uppath = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uppath is None:
        st.stop()
    st.session_state.df = load_data(uppath)
    if st.session_state.df is not None:
        with st.expander("🔍 Data Preview"):
            st.dataframe(st.session_state.df)
            st.write(st.session_state.df.dtypes)
        if st.button("🚀 Go to Dashboard"):
            st.session_state.page = "Dashboard"
            st.session_state.df = prepro.clean_data(st.session_state.df)
            st.rerun()

elif st.session_state.page == "Dashboard":
    # Data Preparation
    salesVsTime = prepro.prep_sales(st.session_state.df)
    groupByHour = prepro.prep_grouphour(st.session_state.df)
    groupByProduct = prepro.prep_groupProduct(st.session_state.df)
    groupByKategori = prepro.prep_groupKategori(st.session_state.df)
    groupByCustomer = prepro.customer_segmentation(
        prepro.prep_customer(st.session_state.df)  # Pastikan data sudah di-prep
    )

    # Top Metrics
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="📅 Avg Daily Revenue",
                value=f"Rp {salesVsTime['nominal_transaksi'].mean():,.2f}",
                delta_color="inverse",
            )
        with col2:
            st.metric(
                label="📦 Avg Daily Products Sold",
                value=f"{salesVsTime['banyak_produk'].mean():,.2f}",
                delta_color="off",
            )
        with col3:
            st.metric(
                label="🛒 Avg Daily Transactions",
                value=f"{salesVsTime['banyak_transaksi'].mean():,.2f}",
                delta_color="normal",
            )

    # Time-Based Analysis
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                salesVsTime,
                x="Tanggal & Waktu",
                y="banyak_transaksi",
                title="📈 Transactions Over Time",
                color_discrete_sequence=["#005f73"],
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                groupByHour,
                x="Jam",
                y="Jumlah_produk",
                title="⏳ Products Ordered Per Hour",
                color_discrete_sequence=["#ee9b00"],
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Product Analysis
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                groupByProduct.nlargest(8, "Jumlah_produk"),
                names="Nama Produk",
                values="Jumlah_produk",
                title="🍩 Best-Selling Products",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Teal,
            )
            st.plotly_chart(fig)
        with col2:
            fig = px.bar(
                groupByKategori,
                x="Kategori",
                y="Total_omset",
                title="📊 Revenue by Category",
                color="Kategori",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            st.plotly_chart(fig)

    # Customer Segmentation (New Section)
    with st.container():
        st.subheader("👥 Customer Segmentation Analysis")
        valueCCount = groupByCustomer["cluster"].value_counts().reset_index()
        valueCCount.columns = ["Cluster", "Count"]

        col1, col2 = st.columns([2, 3])
        with col1:
            fig = px.bar(
                valueCCount,
                x="Cluster",
                y="Count",
                color="Cluster",
                title="📊 Customer Distribution by Cluster",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            metrics = ["totSpen", "totJum", "totJenPro", "totKat"]
            titles = [
                "Rp Avg Spending",
                "Products/Order",
                "Product Varieties",
                "Categories Ordered",
            ]
            cols = st.columns(4)
            for i, (col, metric, title) in enumerate(zip(cols, metrics, titles)):
                with col:
                    avg_value = groupByCustomer[metric].mean()
                    st.metric(
                        label=title, value=f"{avg_value:,.2f}", delta_color="normal"
                    )

    # Advanced Analytics
    with st.container():
        st.subheader("📌 Sales Heatmap by Hour")
        heatmap_data = salesVsTime.pivot_table(
            index="Tanggal & Waktu",
            columns="banyak_transaksi",
            values="nominal_transaksi",
            aggfunc="sum",
        )
        fig = px.imshow(
            heatmap_data,
            labels=dict(color="Revenue"),
            aspect="auto",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.container():
        st.subheader("📌 Cumulative Revenue Over Time")
        salesVsTime["Cumulative_Revenue"] = salesVsTime["nominal_transaksi"].cumsum()
        fig = px.line(
            salesVsTime,
            x="Tanggal & Waktu",
            y="Cumulative_Revenue",
            title="📈 Cumulative Revenue Growth",
            color_discrete_sequence=["#ff5733"],
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("📍 *Developed by Group 15* 🚀")
    # Prediksi Omset Section
    st.subheader("🔮 Revenue Prediction")

    # Input parameter prediksi
    prediction_days = st.slider("Pilih Jumlah Hari Prediksi", 7, 90, 30)

    # Jalankan prediksi
    forecast, model = prepro.predict_revenue(salesVsTime, periods=prediction_days)

    # Visualisasi prediksi
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=salesVsTime["Tanggal & Waktu"],
            y=salesVsTime["nominal_transaksi"],
            name="Historical Revenue",
            line=dict(color="#005f73"),
        )
    )

    # Prediksi
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Predicted Revenue",
            line=dict(color="#ee9b00", dash="dash"),
        )
    )

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            fill="tonexty",
            mode="none",
            name="Confidence Interval",
            fillcolor="rgba(238, 155, 0, 0.2)",
        )
    )

    fig.update_layout(
        title="Revenue Prediction for Next {} Days".format(prediction_days),
        xaxis_title="Date",
        yaxis_title="Revenue",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics prediksi
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Next Day Prediction",
            value=f"Rp {forecast['yhat'].iloc[-prediction_days]:,.2f}",
            delta_color="normal",
        )
    with col2:
        st.metric(
            label="Growth Rate",
            value=f"{(forecast['yhat'].iloc[-1]/forecast['yhat'].iloc[-2]-1):.2%}",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            label="Prediction MAE",
            value=f"Rp {mean_absolute_error(salesVsTime['nominal_transaksi'], forecast['yhat'][:-prediction_days]):,.2f}",
        )
# Di bagian LLM Insight Generator
with st.container():
    st.subheader("🧠 AI-Powered Insight")
    # Persiapan data untuk LLM
    total_transaksi = salesVsTime["banyak_transaksi"].sum()
    avg_revenue = salesVsTime["nominal_transaksi"].mean()
    best_product = groupByProduct.nlargest(1, "Jumlah_produk")["Nama Produk"].values[0]
    top_category = groupByKategori.nlargest(1, "Total_omset")["Kategori"].values[0]

    # Input pertanyaan
    user_query = st.text_input("Ajukan pertanyaan tentang data Anda:")
    generate_insight = st.button("Generate Insight Otomatis")

    # Proses LLM
    if generate_insight or user_query:
        with st.spinner("Memproses dengan Hugging Face..."):
            hf_llm = prepro.load_hf_model()
            if user_query:
                response = hf_llm(
                    user_query
                    + f"\nData pendukung: Total transaksi {total_transaksi}, "
                    + f"Rata-rata omset {avg_revenue}, Produk terlaris {best_product}"
                )
            else:
                prompt = f"""
                Berikan insight bisnis dari data berikut:
                - Total transaksi: {total_transaksi}
                - Rata-rata omset: {avg_revenue}
                - Produk terlaris: {best_product}
                - Kategori teratas: {top_category}
                """
                response = hf_llm(prompt)

            # Tangani format respons
            if isinstance(response, list):
                if len(response) > 0:
                    if isinstance(response[0], dict):
                        generated_text = response[0].get(
                            "generated_text", "Tidak ada respons"
                        )
                    else:
                        generated_text = response[0]  # Jika langsung string
                else:
                    generated_text = "Model tidak menghasilkan output"
            else:
                generated_text = response

            st.text_area("Hasil Analisis AI", value=generated_text, height=200)
