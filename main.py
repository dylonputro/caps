import pandas as pd 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import prepro
import ollama
from neuralforecast.models import NBEATS
from neuralforecast import NeuralForecast

st.set_page_config(layout="wide", page_title="Dashboard Group 15", page_icon="📊")
st.title("Adashboard By Group 15")

# Fungsi prediksi dengan NBEATS
def predict_revenue_nbeats(df, prediction_days=30):
    df_nbeats = df[['Tanggal & Waktu', 'nominal_transaksi']].copy()
    df_nbeats.rename(columns={'Tanggal & Waktu': 'ds', 'nominal_transaksi': 'y'}, inplace=True)

    model = NBEATS(lags=12, input_size=10, n_epochs=100)
    nf = NeuralForecast(models=[model], freq='D')
    nf.fit(df_nbeats)
    forecast = nf.predict(steps=prediction_days)

    predicted_df = pd.DataFrame({
        'Tanggal & Waktu': forecast.index,
        'nominal_transaksi': forecast['NBEATS']
    })
    predicted_df.set_index('Tanggal & Waktu', inplace=True)
    return predicted_df

# Fungsi load data 
def load_data(path: str):
    data = pd.read_csv(path)
    return data 

if "page" not in st.session_state:
    st.session_state.page = "upload"
if "df" not in st.session_state:
    st.session_state.df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

# Halaman Upload
if st.session_state.page == "upload":
    uppath = st.file_uploader("Drop your file please")
    if uppath is None:  
        st.stop()
    st.session_state.df = load_data(uppath)
    if st.session_state.df is not None:
        with st.expander("Data Preview"):  
            st.dataframe(st.session_state.df)
            st.write(st.session_state.df.dtypes)
        standard_columns = ['Tanggal & Waktu', 'ID Struk', 'Tipe Penjualan', 'Nama Pelanggan','Nama Produk', 'Kategori', 'Jumlah Produk', 'Harga Produk', 'Metode Pembayaran']
        submitted_columns = st.session_state.df.columns.tolist()
        if set(submitted_columns) == set(standard_columns):
            st.success("✅ Column names match the standard.")
        else:
            st.warning("⚠️ Column names do not match the standard!")
            for i, col in enumerate(standard_columns):
                default_value = st.session_state.column_mapping.get(col, None)
                st.session_state.column_mapping[col] = st.selectbox(
                    f"Select column for '{col}'", submitted_columns, index=submitted_columns.index(default_value) if default_value in submitted_columns else 0, key=f"col_{i}"
                )
            if st.button("Change"):
                st.session_state.df = prepro.fix_column_name(st.session_state.df, st.session_state.column_mapping)
    if st.button("Continue"):
        st.session_state.page = "Dashboard"
        st.session_state.df = prepro.clean_data(st.session_state.df)
        st.experimental_rerun()

# Halaman Dashboard
elif st.session_state.page == "Dashboard":
    salesVsTime = prepro.prep_sales(st.session_state.df)
    groupByCustomer = prepro.prep_customer(st.session_state.df)
    groupByHour = prepro.prep_grouphour(st.session_state.df)
    groupByProduct = prepro.prep_groupProduct(st.session_state.df)
    groupByKategori = prepro.prep_groupKategori(st.session_state.df)

    # Sales Summary Indicators
    with st.container():
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=salesVsTime['nominal_transaksi'].mean(),
                title={"text": "Rata-Rata Pemasukan Harian"},
                delta={"reference": salesVsTime['nominal_transaksi'].mean() - (salesVsTime["nominal_transaksi"].iloc[-1] - salesVsTime["nominal_transaksi"].iloc[-2]), "relative": False, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                number={"font": {"size": 60, "color": "#1F2A44"}}
            ))
            fig.update_layout(width=400, height=150)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=salesVsTime['banyak_produk'].mean(),
                title={"text": "Rata-Rata Produk Harian"},
                delta={"reference": salesVsTime['banyak_produk'].mean() - (salesVsTime["banyak_produk"].iloc[-1] - salesVsTime["banyak_produk"].iloc[-2]), "relative": False, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                number={"font": {"size": 60, "color": "#1F2A44"}}
            ))
            fig.update_layout(width=400, height=150)
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=salesVsTime['banyak_transaksi'].mean(),
                title={"text": "Rata-Rata Transaksi Dalam Harian"},
                delta={"reference": salesVsTime['banyak_transaksi'].mean() - (salesVsTime["banyak_transaksi"].iloc[-1] - salesVsTime["banyak_transaksi"].iloc[-2]), "relative": False, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                number={"font": {"size": 60, "color": "#1F2A44"}}
            ))
            fig.update_layout(width=400, height=150)
            st.plotly_chart(fig, use_container_width=True)

    # Grafik Sales dan Jam
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_transaksi", title="Banyak Transaksi Seiring Waktu")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(groupByHour, x="Jam", y="Jumlah_produk", title="Rata-rata Banyak Produk yang dipesan dalam Seharian")
            st.plotly_chart(fig, use_container_width=True)

    # Grafik Banyak Ragam Produk & Pemasukan + Prediksi NBEATS
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(salesVsTime, x="Tanggal & Waktu", y="banyak_jenis_produk", title="Banyak Ragam Produk Seiring Waktu")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            datasales = salesVsTime[["Tanggal & Waktu", "nominal_transaksi"]].copy()
            datasales.set_index('Tanggal & Waktu', inplace=True)

            # Grafik asli pemasukan
            fig2 = px.line(datasales, x=datasales.index, y="nominal_transaksi", title="Banyak Pemasukan Seiring Waktu")
            st.plotly_chart(fig2, use_container_width=True)

            if st.button('Make Prediction'):
                # Reset index supaya 'Tanggal & Waktu' jadi kolom lagi
                predicted_df = predict_revenue_nbeats(datasales.reset_index(), prediction_days=30)

                # Buat grafik baru gabung asli + prediksi
                fig_pred = px.line(datasales, x=datasales.index, y='nominal_transaksi', title="Pemasukan Seiring Waktu (Dengan Prediksi N-BEATS)")
                fig_pred.add_traces(go.Scatter(
                    x=predicted_df.index,
                    y=predicted_df['nominal_transaksi'],
                    mode='lines',
                    name='Prediksi N-BEATS',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig_pred, use_container_width=True)

    #Product Dashboard             
    with st.container() : 
        col21, col22 = st.columns(2)
        with col21 :    
            top_5 = groupByProduct.nlargest(8, "Jumlah_produk")
            other_total = groupByProduct.loc[~groupByProduct["Nama Produk"].isin(top_5["Nama Produk"]), "Jumlah_produk"].sum()
            other_row = pd.DataFrame([{"Nama Produk": "Other", "Jumlah_produk": other_total}])
            top_5 = pd.concat([top_5, other_row], ignore_index=True)
            fig = px.pie(top_5, names="Nama Produk", values="Jumlah_produk", hole=0.4, title="Donut chart Produk")
            st.plotly_chart(fig)
        with col22 : 
            fig = px.line(st.session_state.df, x="Tanggal & Waktu", y="Jumlah Produk", color="Kategori", title="Line Chart dengan Banyak Garis Berdasarkan Kategori")
            st.plotly_chart(fig)
        col31, col32 = st.columns(2)
        with col31 :
            fig = px.bar(groupByKategori, x="Kategori", y="Total_omset", title="Bar Plot Berdasarkan Kategori", color="Kategori")
            st.plotly_chart(fig)
        with col32 : 
            fig = px.scatter(st.session_state.df, x="Jumlah Produk", y="Harga Produk", color="Kategori", title="Scatter Plot Berdasarkan Kategori", size_max=10, symbol="Kategori")
            st.plotly_chart(fig)
    #customer segmentation dashboard
    groupByCustomer = prepro.customer_segmentation(groupByCustomer)
    with st.container() :     
        valueCCount= groupByCustomer["cluster"].value_counts().reset_index()
        valueCCount.columns = ["cluster", "count"]
        fig = px.bar(valueCCount, x="cluster", y="count", color="cluster", title="Bar Chart Jumlah Produk per Kategori")
        fig.update_layout(
                    width=800,  
                    height=400  
        )
        st.plotly_chart(fig)
        optionCluster = ["All"] + groupByCustomer["cluster"].unique().tolist()
        option1 = st.selectbox("What cluster ?", optionCluster)
        if option1 == "All" :
            clusteringmask = groupByCustomer.copy()
        else :
            clusteringmask = groupByCustomer[groupByCustomer["cluster"] == option1].copy().reset_index()    
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:    
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=clusteringmask['totSpen'].mean(),
                    title={"text": "Rata-Rata Pengeluaran customer"},
                    number={"font": {"size": 60, "color": "#1F2A44"}}
                ))
                fig.update_layout(
                    width=400,  
                    height=150  
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=clusteringmask['totJum'].mean(),
                    title={"text": "Rata-Rata Jumlah produk"},
                    number={"font": {"size": 60, "color": "#1F2A44"}}
                ))
                fig.update_layout(
                    width=400,  
                    height=150  
                )
                st.plotly_chart(fig, use_container_width=True)
        with col3:
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=clusteringmask['totJenPro'].mean(),
                    title={"text": "Rata-Rata jumlah jenis produk"},
                    number={"font": {"size": 60, "color": "#1F2A44"}}
                ))
                fig.update_layout(
                    width=400,  
                    height=150  
                )
                st.plotly_chart(fig, use_container_width=True)
        with col4:
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=clusteringmask['totKat'].mean(),
                    title={"text": "Rata-Rata jumlah Kategori pesanan"},
                    number={"font": {"size": 60, "color": "#1F2A44"}}
                ))
                fig.update_layout(
                    width=400,  
                    height=150  
                )
                st.plotly_chart(fig, use_container_width=True)
    with st.container(): 
        st.title("🤖 Simple Chatbot with OpenAI")
        client = ollama.Client()
        model  = "granite3-dense:2b"

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_input = st.chat_input("Type your message...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = client.generate(model=model, prompt=user_input)
                    st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})




