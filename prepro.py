import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

def predict_revenue_nbeats(df, prediction_days=30):
    # Salin dan rename kolom supaya sesuai format NeuralForecast: ds (tanggal), y (value)
    df_nbeats = df.reset_index()[['Tanggal & Waktu', 'nominal_transaksi']].copy()
    df_nbeats.rename(columns={'Tanggal & Waktu': 'ds', 'nominal_transaksi': 'y'}, inplace=True)
    
    # Buat model N-BEATS
    model = NBEATS(lags=12, input_size=10, n_epochs=100)
    
    # Inisialisasi NeuralForecast dengan frekuensi harian
    nf = NeuralForecast(models=[model], freq='D')
    
    # Fit model ke data
    nf.fit(df_nbeats)
    
    # Prediksi sebanyak prediction_days ke depan
    forecast = nf.predict(steps=prediction_days)
    
    # Reset index dan rename kolom agar sesuai format
    forecast = forecast.reset_index()
    forecast.rename(columns={'index': 'Tanggal & Waktu', 'NBEATS': 'nominal_transaksi'}, inplace=True)
    forecast.set_index('Tanggal & Waktu', inplace=True)
    
    return forecast

def fix_column_name(df, names): 
    # Rename kolom sesuai mapping yang diberikan
    df = df.rename(columns={v: k for k, v in names.items()})
    return df

def clean_data(df): 
    df["Jumlah Produk"] = pd.to_numeric(df["Jumlah Produk"], errors="coerce")
    df = df[df["Jumlah Produk"] >= 0]
    df["Harga Produk"] = df["Harga Produk"].astype(str).str.replace(",", "", regex=True)
    df["Harga Produk"] = df["Harga Produk"].str.replace(r"[^0-9.]", "", regex=True)
    df["Harga Produk"] = pd.to_numeric(df["Harga Produk"], errors="coerce")
    df["Tanggal & Waktu"] = pd.to_datetime(df["Tanggal & Waktu"], errors="coerce", dayfirst=True)
    df['Total_harga'] = df['Harga Produk'] * df['Jumlah Produk']
    df['Jam'] = df['Tanggal & Waktu'].dt.hour
    return df


def prep_sales(df):
    # Group data per hari untuk summary penjualan
    dff = df.copy()
    dff["Tanggal & Waktu"] = dff["Tanggal & Waktu"].dt.date
    df_grouped = dff.groupby("Tanggal & Waktu", as_index=False).agg(
        banyak_transaksi = ("ID Struk", "nunique"),
        banyak_produk = ("Jumlah Produk", "sum"),
        banyak_jenis_produk = ("Nama Produk", "nunique"), 
        nominal_transaksi = ("Total_harga", "sum")
    )
    return df_grouped

def prep_customer(df):
    # Group data per pelanggan (ID Struk) untuk segmentasi
    df_grouped = df.groupby("ID Struk").agg(
        totSpen=("Total_harga", "sum"), # total spending
        totJum=("Jumlah Produk", "sum"), # total jumlah produk
        totJenPro=("Nama Produk", "nunique"), # total jenis produk
        totKat=("Kategori", "nunique")        # total kategori produk
    ).reset_index()
    return df_grouped

def prep_grouphour(df):
    # Rata-rata jumlah produk berdasarkan jam
    df_grouped = df.groupby("Jam").agg(
        Jumlah_produk  = ("Jumlah Produk", "mean")
    ).reset_index()
    return df_grouped

def prep_groupProduct(df):
    # Ringkasan produk berdasarkan nama produk
    df_grouped = df.groupby("Nama Produk").agg(
        Jumlah_produk  = ("Jumlah Produk", "sum"), 
        Total_omset = ("Total_harga", "sum"),
        Harga_Satuan = ("Harga Produk", "first")
    ).reset_index()
    return df_grouped

def prep_groupKategori(df):
    # Ringkasan produk berdasarkan kategori
    df_grouped = df.groupby("Kategori").agg(
        Jumlah_produk  = ("Jumlah Produk", "sum"), 
        Total_omset = ("Total_harga", "sum"),
        Harga_Satuan = ("Harga Produk", "first")
    ).reset_index()
    return df_grouped

def customer_segmentation(df):
    # Segmentasi customer menggunakan dua model clustering
    scaler = StandardScaler()
    model1 = joblib.load("Segmentasi_pembeli1.pkl")
    model2 = joblib.load("Segmentasi_pembeli22.pkl") 

    df_scaled = scaler.fit_transform(df.drop(columns="ID Struk"))
    labels = model1.fit_predict(df_scaled)

    final_cluster_df = df.copy()
    final_cluster_df['cluster'] = labels

    mask = labels == -1
    X_outlier = df_scaled[mask]
    label = model2.predict(X_outlier)
    label = label + final_cluster_df["cluster"].max() + 1
    final_cluster_df.loc[mask, "cluster"] = label

    return final_cluster_df

def fine_tune_and_predict(df):
    # Pastikan data sudah bersih
    df = clean_data(df)
    # Persiapkan data penjualan harian
    df_grouped = prep_sales(df)
    df_grouped.set_index("Tanggal & Waktu", inplace=True)

    # Panggil fungsi prediksi N-BEATS
    forecast = predict_revenue_nbeats(df_grouped, prediction_days=30)

    # Kalau mau return model dan scaler yang dipakai (dummy untuk saat ini)
    model = "NBEATS model placeholder"
    scaler = MinMaxScaler()

    return forecast['nominal_transaksi'].values, model, scaler
