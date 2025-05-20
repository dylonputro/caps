import pandas as pd 
import numpy as np 
import joblib 
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet, NBeats
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

# Fungsi prediksi dengan model N-BEATS
def fine_tune_and_predict(df, window_size=30, prediction_length=7):
    model = joblib.load("model_nbeats.pkl")
    scaler = joblib.load("scaler.pkl")

    # Pastikan index bertipe datetime
    df.index = pd.to_datetime(df.index)

    # Ambil data nominal_transaksi dan scaling
    data = df["nominal_transaksi"].values.reshape(-1, 1)
    data_scaled = scaler.transform(data)

    # Ambil window terakhir
    input_seq = torch.tensor(data_scaled[-window_size:], dtype=torch.float32).flatten().unsqueeze(0)

    # Prediksi ke depan
    model.eval()
    with torch.no_grad():
        pred_scaled = model(input_seq).numpy().flatten()

    # Balikkan dari scaling
    pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    return pred_actual, model, scaler

def fix_column_name(df, names) : 
    df = df.rename(columns={v: k for k, v in names.items()})
    return df

def clean_data(df) : 
    df["Jumlah Produk"] = pd.to_numeric(df["Jumlah Produk"], errors="coerce")
    df = df[df["Jumlah Produk"] >= 0]
    df["Harga Produk"] = df["Harga Produk"].astype(str).str.replace(",", "", regex=True)
    df["Harga Produk"] = df["Harga Produk"].str.replace(r"[^0-9.]", "", regex=True)
    df["Harga Produk"] = pd.to_numeric(df["Harga Produk"], errors="coerce")
    df["Tanggal & Waktu"] = df["Tanggal & Waktu"].str.replace(r"(\d{2})\.(\d{2})", r"\1:\2", regex=True)
    df["Tanggal & Waktu"] = pd.to_datetime(df["Tanggal & Waktu"], errors="coerce", dayfirst=False)
    df['Total_harga'] = df['Harga Produk']*df['Jumlah Produk']
    df['Jam'] = df['Tanggal & Waktu'].dt.hour
    return df 

def prep_sales(df) : 
    dff = df.copy()
    dff["Tanggal & Waktu"] = dff["Tanggal & Waktu"].dt.date
    df_grouped = dff.groupby("Tanggal & Waktu", as_index=False).agg(
        banyak_transaksi = ("ID Struk", "nunique"),
        banyak_produk = ("Jumlah Produk", "sum"),
        banyak_jenis_produk = ("Nama Produk", "nunique"), 
        nominal_transaksi = ("Total_harga", "sum")
    ).reset_index()
    return df_grouped

def prep_customer(df) :
    df_grouped = df.groupby("ID Struk").agg(
        totSpen=("Total_harga", "sum"), #total spending
        totJum=("Jumlah Produk", "sum"), #total jumlah produk yang dibeli
        totJenPro=("Nama Produk", "nunique"), #total jenis produk yang dibeli
        totKat=("Kategori", "nunique")        #total kategori produk yang dibeli
    ).reset_index()
    return df_grouped

def prep_grouphour(df) : 
    df_grouped = df.groupby("Jam").agg(
        Jumlah_produk  = ("Jumlah Produk", "mean")
    ).reset_index()
    return df_grouped

def prep_groupProduct(df) : 
    df_grouped = df.groupby("Nama Produk").agg(
        Jumlah_produk  = ("Jumlah Produk", "sum"), 
        Total_omset = ("Total_harga", "sum"),
        Harga_Satuan = ("Harga Produk", "first")
    ).reset_index()
    return df_grouped

def prep_groupKategori(df) : 
    df_grouped = df.groupby("Kategori").agg(
        Jumlah_produk  = ("Jumlah Produk", "sum"), 
        Total_omset = ("Total_harga", "sum"),
        Harga_Satuan = ("Harga Produk", "first")
    ).reset_index()
    return df_grouped

 

def customer_segmentation(df) :
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
