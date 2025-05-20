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

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

def fine_tune_and_predict(data):
    # Strip any leading/trailing spaces from column names
    data.columns = data.columns.str.strip()
    
    data['Tanggal & Waktu'] = pd.to_datetime(data['Tanggal & Waktu'], format='%d-%m-%Y %H:%M:%S')

    # Create 'nominal_transaksi' by multiplying 'Jumlah Produk' and 'Harga Produk'
    # First, clean 'Harga Produk' to remove commas and convert to numeric
    data['Harga Produk'] = data['Harga Produk'].str.replace(",", "").astype(float)
    data['nominal_transaksi'] = data['Jumlah Produk'] * data['Harga Produk']

    # Sort by 'Tanggal & Waktu'
    data = data.sort_values('Tanggal & Waktu')

    # Extract date part for grouping
    data['tanggal'] = data['Tanggal & Waktu'].dt.date

    # Aggregate if there are duplicates per day
    data = data.groupby('tanggal').agg({'nominal_transaksi': 'sum'}).reset_index()

    # Normalize
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(data[['nominal_transaksi']])
    data['scaled'] = scaled_values

    # Model ARIMA (use scaled data)
    train = data['scaled']

    # Fit ARIMA model — (p,d,q) can be tuned, here using a simple (1,1,1)
    model = ARIMA(train, order=(1, 1, 1))
    fitted_model = model.fit()

    # Predict 7 days ahead
    forecast = fitted_model.forecast(steps=7)

    # Inverse scale the predicted values
    predicted_values = scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten()

    return predicted_values, fitted_model, scaler




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













