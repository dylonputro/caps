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

def fine_tune_and_predict(data):
    data['time_idx'] = np.arange(len(data))
    scaler = MinMaxScaler()
    data['nominal_transaksi'] = scaler.fit_transform(data[['nominal_transaksi']])
    train_data = data[:int(len(data) * 0.8)]
    val_data = data[int(len(data) * 0.8):]
    max_encoder_length = 20 
    max_prediction_length = 7 
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="nominal_transaksi",
        group_ids=["time_idx"],  # The group ID for time series
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],  # No static categorical features
        static_reals=[],  # No static real features
        time_varying_known_categoricals=[],  # No known categorical features
        time_varying_known_reals=["time_idx"],  # The time index is known
        time_varying_unknown_categoricals=[],  # No unknown categorical features
        time_varying_unknown_reals=["nominal_transaksi"],  # The value is unknown and to be predicted
        target_normalizer=GroupNormalizer(groups=["time_idx"], transformation="softplus"),
    )
    trainer = Trainer(max_epochs=10, gpus=1)
    model = NBeats.from_dataset(training, learning_rate=0.001, hidden_size=64, batch_size=64)
    trainer.fit(model, train_dataloader=DataLoader(training, batch_size=64, shuffle=True))
    input_data = data['nominal_transaksi'][-30:]
    input_tensor = torch.tensor(input_data.values, dtype=torch.float32).unsqueeze(0)
    predictions = model(input_tensor)
    predicted_values = predictions.squeeze().detach().numpy()
    predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()

    return predicted_values, model, scaler

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













