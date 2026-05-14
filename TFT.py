import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss

# Konfiguration
Daten = "Masterarbeit Werte pchip.xlsx"
Variablen = ["Einlagevolumen", "€STR", "Einlagezinssatz"]

Gedächtnis = 100
Prognosehorizont = 100
Batch = 32
Epochs = 20
Lernrate = 1e-3
Seed = 42

TrainingEnde = "2025-03-15"
InputAnfang = "2025-03-15"

pl.seed_everything(Seed)

# Daten laden
df = pd.read_excel(Daten).iloc[1:8280, 0:4].rename(columns={0:"Datum",1:"Einlagevolumen",2:"€STR",3:"Einlagezinssatz"})
df["Datum"] = pd.to_datetime(df["Datum"])

# Pytorch Forecasting benötigt einen Integer-Zeitindex und eine Gruppen-ID
df["time_idx"] = (df["Datum"] - df["Datum"].min()).dt.days
df["group"] = "A" # Da wir nur eine Zeitreihe haben

# Split
training_cutoff = df[df["Datum"] <= TrainingEnde]["time_idx"].max()

# DataSet Erstellung
max_prediction_length = Prognosehorizont
max_encoder_length = Gedächtnis

training_data = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Einlagevolumen",
    group_ids=["group"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group"],
    time_varying_known_reals=["time_idx", "€STR", "Einlagezinssatz"], 
    time_varying_unknown_reals=["Einlagevolumen"],
    add_relative_time_idx=True,
    add_target_scales=True
)

validation_data = TimeSeriesDataSet.from_dataset(training_data, df, predict=True, stop_randomization=True)

train_dataloader = training_data.to_dataloader(train=True, batch_size=Batch, num_workers=0)
val_dataloader = validation_data.to_dataloader(train=False, batch_size=Batch, num_workers=0)

# TFT Modell Initialisierung
tft = TemporalFusionTransformer.from_dataset(
    training_data,
    learning_rate=Lernrate,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(), 
    optimizer="adam"
)

# Training
trainer = pl.Trainer(
    max_epochs=Epochs,
    accelerator="auto",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,
)

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Vorhersage
best_tft = tft
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

# Plotting
best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True)
plt.show()
