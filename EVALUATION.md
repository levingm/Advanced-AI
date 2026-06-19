# Evaluation & Benchmark – Einlagenvolumen-Prognose

## Ziel

Faires Ranking von Modellen (LinReg, FFN, XGBoost, TFT) und Kovariaten-Sets für die **Out-of-Sample-Prognose** des Einlagenvolumens.

## Szenario 2A (realistisch, kein Oracle)

- **Train:** bis `2023-11-30`
- **Test:** 100 Tage ab `2023-12-01`
- **Skalierung:** MinMax nur auf Trainingsdaten gefittet
- **Sequenzmodelle:** Encoder = 100 Tage bis inkl. `TRAINING_ENDE`; keine Zukunfts-Kovariaten
- **TFT:** Kovariaten im Decoder per **naive hold** (letzter bekannter Wert), nicht ex-post Ist-Werte
- **Kein Leakage:** Trainingssequenzen enden, bevor Test-Targets ins Training gelangen würden

## Benchmark ausführen

```bash
python covariate_benchmark.py --modelle linreg ffn --kovariaten-modus presets --simulationen 1 --viz all
```

Ausgabe:
- `covariate_benchmark_results.csv`
- Heatmap, Boxplot, Scatter, Top-5 (mit `--viz all`)

## Visualisierungen interpretieren

| Plot | Frage |
|------|--------|
| Heatmap | Welches Modell bei welchem Kovariaten-Set? |
| Boxplot | Welches Modell ist **stabil** über viele Sets? |
| Scatter | Hilft mehr Kovariaten? |
| Top-5 | Beste Kombination gesamt |

## Forecast-Plot (Option 3)

```bash
python forecast_plot.py --modell linreg --label alle --simulationen 1
python forecast_plot.py --modell ffn --label nur_zinsen --output mein_plot.png
```

## Limitationen (Paper / Seminar)

1. **Ein Testfenster** – Ergebnis hängt vom gewählten Split ab
2. **Data Snooping** – viele Kovariaten-Kombinationen → explorativ, nicht inferenz-statistisch korrigiert
3. **TFT naive hold** – Kovariaten bleiben konstant; realistischer als ex-post, aber keine Szenarien
4. **XGBoost-Skript** heißt `xgboost.py` – kann das Python-Paket shadowen; Benchmark nutzt `importlib` in `common.py`

## Metriken

- **MAE_Volumen** – primär für Modellvergleich (Level)
- **MAE_Diff** – Fehler auf täglicher Änderung
- **R2_Volumen** – kann negativ sein bei schlechten Prognosen
