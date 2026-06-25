#!/usr/bin/env bash
set -euo pipefail

# run_all_experiments.sh
# Vollautomatisiertes Script mit Phasen:
#  - smoke: schneller Import-/Kombi-Check
#  - baseline: Einzelpunkt-Baseline (LinReg, XGBoost)
#  - baseline_with_naive: wie baseline + naive Benchmarks
#  - rolling: Rolling-Origin Evaluation (monatlich)
#  - tuning: Zeitserien-konformes Tuning (XGBoost / FFN) für Kandidaten
#  - ffn: FFN-Run & optionales tuning
#  - tft: TFT-Run (sehr teuer; GPU empfohlen)
#  - plots: einzelne Modellplots (ffn/linreg/tft/xgboost1)
#  - all: führt smoke -> baseline -> rolling -> tuning -> ffn -> tft -> plots nacheinander aus
#
# Anpassbare Variablen:
PYTHON_BIN=${PYTHON_BIN:-python}        # Python-Binary (z.B. python3)
VENV_DIR=${VENV_DIR:-Desktop/Advanced-AI/.venv}
KOV_MODE=${KOV_MODE:-presets}           # presets | singletons | pairs | all | full
SIMS=${SIMS:-3}                         # Ensemble-Simulationen (klein = schneller)
ROLLING_FREQ=${ROLLING_FREQ:-10Y}        # MS = monatlich, QS = quartalsweise
BASELINE_MODELS=${BASELINE_MODELS:-"linreg xgboost"}
EXTENDED_MODELS=${EXTENDED_MODELS:-"naive_rw naive_hold linreg xgboost"}
TFT_MODELS=${TFT_MODELS:-"tft"}
REQUIREMENTS_FILE=${REQUIREMENTS_FILE:-requirements-locked.txt}  # optional
SEED=${SEED:-42}

# Thread-limits (reduziert Hangs)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# Optional: Wenn du GPU nutzen willst, installiere torch manuell per
# https://pytorch.org und setze hier TORCH_INSTALL_CMD entsprechend, z.B.:
# TORCH_INSTALL_CMD="pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio"
TORCH_INSTALL_CMD=${TORCH_INSTALL_CMD:-""}

function help_text() {
  cat <<EOF
Usage: $0 <phase>

Phases:
  smoke                - Smoke test (imports + presets preview)
  baseline             - Quick baseline single-point (linreg, xgboost)
  baseline_with_naive  - Baseline including naive benchmarks
  rolling              - Rolling-Origin evaluation (monthly by default)
  tuning               - Run time-series tuning (GridSearchCV TimeSeriesSplit) for XGBoost/FFN (can be slow)
  ffn                  - Run FFN model plot (uses treesearch_plot -> run_ensemble_volume_paths)
  tft                  - Run TFT (very slow on CPU; use GPU)
  plots                - Produce model-specific plots (ffn/linreg/tft/xgboost1)
  all                  - Run smoke -> baseline_with_naive -> rolling -> tuning -> ffn -> tft -> plots
  help                 - show this message

Examples:
  ./run_all_experiments.sh smoke
  ./run_all_experiments.sh baseline
  ./run_all_experiments.sh all
EOF
}

function create_venv_and_install() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[env] Erstelle venv ${VENV_DIR}..."
    ${PYTHON_BIN} -m venv "${VENV_DIR}"
  fi

  # Activate venv (cross-platform: Unix (bin) or Windows Git Bash (Scripts))
  if [ -f "${VENV_DIR}/bin/activate" ]; then
    # Unix / Linux / macOS
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
  elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
    # Windows (Git Bash) - use the bash-compatible activate
    # shellcheck source=/dev/null
    source "${VENV_DIR}/Scripts/activate"
  else
    echo "Keine Aktivierungsskript gefunden in ${VENV_DIR}. Falls du PowerShell verwendest, aktiviere mit:"
    echo "  .\\${VENV_DIR}\\Scripts\\Activate.ps1  (in PowerShell)"
    echo "Oder aktiviere manuell in Git Bash: source ${VENV_DIR}/Scripts/activate"
    exit 1
  fi

  echo "[env] pip upgraden..."
  python -m pip install --upgrade pip setuptools wheel

  echo "[env] Installiere Basis-Pakete..."
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl tqdm

  if [[ -n "${TORCH_INSTALL_CMD}" ]]; then
    echo "[env] Benutzerdefinierter Torch-Install: ${TORCH_INSTALL_CMD}"
    eval "${TORCH_INSTALL_CMD}"
  else
    echo "[env] Installiere CPU-only Torch + Lightning + pytorch-forecasting (kann groß sein)..."
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
  fi

  pip install lightning pytorch-forecasting

  echo "[env] (Optional) Freeze requirements to ${REQUIREMENTS_FILE}"
  pip freeze > "${REQUIREMENTS_FILE}" || true

  echo "[env] Fertig. Venv befindet sich in: ${VENV_DIR}"
  echo "[env] Um die venv in Git Bash manuell zu aktivieren: source ${VENV_DIR}/Scripts/activate"
}

function smoke_test() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[smoke] Schnelltest: Import + Presets anzeigen"
  python - <<PY
import common as tc
print("SEED:", tc.SEED)
print("Kovariaten Presets (erste 6):", [lbl for _,lbl in tc.generate_covariate_combinations('presets')][:6])
PY
}

function baseline() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[baseline] Einzelpunkt-Baseline: MODELS=${BASELINE_MODELS}, KOV_MODE=${KOV_MODE}, SIMS=${SIMS}"
  python covariate_benchmark.py --modelle ${BASELINE_MODELS} --kovariaten-modus=${KOV_MODE} --simulationen ${SIMS} --viz=heatmap --plot
}

function baseline_with_naive() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[baseline_with_naive] Einzelpunkt inkl. naive Benchmarks: MODELS=${EXTENDED_MODELS}, KOV_MODE=${KOV_MODE}, SIMS=${SIMS}"
  python covariate_benchmark.py --modelle ${EXTENDED_MODELS} --kovariaten-modus=${KOV_MODE} --simulationen ${SIMS} --viz=all --plot
}

function rolling_eval() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[rolling] Rolling-Origin evaluation: MODELS=${BASELINE_MODELS}, KOV_MODE=${KOV_MODE}, SIMS=${SIMS}, FREQ=${ROLLING_FREQ}"
  python covariate_benchmark.py --modelle ${BASELINE_MODELS} --kovariaten-modus=${KOV_MODE} --rolling --rolling-freq=${ROLLING_FREQ} --simulationen ${SIMS}
}

function tuning() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[tuning] Zeitserien-konformes Tuning (XGBoost / FFN) für alle Kovariaten (kann lange dauern)..."
  python - <<PY
import common as tc
print("Starte Tuning: SEED=", tc.SEED)
# Beispiel: Tuning für XGBoost über alle Kovariaten (Achtung: Rechenintensiv)
mean, ki90, ki98, paths, df = tc.run_ensemble_volume_paths('xgboost', tc.ALL_COVARIATES, simulations=3, do_tune=True, training_end=tc.TRAINING_ENDE)
print("XGBoost tuning finished. Forecast shape:", mean.shape)
# Tuning für FFN (optional)
mean_f, ki90_f, ki98_f, paths_f, df_f = tc.run_ensemble_volume_paths('ffn', tc.ALL_COVARIATES, simulations=3, do_tune=True, training_end=tc.TRAINING_ENDE)
print("FFN tuning finished. Forecast shape:", mean_f.shape)
PY
}

function run_ffn_plot() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[ffn] Erzeuge FFN Plot (treesearch_plot -> ffn.py)"
  python ffn.py
}

function run_linreg_plot() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[linreg] Erzeuge LinReg Plot (treesearch_plot -> linreg.py)"
  python linreg.py
}

function run_tft_plot() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[tft] Erzeuge TFT Plot (sehr teuer auf CPU!):"
  python tft.py
}

function run_xgboost1_plot() {
  source "${VENV_DIR}/Scripts/activate"
  echo "[xgboost1] Erzeuge XGBoost Beispielplot (nur Makro)"
  python xgboost1.py
}

function all_phases() {
  create_venv_and_install
  smoke_test
  baseline_with_naive
  rolling_eval
  tuning
  run_ffn_plot
  run_linreg_plot
  # TFT nur optional: dekommentieren falls verfügbar / nötig
  echo "TFT-Plot wird jetzt NICHT automatisch ausgeführt. Rufe './run_all_experiments.sh tft' separat auf, wenn du GPU/Resourcen hast."
  # run_tft_plot
  run_xgboost1_plot
  echo "[all] Alle Phasen abgeschlossen."
}

# Main
if [[ $# -lt 1 ]]; then
  help_text
  exit 1
fi

PHASE="$1"

case "${PHASE}" in
  help) help_text ;;
  create_env) create_venv_and_install ;;
  smoke) create_venv_and_install; smoke_test ;;
  baseline) create_venv_and_install; baseline ;;
  baseline_with_naive) create_venv_and_install; baseline_with_naive ;;
  rolling) create_venv_and_install; rolling_eval ;;
  tuning) create_venv_and_install; tuning ;;
  ffn) create_venv_and_install; run_ffn_plot ;;
  linreg) create_venv_and_install; run_linreg_plot ;;
  tft) create_venv_and_install; run_tft_plot ;;
  xgboost1) create_venv_and_install; run_xgboost1_plot ;;
  plots) create_venv_and_install; run_ffn_plot; run_linreg_plot; run_xgboost1_plot ;;
  all) create_venv_and_install; all_phases ;;
  *) echo "Unbekannte Phase: ${PHASE}"; help_text; exit 2 ;;
esac

echo "[done] Phase '${PHASE}' beendet."