# -*- coding: utf-8 -*-
"""
TFM: Sistema inteligente de análisis, clasificación y regresión de tendencias/precios en ETFs
Autor: David Piñuel Bosque

VERSIÓN CORREGIDA Y AUDITABLE
==============================
ETFs de trabajo:
    - SXR8: iShares Core S&P 500 UCITS ETF
    - SXRV: iShares NASDAQ 100 UCITS ETF

Entrada esperada:
    data/raw/sxr8.csv
    data/raw/sxrv.csv

Esta versión incorpora:
    1. Parser de fechas robusto. NO usa dayfirst=True indiscriminadamente.
    2. Preprocesamiento y auditoría de datos antes de entrenar.
    3. Eliminación de duplicados y registros OHLC imposibles.
    4. CSV limpios persistentes en data/processed/.
    5. Informe explícito de módulos activos/inactivos.
    6. Logistic Regression, Random Forest y XGBoost si está disponible.
    7. LSTM opcional con TensorFlow/Keras si está disponible.
    8. Validación temporal con TimeSeriesSplit y gap para evitar leakage.
    9. Holdout cronológico final independiente.
   10. Ensemble cuyos pesos se calculan SIN utilizar el holdout final.
   11. Horizontes de 1, 5, 10, 15, 20, 30, 45 y 60 sesiones.
   12. Backtesting contra Buy & Hold.
   13. Noticias recientes mediante Google News RSS + FinBERT, opcionales.
   14. Sentimiento histórico SOLO entra en el modelo si existe un CSV histórico
       alineado por fecha. Si no existe, las noticias actuales se muestran como
       contexto, pero NO se inyectan en un modelo entrenado sin sentimiento.
   15. Registro persistente de predicciones y evaluación posterior.
   16. Toda la salida de consola se copia además a un TXT.
   17. Histórico largo de noticias mediante Alpha Vantage NEWS_SENTIMENT.
   18. FinBERT aplicado tanto al histórico como a noticias recientes.
   19. Variables externas desde FRED: S&P500, Nasdaq100, VIX, EUR/USD,
       Treasury 10Y/2Y y Fed Funds.
   20. Lag de una sesión en variables externas para reducir look-ahead bias.
   21. Histórico Alpha Vantage multifamilia con score de relevancia.
   22. Validación no solapada por fases para horizontes largos.
   23. Bootstrap por bloques para intervalo de confianza del AUC.
   24. Reproducibilidad reforzada: semillas globales, TensorFlow determinista
       y modelos clásicos en un solo hilo.
   25. Segunda fuente histórica independiente: The Guardian Open Platform.
   26. Fusión y deduplicación Alpha Vantage + Guardian antes de FinBERT diario.

IMPORTANTE METODOLÓGICAMENTE
============================
- Una fecha mal interpretada puede reordenar los precios y crear retornos
  artificiales. Por eso este programa detecta el formato de fecha de forma
  explícita y falla si encuentra un formato ambiguo que no puede resolver.
- Los retornos extremos se AUDITAN y se guardan en un fichero de anomalías.
  No se eliminan automáticamente salvo errores estructurales, para no borrar
  caídas/subidas reales del mercado sin justificación.
- El modelo no "aprende" de una predicción inmediatamente. En cada nueva
  ejecución se reentrena con datos históricos actualizados y se evalúan las
  predicciones pasadas cuando ya existe información futura suficiente.
- Las predicciones son experimentales y no constituyen asesoramiento financiero.

Dependencias mínimas:
    pip install pandas numpy matplotlib scikit-learn joblib

Opcionales:
    pip install xgboost
    pip install tensorflow
    pip install feedparser transformers torch sentencepiece requests shap
    # The Guardian Open Platform requiere GUARDIAN_API_KEY (no librería extra).
"""

from __future__ import annotations

import os
import sys
import random

# -----------------------------------------------------------------------------
# Reproducibilidad: estas variables deben existir antes de importar TensorFlow.
# PYTHONHASHSEED solo es completamente efectivo si Python/Spyder se inicia con
# ese valor ya definido; aun así lo fijamos aquí y además reseedamos todas las
# librerías durante el entrenamiento.
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import re
import json
import warnings
import traceback
import platform
import hashlib
import time
from html import unescape
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except Exception as exc:
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = repr(exc)

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Dependencias opcionales: guardamos también el motivo del fallo de importación.
# -----------------------------------------------------------------------------

XGBOOST_AVAILABLE = False
XGBOOST_IMPORT_ERROR = ""
try:
    import xgboost
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception as exc:
    XGBOOST_IMPORT_ERROR = repr(exc)

TENSORFLOW_AVAILABLE = False
TENSORFLOW_IMPORT_ERROR = ""
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
    TENSORFLOW_AVAILABLE = True
except Exception as exc:
    TENSORFLOW_IMPORT_ERROR = repr(exc)

FEEDPARSER_AVAILABLE = False
FEEDPARSER_IMPORT_ERROR = ""
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except Exception as exc:
    FEEDPARSER_IMPORT_ERROR = repr(exc)

TRANSFORMERS_AVAILABLE = False
TRANSFORMERS_IMPORT_ERROR = ""
try:
    import transformers
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception as exc:
    TRANSFORMERS_IMPORT_ERROR = repr(exc)

TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = ""
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as exc:
    TORCH_IMPORT_ERROR = repr(exc)

warnings.filterwarnings("ignore")


class Reproducibility:
    """Centraliza las semillas y opciones deterministas del experimento."""

    @staticmethod
    def stable_seed(base_seed: int, *parts: Any) -> int:
        raw = "|".join([str(base_seed)] + [str(p) for p in parts])
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16) % (2**31 - 1)

    @staticmethod
    def apply(seed: int, deterministic_tensorflow: bool = True):
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if TORCH_AVAILABLE:
            try:
                torch.manual_seed(seed)
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                if hasattr(torch, "use_deterministic_algorithms"):
                    torch.use_deterministic_algorithms(True, warn_only=True)
                if hasattr(torch, "set_num_threads"):
                    torch.set_num_threads(1)
            except Exception:
                pass

        if TENSORFLOW_AVAILABLE:
            try:
                tf.keras.utils.set_random_seed(seed)
            except Exception:
                try:
                    tf.random.set_seed(seed)
                except Exception:
                    pass

            if deterministic_tensorflow:
                try:
                    tf.config.experimental.enable_op_determinism()
                except Exception:
                    pass

    @staticmethod
    def configure_runtime(seed: int, deterministic_tensorflow: bool = True):
        Reproducibility.apply(seed, deterministic_tensorflow)

        if TENSORFLOW_AVAILABLE:
            # Un solo hilo reduce pequeñas variaciones numéricas entre ejecuciones.
            try:
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
            except Exception:
                # TensorFlow puede haber inicializado ya el runtime. Las semillas y
                # enable_op_determinism siguen siendo efectivas.
                pass


Reproducibility.apply(42, deterministic_tensorflow=True)


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

@dataclass
class Config:
    tickers: List[str] = field(default_factory=lambda: ["SXR8", "SXRV"])

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    historical_sentiment_dir: str = "data/news_sentiment"
    external_data_dir: str = "data/external"
    output_dir: str = "resultados_tfm"

    start_date: str = "2015-01-01"
    end_date: Optional[str] = None
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 15, 20, 30, 45, 60])

    # Si el formato del CSV no puede detectarse con seguridad, especificarlo aquí.
    # Ejemplos:
    #   {"SXR8": "%d/%m/%Y", "SXRV": "%Y-%m-%d"}
    date_format_overrides: Dict[str, str] = field(default_factory=dict)

    cv_splits: int = 5
    test_size_ratio: float = 0.20

    # Reproducibilidad científica.
    random_seed: int = 42
    deterministic_tensorflow: bool = True
    single_thread_models: bool = True

    initial_capital: float = 10_000.0
    transaction_cost: float = 0.001
    buy_threshold: float = 0.60
    sell_threshold: float = 0.40
    allow_short: bool = False

    # Auditoría de datos. Los retornos sospechosos se marcan, no se borran.
    suspicious_daily_return: float = 0.20
    hard_daily_return_limit: float = 2.00
    max_calendar_gap_days: int = 14

    # Deep Learning
    use_lstm: bool = True
    require_xgboost: bool = True
    require_tensorflow: bool = True
    lstm_lookback: int = 30
    lstm_epochs: int = 35
    lstm_batch_size: int = 32

    # Noticias / NLP
    use_news: bool = True
    require_feedparser: bool = True
    require_transformers: bool = True
    require_torch: bool = True
    max_news_items: int = 100
    news_when_days: int = 30
    finbert_model: str = "ProsusAI/finbert"
    news_cache_dir: str = "data/news_cache"

    # El sentimiento solo se introduce en entrenamiento si existe un histórico
    # suficiente y razonablemente cubierto. Esto evita entrenar 10 años con 0
    # y usar sentimiento real solo al final.
    min_sentiment_days_for_model: int = 500
    min_sentiment_coverage_for_model: float = 0.35

    # Ventanas utilizadas para construir variables NLP temporales.
    sentiment_windows: List[int] = field(
        default_factory=lambda: [1, 5, 10, 15, 20, 30, 45, 60]
    )

    # Una noticia publicada el día t se utiliza desde t+1 para ser
    # conservadores respecto a la hora exacta de publicación/cierre.
    sentiment_lag_days: int = 1

    # -----------------------------------------------------------------
    # Histórico largo de noticias: Alpha Vantage
    # -----------------------------------------------------------------
    use_alpha_vantage_history: bool = True
    require_alpha_vantage_history: bool = True
    alpha_vantage_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY")
    )
    alpha_vantage_history_dir: str = "data/news_history_alpha"
    alpha_vantage_history_start: str = "2018-01-01"

    # Backfill adaptativo. Las consultas amplias empiezan en 180 días;
    # si Alpha Vantage devuelve demasiados resultados, el bloque se divide
    # automáticamente hasta llegar a ventanas más pequeñas.
    alpha_vantage_initial_chunk_days: int = 180
    alpha_vantage_min_chunk_days: int = 14
    alpha_vantage_limit: int = 1000
    alpha_vantage_truncation_ratio: float = 0.95

    # Cuota segura para repartir las llamadas entre SXR8 y SXRV.
    # Con una cuenta de mayor cuota puedes aumentar estos valores.
    alpha_vantage_max_total_calls_per_run: int = 20
    alpha_vantage_max_calls_per_ticker_per_run: int = 10
    alpha_vantage_pause_seconds: float = 1.5
    alpha_vantage_timeout_seconds: int = 60

    # Filtrado de relevancia de noticias.
    alpha_vantage_min_relevance: float = 0.22

    # -----------------------------------------------------------------
    # Segunda fuente histórica independiente: The Guardian Open Platform
    # -----------------------------------------------------------------
    use_guardian_history: bool = True
    require_guardian_history: bool = True
    guardian_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GUARDIAN_API_KEY")
    )
    guardian_history_dir: str = "data/news_history_guardian"
    guardian_history_start: str = "2015-01-01"
    guardian_initial_chunk_days: int = 180
    guardian_min_chunk_days: int = 14
    guardian_page_size: int = 50
    guardian_max_pages_per_window: int = 10
    guardian_max_total_calls_per_run: int = 20
    guardian_max_calls_per_ticker_per_run: int = 10
    guardian_pause_seconds: float = 0.75
    guardian_timeout_seconds: int = 60
    guardian_min_relevance: float = 0.20

    # -----------------------------------------------------------------
    # Validación robusta para targets solapados
    # -----------------------------------------------------------------
    validate_overlapping_targets: bool = True
    overlap_bootstrap_iterations: int = 300
    overlap_random_seed: int = 42

    # -----------------------------------------------------------------
    # Histórico NLP real
    # -----------------------------------------------------------------
    # Google News RSS no ofrece un archivo histórico largo. Para construir
    # una serie histórica reproducible usamos GDELT DOC 2.0 TimelineTone
    # (cobertura desde 2017) y mantenemos FinBERT para las noticias recientes.
    # El tono GDELT se normaliza a [-1, 1], de forma compatible con FinBERT.
    use_gdelt_history: bool = False
    require_gdelt_history: bool = False
    gdelt_history_start: str = "2017-01-01"
    gdelt_timeout_seconds: int = 90
    gdelt_retries: int = 3

    # GDELT aplica rate limiting. No queremos bloquear Spyder durante
    # minutos si devuelve HTTP 429. Se respeta Retry-After hasta este máximo.
    gdelt_max_retry_wait_seconds: int = 20
    gdelt_request_pause_seconds: float = 2.0

    # -----------------------------------------------------------------
    # Variables externas de mercado y macroeconomía - FRED
    # -----------------------------------------------------------------
    use_external_features: bool = True
    require_external_features: bool = True

    # Desplazamos las variables externas una sesión para evitar dudas sobre
    # la hora exacta de publicación y prevenir look-ahead bias.
    external_feature_lag_sessions: int = 1

    # Series FRED utilizadas:
    # SP500      -> S&P 500
    # NASDAQ100  -> Nasdaq 100
    # VIXCLS     -> VIX
    # DEXUSEU    -> USD por EUR (EUR/USD)
    # DGS10      -> Treasury USA 10 años
    # DGS2       -> Treasury USA 2 años
    # DFF        -> Effective Federal Funds Rate
    fred_series: Dict[str, str] = field(default_factory=lambda: {
        "sp500": "SP500",
        "nasdaq100": "NASDAQ100",
        "vix": "VIXCLS",
        "eurusd": "DEXUSEU",
        "us10y": "DGS10",
        "us2y": "DGS2",
        "fedfunds": "DFF",
    })

    # Estudio de ablación NLP: CON sentimiento vs SIN sentimiento.
    compare_sentiment_models: bool = True
    compare_lstm_sentiment: bool = True

    # Explicabilidad.
    use_shap: bool = True
    require_shap: bool = True
    shap_max_samples: int = 500
    shap_top_features: int = 25

    # -----------------------------------------------------------------
    # FASE FINAL EXPERIMENTAL
    # -----------------------------------------------------------------
    run_final_experiments: bool = True
    final_experiment_horizons: List[int] = field(
        default_factory=lambda: [1, 5, 10, 15, 20, 30, 45, 60]
    )

    # Ablación incremental:
    #   TECHNICAL -> TECHNICAL+MACRO -> TECHNICAL+MACRO+NLP
    run_feature_group_ablation: bool = True

    # Robustez: retirar niveles absolutos de tipos para comprobar que
    # Fed Funds / Treasury no son meros proxies del régimen temporal.
    run_rate_level_robustness: bool = True

    # Estrategia tradicional adicional a Buy & Hold.
    run_ma50_ma200_baseline: bool = True

    # -----------------------------------------------------------------
    # REGRESIÓN DE PRECIO FUTURO
    # -----------------------------------------------------------------
    use_price_regression: bool = True
    regression_target: str = "log_return"
    regression_models: List[str] = field(
        default_factory=lambda: ["Ridge", "RandomForestRegressor", "XGBRegressor"]
    )
    regression_interval_low: float = 0.10
    regression_interval_high: float = 0.90
    regression_min_train_rows: int = 400

    # Umbral mínimo esperado sobre costes para considerar que el objetivo
    # de precio respalda una señal alcista. NO es asesoramiento financiero.
    regression_min_expected_edge: float = 0.003

    # Congelación del dataset experimental. En la primera ejecución guarda
    # una fotografía reproducible; posteriormente puede activarse reuse.
    freeze_experimental_dataset: bool = True
    reuse_frozen_dataset: bool = True
    frozen_data_dir: str = "data/frozen_experiment"

    prediction_log: str = "resultados_tfm/prediction_history.csv"

    def __post_init__(self):
        if self.end_date is None:
            self.end_date = datetime.today().strftime("%Y-%m-%d")

        for directory in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.historical_sentiment_dir,
            self.news_cache_dir,
            self.alpha_vantage_history_dir,
            self.guardian_history_dir,
            self.external_data_dir,
            self.frozen_data_dir,
            self.output_dir,
            os.path.join(self.output_dir, "logs"),
            os.path.join(self.output_dir, "preprocessing"),
        ]:
            os.makedirs(directory, exist_ok=True)

        pred_dir = os.path.dirname(self.prediction_log)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)


# =============================================================================
# 2. LOG COMPLETO DE CONSOLA
# =============================================================================

class TeeOutput:
    def __init__(self, console_stream, file_stream):
        self.console_stream = console_stream
        self.file_stream = file_stream

    def write(self, message):
        self.console_stream.write(message)
        self.file_stream.write(message)
        return len(message)

    def flush(self):
        self.console_stream.flush()
        self.file_stream.flush()

    def isatty(self):
        try:
            return self.console_stream.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        return getattr(self.console_stream, "encoding", "utf-8")


class ConsoleLogger:
    def __init__(self, output_dir: str):
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(logs_dir, f"ejecucion_tfm_{timestamp}.txt")
        self.original_stdout = None
        self.original_stderr = None
        self.log_file = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = TeeOutput(self.original_stdout, self.log_file)
        sys.stderr = TeeOutput(self.original_stderr, self.log_file)

        print("=" * 118)
        print("REGISTRO COMPLETO DE EJECUCIÓN - TFM ETFs")
        print("=" * 118)
        print(f"Inicio:                  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python:                  {sys.version.split()[0]}")
        print(f"Sistema:                 {platform.platform()}")
        print(f"Directorio de trabajo:   {os.getcwd()}")
        print(f"Archivo de log:          {os.path.abspath(self.log_path)}")
        print("=" * 118)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            print("\n" + "!" * 118)
            print("LA EJECUCIÓN HA FINALIZADO CON UN ERROR")
            print("!" * 118)
            traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        else:
            print("\n" + "=" * 118)
            print("EJECUCIÓN FINALIZADA CORRECTAMENTE")
            print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 118)

        print(f"\nSalida completa guardada en: {os.path.abspath(self.log_path)}")

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            if self.log_file is not None:
                self.log_file.flush()
                self.log_file.close()
        return False


# =============================================================================
# 3. ESTADO REAL DE LOS COMPONENTES
# =============================================================================

class ComponentStatus:
    @staticmethod
    def _version(module_name: str) -> str:
        try:
            module = sys.modules.get(module_name)
            if module is None:
                module = __import__(module_name)
            return getattr(module, "__version__", "desconocida")
        except Exception:
            return "desconocida"

    @staticmethod
    def ensure_required(config: Config):
        """Detiene la ejecución si faltan motores que el experimento exige."""
        missing = []

        if config.require_xgboost and not XGBOOST_AVAILABLE:
            missing.append(
                "XGBoost no está disponible. Instala con: pip install xgboost"
                f"\n  Error de importación: {XGBOOST_IMPORT_ERROR}"
            )

        if config.use_lstm and config.require_tensorflow and not TENSORFLOW_AVAILABLE:
            missing.append(
                "TensorFlow no está disponible y la LSTM está marcada como obligatoria. "
                "Se recomienda Python 3.11 y: pip install tensorflow"
                f"\n  Error de importación: {TENSORFLOW_IMPORT_ERROR}"
            )

        if config.use_news and config.require_feedparser and not FEEDPARSER_AVAILABLE:
            missing.append(
                "feedparser no está disponible y Google News RSS está marcado como obligatorio. "
                "Instala con: pip install feedparser"
                f"\n  Error de importación: {FEEDPARSER_IMPORT_ERROR}"
            )

        if config.use_news and config.require_transformers and not TRANSFORMERS_AVAILABLE:
            missing.append(
                "transformers no está disponible y FinBERT está marcado como obligatorio. "
                "Instala con: pip install transformers"
                f"\n  Error de importación: {TRANSFORMERS_IMPORT_ERROR}"
            )

        if config.use_news and config.require_torch and not TORCH_AVAILABLE:
            missing.append(
                "PyTorch no está disponible y FinBERT lo necesita. "
                "Instala con: pip install torch"
                f"\n  Error de importación: {TORCH_IMPORT_ERROR}"
            )

        if config.use_shap and config.require_shap and not SHAP_AVAILABLE:
            missing.append(
                "SHAP no está disponible. Instala con: pip install shap"
                f"\n  Error de importación: {SHAP_IMPORT_ERROR}"
            )

        if (
            config.use_alpha_vantage_history
            and config.require_alpha_vantage_history
            and not config.alpha_vantage_api_key
        ):
            cache_files = [
                os.path.join(
                    config.alpha_vantage_history_dir,
                    f"{ticker.lower()}_articles.csv",
                )
                for ticker in config.tickers
            ]
            if not all(os.path.exists(p) for p in cache_files):
                missing.append(
                    "Falta la variable de entorno ALPHAVANTAGE_API_KEY "
                    "y todavía no existe un histórico Alpha Vantage completo."
                )

        if (
            config.use_guardian_history
            and config.require_guardian_history
            and not config.guardian_api_key
        ):
            cache_files = [
                os.path.join(
                    config.guardian_history_dir,
                    f"{ticker.lower()}_articles.csv",
                )
                for ticker in config.tickers
            ]
            if not all(os.path.exists(p) for p in cache_files):
                missing.append(
                    "Falta GUARDIAN_API_KEY y todavía no existe un histórico "
                    "local de The Guardian. Registra una clave en The Guardian "
                    "Open Platform y define la variable de entorno GUARDIAN_API_KEY."
                )

        if missing:
            print("\n" + "!" * 118)
            print("FALTAN DEPENDENCIAS OBLIGATORIAS PARA ESTE EXPERIMENTO")
            print("!" * 118)
            for item in missing:
                print(f"- {item}")
            print("!" * 118)
            raise RuntimeError(
                "No se inicia el entrenamiento para evitar ejecutar un ensemble incompleto."
            )

    @staticmethod
    def print_status(config: Config):
        print("\n" + "=" * 118)
        print("COMPONENTES DISPONIBLES")
        print("=" * 118)
        print(f"pandas                    OK   versión {pd.__version__}")
        print(f"numpy                     OK   versión {np.__version__}")
        print("scikit-learn              OK")
        print("Logistic Regression       OK")
        print("Random Forest             OK")
        print(f"Reproducibilidad           OK   seed={config.random_seed}, TF determinista={config.deterministic_tensorflow}")

        if XGBOOST_AVAILABLE:
            print(f"XGBoost                    OK   versión {ComponentStatus._version('xgboost')}")
        else:
            print(f"XGBoost                    NO   {XGBOOST_IMPORT_ERROR}")
            print("  -> Instalar con: pip install xgboost")

        if config.use_lstm:
            if TENSORFLOW_AVAILABLE:
                print(f"TensorFlow / LSTM          OK   versión {ComponentStatus._version('tensorflow')}")
            else:
                print(f"TensorFlow / LSTM          NO   {TENSORFLOW_IMPORT_ERROR}")
                print("  -> Recomendado: entorno Conda con Python 3.11 + pip install tensorflow")
        else:
            print("TensorFlow / LSTM          DESACTIVADO POR CONFIGURACIÓN")

        if config.use_news:
            if FEEDPARSER_AVAILABLE:
                print("Google News RSS/feedparser OK")
            else:
                print(f"Google News RSS/feedparser NO   {FEEDPARSER_IMPORT_ERROR}")
                print("  -> Instalar con: pip install feedparser")

            if TRANSFORMERS_AVAILABLE:
                print(f"Transformers               OK   versión {ComponentStatus._version('transformers')}")
            else:
                print(f"Transformers               NO   {TRANSFORMERS_IMPORT_ERROR}")
                print("  -> Instalar con: pip install transformers")

            if TORCH_AVAILABLE:
                print(f"PyTorch                    OK   versión {ComponentStatus._version('torch')}")
            else:
                print(f"PyTorch                    NO   {TORCH_IMPORT_ERROR}")
                print("  -> Instalar con: pip install torch")
        else:
            print("Noticias / FinBERT         DESACTIVADO POR CONFIGURACIÓN")

        if config.use_shap:
            if SHAP_AVAILABLE:
                print(f"SHAP                       OK   versión {ComponentStatus._version('shap')}")
            else:
                print(f"SHAP                       NO   {SHAP_IMPORT_ERROR}")
                print("  -> Instalar con: pip install shap")
        else:
            print("SHAP                       DESACTIVADO POR CONFIGURACIÓN")

        print("=" * 118)


# =============================================================================
# 4. PARSER DE FECHAS ROBUSTO
# =============================================================================

class RobustDateParser:
    """Detecta el formato de fecha sin aplicar dayfirst=True de forma global."""

    FORMATS = {
        "iso_dash": (re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), "%Y-%m-%d"),
        "iso_slash": (re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"), "%Y/%m/%d"),
        "iso_dot": (re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}$"), "%Y.%m.%d"),
        "euro_slash": (re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"), None),
        "euro_dash": (re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"), None),
        "euro_dot": (re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"), None),
    }

    @staticmethod
    def parse(series: pd.Series, ticker: str, override_format: Optional[str] = None) -> Tuple[pd.Series, str]:
        values = series.astype(str).str.strip()
        values = values.replace({"nan": np.nan, "NaT": np.nan, "None": np.nan, "": np.nan})
        non_null = values.dropna()

        if non_null.empty:
            raise ValueError(f"{ticker}: la columna de fechas está vacía.")

        if override_format:
            parsed = pd.to_datetime(values, format=override_format, errors="coerce")
            return parsed, f"override:{override_format}"

        sample = non_null.head(min(len(non_null), 300))

        # ISO no es ambiguo.
        for name in ["iso_dash", "iso_slash", "iso_dot"]:
            regex, fmt = RobustDateParser.FORMATS[name]
            if sample.map(lambda x: bool(regex.match(str(x)))).all():
                parsed = pd.to_datetime(values, format=fmt, errors="coerce")
                return parsed, fmt

        # Formatos con año al final: determinamos D/M/Y vs M/D/Y usando valores > 12.
        for name in ["euro_slash", "euro_dash", "euro_dot"]:
            regex, _ = RobustDateParser.FORMATS[name]
            if sample.map(lambda x: bool(regex.match(str(x)))).all():
                separator = {"euro_slash": "/", "euro_dash": "-", "euro_dot": "."}[name]
                parts = sample.str.split(separator, expand=True)
                first = pd.to_numeric(parts[0], errors="coerce")
                second = pd.to_numeric(parts[1], errors="coerce")

                first_gt_12 = bool((first > 12).any())
                second_gt_12 = bool((second > 12).any())

                if first_gt_12 and second_gt_12:
                    raise ValueError(
                        f"{ticker}: fechas inválidas; tanto el primer como el segundo componente superan 12."
                    )

                if first_gt_12:
                    fmt = f"%d{separator}%m{separator}%Y"
                elif second_gt_12:
                    fmt = f"%m{separator}%d{separator}%Y"
                else:
                    raise ValueError(
                        f"{ticker}: formato de fecha ambiguo ({sample.iloc[0]}). "
                        "No es posible saber si es día/mes/año o mes/día/año. "
                        "Indica el formato en Config.date_format_overrides, por ejemplo "
                        f"{{'{ticker}': '%d{separator}%m{separator}%Y'}}."
                    )

                parsed = pd.to_datetime(values, format=fmt, errors="coerce")
                return parsed, fmt

        # Último intento: pandas en modo estricto, sin dayfirst.
        try:
            parsed = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=False)
            valid_ratio = parsed.notna().mean()
            if valid_ratio >= 0.95:
                return parsed, "mixed/dayfirst=False"
        except Exception:
            pass

        raise ValueError(
            f"{ticker}: no se ha podido detectar de forma segura el formato de fecha. "
            "Especifica Config.date_format_overrides."
        )


# =============================================================================
# 5. PREPROCESAMIENTO Y AUDITORÍA
# =============================================================================

class DataPreprocessor:
    COLUMN_ALIASES = {
        "Date": "date", "Fecha": "date", "date": "date",
        "Open": "Open", "Apertura": "Open",
        "High": "High", "Máximo": "High", "Maximo": "High",
        "Low": "Low", "Mínimo": "Low", "Minimo": "Low",
        "Close": "Close", "Cierre": "Close", "Último": "Close",
        "Ultimo": "Close", "Price": "Close",
        "Volume": "Volume", "Volumen": "Volume", "Vol.": "Volume",
    }

    def __init__(self, config: Config):
        self.config = config

    def load_and_clean(self, ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        ticker = ticker.upper()
        raw_path = os.path.join(self.config.raw_data_dir, f"{ticker.lower()}.csv")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"No existe {raw_path}")

        print(f"\nPreprocesando {ticker}: {raw_path}")
        raw = self._read_csv_robust(raw_path)
        report: Dict[str, Any] = {
            "ticker": ticker,
            "source_file": raw_path,
            "rows_original": int(len(raw)),
        }

        df = raw.copy()
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={c: self.COLUMN_ALIASES.get(c, c) for c in df.columns})

        required = ["date", "Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{ticker}: faltan columnas obligatorias {missing}. "
                f"Columnas detectadas: {list(df.columns)}"
            )

        if "Volume" not in df.columns:
            df["Volume"] = 0.0
            report["volume_column_missing"] = True
        else:
            report["volume_column_missing"] = False

        # Fechas robustas.
        override = self.config.date_format_overrides.get(ticker)
        parsed_dates, detected_format = RobustDateParser.parse(df["date"], ticker, override)
        df["date"] = parsed_dates
        report["date_format_detected"] = detected_format

        invalid_date_mask = df["date"].isna()
        report["invalid_dates_removed"] = int(invalid_date_mask.sum())
        df = df.loc[~invalid_date_mask].copy()

        # Números.
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = self._to_numeric(df[col])

        invalid_numeric_mask = df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
        report["invalid_numeric_rows_removed"] = int(invalid_numeric_mask.sum())
        df = df.loc[~invalid_numeric_mask].copy()
        df["Volume"] = df["Volume"].fillna(0.0)

        # Duplicados de fecha: conservar la última observación.
        duplicates_before = int(df.duplicated(subset=["date"], keep="last").sum())
        report["duplicate_dates_removed"] = duplicates_before
        df = df.drop_duplicates(subset=["date"], keep="last").copy()

        # Registros OHLC imposibles.
        positive_mask = (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        high_ok = df["High"] >= df[["Open", "Close", "Low"]].max(axis=1)
        low_ok = df["Low"] <= df[["Open", "Close", "High"]].min(axis=1)
        ohlc_ok = positive_mask & high_ok & low_ok
        report["invalid_ohlc_rows_removed"] = int((~ohlc_ok).sum())
        df = df.loc[ohlc_ok].copy()

        negative_volume = df["Volume"] < 0
        report["negative_volume_rows_removed"] = int(negative_volume.sum())
        df = df.loc[~negative_volume].copy()

        # Orden cronológico definitivo.
        df = df.sort_values("date").reset_index(drop=True)
        df["Ticker"] = ticker

        # Rango temporal solicitado.
        start = pd.to_datetime(self.config.start_date, format="%Y-%m-%d")
        end = pd.to_datetime(self.config.end_date, format="%Y-%m-%d")
        outside = (df["date"] < start) | (df["date"] > end)
        report["rows_outside_requested_range_removed"] = int(outside.sum())
        df = df.loc[~outside].copy().reset_index(drop=True)

        if len(df) < 260:
            raise ValueError(
                f"{ticker}: solo quedan {len(df)} sesiones después de limpiar. "
                "Se recomienda al menos un año de datos diarios."
            )

        # Auditoría de retornos y huecos. Se marcan, no se eliminan automáticamente.
        df["audit_return_1d"] = df["Close"].pct_change()
        df["audit_calendar_gap_days"] = df["date"].diff().dt.days

        suspicious_mask = df["audit_return_1d"].abs() > self.config.suspicious_daily_return
        hard_mask = df["audit_return_1d"].abs() > self.config.hard_daily_return_limit
        gap_mask = df["audit_calendar_gap_days"] > self.config.max_calendar_gap_days

        report["suspicious_return_rows"] = int(suspicious_mask.sum())
        report["hard_return_limit_rows"] = int(hard_mask.sum())
        report["large_calendar_gaps"] = int(gap_mask.sum())
        report["max_abs_daily_return"] = (
            float(df["audit_return_1d"].abs().max())
            if df["audit_return_1d"].notna().any() else 0.0
        )
        report["rows_clean"] = int(len(df))
        report["first_date"] = str(df["date"].min().date())
        report["last_date"] = str(df["date"].max().date())
        report["dates_monotonic"] = bool(df["date"].is_monotonic_increasing)
        report["remaining_duplicate_dates"] = int(df["date"].duplicated().sum())

        # Guardamos anomalías para inspección manual.
        anomaly_mask = suspicious_mask | hard_mask | gap_mask
        anomalies = df.loc[
            anomaly_mask,
            ["date", "Open", "High", "Low", "Close", "Volume", "audit_return_1d", "audit_calendar_gap_days"]
        ].copy()

        preprocessing_dir = os.path.join(self.config.output_dir, "preprocessing")
        os.makedirs(preprocessing_dir, exist_ok=True)

        anomalies_path = os.path.join(preprocessing_dir, f"{ticker.lower()}_anomalies.csv")
        anomalies.to_csv(anomalies_path, index=False)
        report["anomalies_file"] = anomalies_path

        # Quitamos columnas de auditoría antes de persistir dataset limpio.
        clean = df.drop(columns=["audit_return_1d", "audit_calendar_gap_days"]).copy()
        clean = clean[["date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

        processed_path = os.path.join(self.config.processed_data_dir, f"{ticker.lower()}_clean.csv")
        clean.to_csv(processed_path, index=False, date_format="%Y-%m-%d")
        report["processed_file"] = processed_path

        report_path = os.path.join(preprocessing_dir, f"{ticker.lower()}_preprocessing_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["report_file"] = report_path

        self._print_report(report)

        if report["hard_return_limit_rows"] > 0:
            print(
                f"ADVERTENCIA CRÍTICA: {ticker} contiene {report['hard_return_limit_rows']} retornos diarios "
                f"por encima de {self.config.hard_daily_return_limit:.0%}. Revisa {anomalies_path}."
            )

        return clean, report

    @staticmethod
    def _read_csv_robust(path: str) -> pd.DataFrame:
        attempts = [
            {"sep": ",", "encoding": "utf-8"},
            {"sep": ";", "encoding": "utf-8"},
            {"sep": ",", "encoding": "latin1"},
            {"sep": ";", "encoding": "latin1"},
        ]

        last_exc = None
        for kwargs in attempts:
            try:
                df = pd.read_csv(path, **kwargs)
                if df.shape[1] >= 5:
                    return df
            except Exception as exc:
                last_exc = exc

        raise ValueError(f"No se pudo leer correctamente {path}: {last_exc}")

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        def parse(value: Any) -> float:
            if pd.isna(value):
                return np.nan

            x = str(value).strip().replace("%", "").replace(" ", "")
            multiplier = 1.0
            if x.upper().endswith("K"):
                multiplier = 1e3
                x = x[:-1]
            elif x.upper().endswith("M"):
                multiplier = 1e6
                x = x[:-1]
            elif x.upper().endswith("B"):
                multiplier = 1e9
                x = x[:-1]

            if "," in x and "." in x:
                # El último separador suele ser el decimal.
                if x.rfind(",") > x.rfind("."):
                    x = x.replace(".", "").replace(",", ".")
                else:
                    x = x.replace(",", "")
            elif "," in x:
                x = x.replace(",", ".")

            try:
                return float(x) * multiplier
            except Exception:
                return np.nan

        return series.apply(parse)

    @staticmethod
    def _print_report(report: Dict[str, Any]):
        print("\nPREPROCESAMIENTO / AUDITORÍA")
        print("-" * 90)
        keys = [
            "ticker", "rows_original", "date_format_detected", "invalid_dates_removed",
            "invalid_numeric_rows_removed", "duplicate_dates_removed", "invalid_ohlc_rows_removed",
            "negative_volume_rows_removed", "rows_outside_requested_range_removed", "rows_clean",
            "first_date", "last_date", "dates_monotonic", "remaining_duplicate_dates",
            "suspicious_return_rows", "hard_return_limit_rows", "large_calendar_gaps",
            "max_abs_daily_return", "processed_file", "anomalies_file",
        ]
        for key in keys:
            print(f"{key:40s}: {report.get(key)}")
        print("-" * 90)


# =============================================================================
# 6. SENTIMIENTO HISTÓRICO REAL: GDELT + FINBERT
# =============================================================================

class GDELTHistoricalSentimentBuilder:
    """
    Construye un histórico diario de sentimiento financiero a partir de
    GDELT DOC 2.0.

    Motivo:
        Google News RSS es excelente para noticias recientes, pero no es una
        API histórica. GDELT permite consultar una serie temporal de cobertura
        de noticias desde 2017.

    Metodología:
        - TimelineTone: tono medio diario de las noticias que cumplen la query.
        - TimelineVolRaw: número de noticias diario.
        - El tono GDELT se normaliza a [-1, 1] mediante tanh(tone / 5).
        - Las noticias recientes procesadas con FinBERT tienen prioridad sobre
          GDELT para las fechas en las que ambas fuentes existen.

    IMPORTANTE:
        El histórico previo a 2017 no se inventa: queda sin sentimiento.
        Esto permite que el modelo entrene NLP únicamente en el periodo para el
        que existe evidencia histórica real.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    QUERY_MAP = {
        "SXR8": '("S&P 500" OR "Federal Reserve" OR "US stocks" OR "Wall Street" OR "US inflation") sourcelang:english',
        "SXRV": '("Nasdaq 100" OR "Nasdaq" OR "technology stocks" OR semiconductors OR Nvidia OR Microsoft OR Apple) sourcelang:english',
    }

    def __init__(self, config: Config):
        self.config = config

    def update(self, ticker: str) -> Optional[pd.DataFrame]:
        if not self.config.use_gdelt_history:
            return None

        ticker = ticker.upper()
        path = os.path.join(
            self.config.historical_sentiment_dir,
            f"{ticker.lower()}_sentiment.csv",
        )

        start = max(
            pd.Timestamp(self.config.start_date),
            pd.Timestamp(self.config.gdelt_history_start),
        )
        end = pd.Timestamp(self.config.end_date)

        if end < start:
            return None

        print(
            f"GDELT histórico {ticker}: {start.date()} -> {end.date()} "
            "(TimelineTone + TimelineVolRaw)"
        )

        try:
            tone = self._fetch_timeline(ticker, "timelinetone", start, end)
            volume = self._fetch_timeline(ticker, "timelinevolraw", start, end)

            if tone.empty:
                raise RuntimeError("GDELT no devolvió TimelineTone.")

            tone = tone.rename(columns={"value": "gdelt_tone"})
            if not volume.empty:
                volume = volume.rename(columns={"value": "gdelt_news_count"})
                gdelt = tone.merge(volume, on="date", how="left")
            else:
                gdelt = tone.copy()
                gdelt["gdelt_news_count"] = 1.0

            gdelt["gdelt_tone"] = pd.to_numeric(
                gdelt["gdelt_tone"], errors="coerce"
            )
            gdelt["gdelt_news_count"] = pd.to_numeric(
                gdelt["gdelt_news_count"], errors="coerce"
            ).fillna(0.0)

            gdelt = gdelt.dropna(subset=["date", "gdelt_tone"]).copy()

            # GDELT Tone no está acotado exactamente a [-1, 1].
            # tanh conserva el signo y limita el efecto de valores extremos.
            gdelt["sentiment_mean"] = np.tanh(gdelt["gdelt_tone"] / 5.0)

            # Proxies continuos compatibles con las features FinBERT.
            gdelt["sentiment_positive_ratio"] = np.clip(
                gdelt["sentiment_mean"], 0.0, 1.0
            )
            gdelt["sentiment_negative_ratio"] = np.clip(
                -gdelt["sentiment_mean"], 0.0, 1.0
            )
            gdelt["news_count"] = gdelt["gdelt_news_count"].clip(lower=0.0)
            gdelt["sentiment_available"] = (
                gdelt["news_count"].fillna(0) > 0
            ).astype(float)
            gdelt["sentiment_source"] = "GDELT_TONE"

            gdelt = gdelt[
                [
                    "date",
                    "sentiment_mean",
                    "sentiment_positive_ratio",
                    "sentiment_negative_ratio",
                    "news_count",
                    "sentiment_available",
                    "sentiment_source",
                    "gdelt_tone",
                ]
            ].sort_values("date")

            merged = self._merge_with_existing(path, gdelt)
            merged.to_csv(path, index=False, date_format="%Y-%m-%d")

            real_days = int((merged["sentiment_available"] > 0).sum())
            print(
                f"Histórico NLP actualizado: {path} "
                f"({len(merged)} fechas, {real_days} días disponibles)."
            )
            return merged

        except Exception as exc:
            print(f"GDELT histórico {ticker}: ERROR -> {exc}")
            if self.config.require_gdelt_history:
                raise
            if os.path.exists(path):
                print("Se utilizará el histórico NLP ya guardado en caché.")
                try:
                    return pd.read_csv(path)
                except Exception:
                    pass
            return None

    def _fetch_timeline(
        self,
        ticker: str,
        mode: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Descarga una timeline de GDELT.

        GDELT puede responder HTTP 429 durante periodos de carga elevada.
        Esta función:
          - respeta Retry-After cuando existe;
          - aplica espera exponencial limitada;
          - no convierte un 429 temporal en un fallo difícil de diagnosticar;
          - deja que update() use la caché local si finalmente no responde.
        """
        params = {
            "query": self.QUERY_MAP.get(ticker, ticker),
            "mode": mode,
            "format": "json",
            "startdatetime": start.strftime("%Y%m%d000000"),
            "enddatetime": end.strftime("%Y%m%d235959"),
            "timelinesmooth": 0,
        }

        last_error = None

        for attempt in range(1, self.config.gdelt_retries + 1):
            try:
                # Pequeña pausa preventiva entre peticiones GDELT.
                if attempt == 1 and self.config.gdelt_request_pause_seconds > 0:
                    time.sleep(self.config.gdelt_request_pause_seconds)

                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.config.gdelt_timeout_seconds,
                    headers={
                        "User-Agent": "Mozilla/5.0 TFM-ETF-Research/1.0"
                    },
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        retry_after_seconds = float(retry_after)
                    except (TypeError, ValueError):
                        retry_after_seconds = 2.0 ** attempt

                    wait_seconds = min(
                        max(retry_after_seconds, 2.0 ** attempt),
                        float(self.config.gdelt_max_retry_wait_seconds),
                    )

                    last_error = RuntimeError(
                        f"GDELT rate limit HTTP 429 ({mode})."
                    )

                    print(
                        f"GDELT {ticker} {mode}: HTTP 429. "
                        f"Reintento {attempt}/{self.config.gdelt_retries} "
                        f"en {wait_seconds:.0f}s..."
                    )

                    if attempt < self.config.gdelt_retries:
                        time.sleep(wait_seconds)
                        continue

                    raise last_error

                response.raise_for_status()

                # Algunos fallos temporales pueden devolver HTML en lugar de JSON.
                content_type = response.headers.get("Content-Type", "")
                if "json" not in content_type.lower():
                    body_start = response.text[:160].replace("\n", " ")
                    raise RuntimeError(
                        f"GDELT devolvió contenido no JSON ({content_type}): "
                        f"{body_start}"
                    )

                payload = response.json()
                rows = self._extract_timeline_rows(payload)

                if not rows:
                    raise RuntimeError(
                        f"Respuesta GDELT sin datos reconocibles para {mode}."
                    )

                df = pd.DataFrame(rows, columns=["date", "value"])

                # utc=True evita mezclas de datetimes aware/naive.
                dates = pd.to_datetime(
                    df["date"],
                    errors="coerce",
                    utc=True,
                )
                df["date"] = dates.dt.tz_convert(None).dt.normalize()
                df["value"] = pd.to_numeric(df["value"], errors="coerce")

                return (
                    df.dropna(subset=["date", "value"])
                    .groupby("date", as_index=False)["value"]
                    .mean()
                    .sort_values("date")
                    .reset_index(drop=True)
                )

            except Exception as exc:
                last_error = exc

                if attempt < self.config.gdelt_retries:
                    wait_seconds = min(
                        2.0 ** attempt,
                        float(self.config.gdelt_max_retry_wait_seconds),
                    )
                    print(
                        f"GDELT {ticker} {mode}: error temporal "
                        f"({type(exc).__name__}). "
                        f"Reintento {attempt}/{self.config.gdelt_retries} "
                        f"en {wait_seconds:.0f}s..."
                    )
                    time.sleep(wait_seconds)

        raise RuntimeError(
            f"No se pudo descargar GDELT {mode}: {last_error}"
        )


    @staticmethod
    def _extract_timeline_rows(payload: Any) -> List[Tuple[Any, Any]]:
        """
        GDELT ha utilizado más de una envoltura JSON a lo largo del tiempo.
        Este extractor acepta las formas comunes:
            {"timeline": [{"data": [{"date": ..., "value": ...}, ...]}]}
            {"timeline": [{"date": ..., "value": ...}, ...]}
            {"data": [{"date": ..., "value": ...}, ...]}
        """
        rows: List[Tuple[Any, Any]] = []

        def walk(obj: Any):
            if isinstance(obj, dict):
                if "date" in obj and "value" in obj:
                    rows.append((obj.get("date"), obj.get("value")))
                    return
                for key in ("timeline", "data"):
                    if key in obj:
                        walk(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(payload)
        return rows

    @staticmethod
    def _merge_with_existing(path: str, gdelt: pd.DataFrame) -> pd.DataFrame:
        common = [
            "date",
            "sentiment_mean",
            "sentiment_positive_ratio",
            "sentiment_negative_ratio",
            "news_count",
            "sentiment_available",
            "sentiment_source",
            "gdelt_tone",
        ]

        gdelt = gdelt.copy()
        gdelt["date"] = pd.to_datetime(gdelt["date"], errors="coerce")

        if not os.path.exists(path):
            return gdelt[common].copy()

        try:
            existing = pd.read_csv(path)
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")

            if "sentiment_source" not in existing.columns:
                # Los históricos creados por la versión Google News/FinBERT
                # anterior son FinBERT reales.
                existing["sentiment_source"] = "FINBERT_GOOGLE_NEWS"
            if "gdelt_tone" not in existing.columns:
                existing["gdelt_tone"] = np.nan

            for col in common:
                if col not in existing.columns:
                    existing[col] = np.nan

            combined = pd.concat(
                [gdelt[common], existing[common]],
                ignore_index=True,
            )

            # Prioridad: FinBERT > GDELT para una misma fecha.
            combined["_priority"] = combined["sentiment_source"].map(
                {
                    "GDELT_TONE": 1,
                    "FINBERT_GOOGLE_NEWS": 2,
                }
            ).fillna(1)

            combined = (
                combined.sort_values(["date", "_priority"])
                .drop_duplicates("date", keep="last")
                .drop(columns="_priority")
                .sort_values("date")
                .reset_index(drop=True)
            )
            return combined
        except Exception:
            return gdelt[common].copy()


# =============================================================================
# 7. CARGA DE SENTIMIENTO HISTÓRICO
# =============================================================================

class HistoricalSentimentLoader:
    BASE_COLUMNS = [
        "date",
        "sentiment_mean",
        "sentiment_positive_ratio",
        "sentiment_negative_ratio",
        "news_count",
        "sentiment_available",
    ]

    def __init__(self, config: Config):
        self.config = config

    def load(
        self,
        ticker: str,
        market_dates: Optional[pd.Series] = None,
    ) -> Optional[pd.DataFrame]:

        path = os.path.join(
            self.config.historical_sentiment_dir,
            f"{ticker.lower()}_sentiment.csv",
        )

        if not os.path.exists(path):
            print(
                f"Sentimiento histórico {ticker}: NO DISPONIBLE."
            )
            return None

        try:
            raw = pd.read_csv(path)
            missing = [
                c for c in self.BASE_COLUMNS
                if c not in raw.columns
            ]
            if missing:
                print(
                    f"Sentimiento histórico {ticker}: "
                    f"columnas faltantes {missing}."
                )
                return None

            raw["date"] = pd.to_datetime(
                raw["date"],
                errors="coerce",
            ).dt.normalize()
            raw = (
                raw.dropna(subset=["date"])
                .drop_duplicates("date", keep="last")
                .sort_values("date")
            )

            for col in self.BASE_COLUMNS[1:]:
                raw[col] = pd.to_numeric(
                    raw[col],
                    errors="coerce",
                ).fillna(0.0)

            real_days = int(
                (raw["sentiment_available"] > 0).sum()
            )

            # Construcción de calendario diario completo.
            full_index = pd.date_range(
                raw["date"].min(),
                raw["date"].max(),
                freq="D",
            )

            daily = (
                raw.set_index("date")
                .reindex(full_index)
            )
            daily.index.name = "date"

            daily["news_count"] = daily["news_count"].fillna(0.0)
            daily["sentiment_available"] = (
                daily["news_count"] > 0
            ).astype(float)

            # En días sin noticias NO tratamos el sentimiento como neutral.
            # La media ponderada se calcula solo usando días con artículos.
            daily["sentiment_weighted_sum"] = (
                daily["sentiment_mean"].fillna(0.0)
                * daily["news_count"]
            )
            daily["positive_weighted_sum"] = (
                daily["sentiment_positive_ratio"].fillna(0.0)
                * daily["news_count"]
            )
            daily["negative_weighted_sum"] = (
                daily["sentiment_negative_ratio"].fillna(0.0)
                * daily["news_count"]
            )

            feature_data = pd.DataFrame(index=daily.index)

            for window in self.config.sentiment_windows:
                min_periods = 1

                count_roll = daily["news_count"].rolling(
                    f"{int(window)}D",
                    min_periods=min_periods,
                ).sum()

                sent_sum = daily[
                    "sentiment_weighted_sum"
                ].rolling(
                    f"{int(window)}D",
                    min_periods=min_periods,
                ).sum()

                pos_sum = daily[
                    "positive_weighted_sum"
                ].rolling(
                    f"{int(window)}D",
                    min_periods=min_periods,
                ).sum()

                neg_sum = daily[
                    "negative_weighted_sum"
                ].rolling(
                    f"{int(window)}D",
                    min_periods=min_periods,
                ).sum()

                safe_count = count_roll.replace(0, np.nan)

                feature_data[
                    f"sentiment_mean_{window}d"
                ] = (sent_sum / safe_count).fillna(0.0)

                feature_data[
                    f"sentiment_positive_ratio_{window}d"
                ] = (pos_sum / safe_count).fillna(0.0)

                feature_data[
                    f"sentiment_negative_ratio_{window}d"
                ] = (neg_sum / safe_count).fillna(0.0)

                feature_data[
                    f"news_count_{window}d"
                ] = count_roll.fillna(0.0)

                feature_data[
                    f"news_count_log_{window}d"
                ] = np.log1p(count_roll.fillna(0.0))

                feature_data[
                    f"sentiment_available_{window}d"
                ] = (count_roll > 0).astype(float)

            # Momentum NLP entre ventanas cortas y largas.
            pairs = [(5, 20), (10, 30), (15, 45), (20, 60)]
            for short_w, long_w in pairs:
                short_col = f"sentiment_mean_{short_w}d"
                long_col = f"sentiment_mean_{long_w}d"
                if (
                    short_col in feature_data.columns
                    and long_col in feature_data.columns
                ):
                    feature_data[
                        f"sentiment_momentum_{short_w}_{long_w}"
                    ] = (
                        feature_data[short_col]
                        - feature_data[long_col]
                    )

            feature_data = feature_data.reset_index()

            # Anti-look-ahead: el sentimiento del día t se hace efectivo
            # desde t + lag días.
            lag = max(int(self.config.sentiment_lag_days), 0)
            feature_data["date"] = (
                feature_data["date"]
                + pd.Timedelta(days=lag)
            )

            coverage = np.nan
            if market_dates is not None:
                market_days = (
                    pd.Series(pd.to_datetime(market_dates))
                    .dt.normalize()
                    .drop_duplicates()
                )

                # Usamos la ventana de 20 días para medir cobertura operativa.
                coverage_col = (
                    "sentiment_available_20d"
                    if "sentiment_available_20d" in feature_data.columns
                    else f"sentiment_available_{self.config.sentiment_windows[-1]}d"
                )

                lookup = feature_data[
                    ["date", coverage_col]
                ].copy()

                merged = pd.DataFrame(
                    {"date": market_days}
                ).merge(
                    lookup,
                    on="date",
                    how="left",
                )

                coverage = (
                    merged[coverage_col]
                    .fillna(0.0)
                    .gt(0)
                    .mean()
                )

            print(
                f"Sentimiento histórico {ticker}: "
                f"{real_days} días reales de noticias"
                + (
                    f", cobertura operativa={coverage:.1%}"
                    if pd.notna(coverage) else ""
                )
                + "."
            )

            if real_days < self.config.min_sentiment_days_for_model:
                print(
                    f"NLP histórico {ticker}: NO APTO. "
                    f"Se requieren {self.config.min_sentiment_days_for_model} "
                    f"días reales; hay {real_days}."
                )
                return None

            if (
                pd.notna(coverage)
                and coverage
                < self.config.min_sentiment_coverage_for_model
            ):
                print(
                    f"NLP histórico {ticker}: NO APTO. "
                    f"Cobertura {coverage:.1%} < "
                    f"{self.config.min_sentiment_coverage_for_model:.1%}."
                )
                return None

            print(
                f"NLP histórico {ticker}: APTO PARA ENTRENAMIENTO "
                f"({len(feature_data.columns)-1} features NLP)."
            )
            return feature_data

        except Exception as exc:
            print(
                f"Sentimiento histórico {ticker}: error {exc}."
            )
            return None


class NewsSentimentAnalyzer:
    """
    Descarga titulares recientes desde Google News RSS, los clasifica con
    FinBERT y mantiene dos cachés persistentes:

      data/news_cache/<ticker>_articles.csv
      data/news_sentiment/<ticker>_sentiment.csv

    Solo los artículos nuevos se vuelven a pasar por FinBERT. De este modo el
    sistema va construyendo un histórico real de sentimiento con cada ejecución.

    IMPORTANTE: Google News RSS es útil para noticias recientes, no garantiza
    un archivo histórico completo desde 2015. Por eso las variables NLP solo
    entran en el modelo cuando la cobertura acumulada supera los umbrales de
    Config.
    """

    QUERY_MAP = {
        "SXR8": '("S&P 500" OR "Federal Reserve" OR inflation OR "US stocks" OR earnings)',
        "SXRV": '("Nasdaq 100" OR "technology stocks" OR semiconductors OR Nvidia OR Microsoft OR Apple)',
    }

    def __init__(self, config: Config):
        self.config = config
        self.finbert = None
        self.finbert_load_error = ""

        if config.use_news:
            if not (FEEDPARSER_AVAILABLE and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE):
                return

            try:
                device = 0 if torch.cuda.is_available() else -1
                device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
                print(f"Cargando FinBERT ({device_name})...")

                self.finbert = hf_pipeline(
                    "text-classification",
                    model=config.finbert_model,
                    tokenizer=config.finbert_model,
                    top_k=None,
                    device=device,
                )
                print("FinBERT: OK")
            except Exception as exc:
                self.finbert_load_error = repr(exc)
                print(f"FinBERT: NO DISPONIBLE ({self.finbert_load_error})")

    @staticmethod
    def _empty_score() -> Dict[str, float]:
        return {
            "sentiment_mean": 0.0,
            "sentiment_positive_ratio": 0.0,
            "sentiment_negative_ratio": 0.0,
            "news_count": 0.0,
            "sentiment_available": 0.0,
        }

    @staticmethod
    def _article_id(title: str, link: str) -> str:
        raw = f"{title.strip()}|{link.strip()}".encode("utf-8", errors="ignore")
        return hashlib.sha256(raw).hexdigest()[:24]

    def fetch_score_and_update_history(self, ticker: str) -> Tuple[Dict[str, float], pd.DataFrame]:
        ticker = ticker.upper()
        if not self.config.use_news:
            return self._empty_score(), pd.DataFrame()
        if self.finbert is None:
            print("FinBERT no está cargado; no se puntúan noticias.")
            return self._empty_score(), pd.DataFrame()

        fresh = self._fetch_google_news(ticker)
        if fresh.empty:
            print(f"Google News RSS {ticker}: 0 noticias recuperadas.")
            return self._empty_score(), pd.DataFrame()

        cache_path = os.path.join(self.config.news_cache_dir, f"{ticker.lower()}_articles.csv")
        cached = self._load_article_cache(cache_path)

        fresh["article_id"] = [
            self._article_id(t, l)
            for t, l in zip(fresh["title"].fillna(""), fresh["link"].fillna(""))
        ]

        already = set(cached["article_id"].astype(str)) if not cached.empty else set()
        new_articles = fresh[~fresh["article_id"].isin(already)].copy()

        print(
            f"Google News RSS {ticker}: {len(fresh)} titulares, "
            f"{len(new_articles)} nuevos para FinBERT."
        )

        if not new_articles.empty:
            scored = self._score_articles(new_articles)

            # cached procede de CSV y scored de memoria. Normalizamos los dos
            # ANTES y DESPUÉS del concat para que published_at tenga un único tipo.
            cached = self._normalize_article_cache(cached)
            scored = self._normalize_article_cache(scored)

            cached = pd.concat(
                [cached, scored],
                ignore_index=True,
                sort=False,
            )
            cached = self._normalize_article_cache(cached)

            cached = (
                cached.drop_duplicates("article_id", keep="last")
                .sort_values("published_at", kind="stable")
                .reset_index(drop=True)
            )

            # Para persistencia usamos ISO 8601, evitando formatos ambiguos.
            cache_to_save = cached.copy()
            cache_to_save["published_at"] = (
                cache_to_save["published_at"]
                .dt.strftime("%Y-%m-%dT%H:%M:%S")
            )
            cache_to_save["date"] = (
                cache_to_save["date"]
                .dt.strftime("%Y-%m-%d")
            )
            cache_to_save.to_csv(cache_path, index=False)

        else:
            scored = pd.DataFrame()
            cached = self._normalize_article_cache(cached)

        # Reagregamos el histórico diario completo desde la caché de artículos.
        self._write_daily_history(ticker, cached)

        # El contexto actual se calcula con los artículos del RSS actual,
        # buscando sus puntuaciones en la caché ya actualizada.
        current_ids = set(fresh["article_id"].astype(str))
        cached = self._normalize_article_cache(cached)
        current_scored = cached[
            cached["article_id"].astype(str).isin(current_ids)
        ].copy()
        current_score = self._aggregate_score(current_scored)
        return current_score, current_scored

    def _fetch_google_news(self, ticker: str) -> pd.DataFrame:
        if not FEEDPARSER_AVAILABLE:
            return pd.DataFrame()

        query = self.QUERY_MAP.get(ticker, ticker)
        # Google News admite "when:Nd" como parte de la consulta. No se usa
        # como fuente histórica garantizada; solo para ampliar noticias recientes.
        query = f"{query} when:{int(self.config.news_when_days)}d"
        url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        )

        try:
            feed = feedparser.parse(url)
            if getattr(feed, "bozo", False):
                print(f"Google News RSS {ticker}: aviso parser {getattr(feed, 'bozo_exception', '')}")

            rows = []
            for entry in feed.entries[: self.config.max_news_items]:
                published = getattr(entry, "published", None)
                parsed_date = pd.to_datetime(published, utc=True, errors="coerce")
                if pd.isna(parsed_date):
                    continue
                parsed_date = parsed_date.tz_convert(None)

                source_obj = getattr(entry, "source", None)
                source_name = getattr(source_obj, "title", "Google News") if source_obj else "Google News"
                title = unescape(str(getattr(entry, "title", ""))).strip()
                link = str(getattr(entry, "link", "")).strip()

                if not title:
                    continue

                rows.append({
                    "published_at": parsed_date,
                    "date": parsed_date.normalize(),
                    "title": title,
                    "link": link,
                    "source": source_name,
                    "query": query,
                })

            return pd.DataFrame(rows).drop_duplicates(["title", "link"])
        except Exception as exc:
            print(f"Google News RSS {ticker}: error {exc}")
            return pd.DataFrame()

    @staticmethod
    def _normalize_article_cache(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza tipos de la caché de noticias.

        Es imprescindible llamar a esta función DESPUÉS de cualquier concat.
        Una parte de la caché puede venir de CSV (strings) y otra parte de
        FinBERT (Timestamp). Pandas no puede ordenar str y Timestamp juntos.
        """
        columns = [
            "article_id", "published_at", "date", "title", "link", "source", "query",
            "label", "confidence", "sentiment_score",
        ]

        if df is None or df.empty:
            return pd.DataFrame(columns=columns)

        out = df.copy()

        for col in columns:
            if col not in out.columns:
                out[col] = np.nan

        # format="mixed" es útil cuando la caché contiene versiones antiguas
        # con distintos formatos. utc=True homogeniza timezone-aware/naive.
        published = pd.to_datetime(
            out["published_at"],
            errors="coerce",
            format="mixed",
            utc=True,
        )
        out["published_at"] = published.dt.tz_convert(None)

        dates = pd.to_datetime(
            out["date"],
            errors="coerce",
            format="mixed",
            utc=True,
        )
        out["date"] = dates.dt.tz_convert(None).dt.normalize()

        out["article_id"] = out["article_id"].astype("string")
        out["title"] = out["title"].fillna("").astype(str)
        out["link"] = out["link"].fillna("").astype(str)
        out["source"] = out["source"].fillna("").astype(str)
        out["query"] = out["query"].fillna("").astype(str)

        out["confidence"] = pd.to_numeric(
            out["confidence"], errors="coerce"
        )
        out["sentiment_score"] = pd.to_numeric(
            out["sentiment_score"], errors="coerce"
        )

        out = out.dropna(
            subset=["article_id", "published_at"]
        )

        return out[columns].reset_index(drop=True)

    @staticmethod
    def _load_article_cache(path: str) -> pd.DataFrame:
        columns = [
            "article_id", "published_at", "date", "title", "link", "source", "query",
            "label", "confidence", "sentiment_score",
        ]

        if not os.path.exists(path):
            return pd.DataFrame(columns=columns)

        try:
            df = pd.read_csv(path)
            return NewsSentimentAnalyzer._normalize_article_cache(df)
        except Exception as exc:
            print(
                f"Aviso: no se pudo leer correctamente la caché "
                f"{path}: {exc}. Se reconstruirá."
            )
            return pd.DataFrame(columns=columns)

    def _score_articles(self, news: pd.DataFrame) -> pd.DataFrame:
        out = news.copy()
        labels = []
        confidences = []
        scores = []

        titles = out["title"].fillna("").astype(str).tolist()
        # Pipeline admite listas y es bastante más eficiente que una llamada por título.
        batch_size = 16
        for start in range(0, len(titles), batch_size):
            batch = [t[:512] for t in titles[start:start + batch_size]]
            try:
                results = self.finbert(batch, batch_size=batch_size, truncation=True)
            except TypeError:
                results = self.finbert(batch)

            for result in results:
                # top_k=None -> lista de etiquetas para cada texto.
                candidates = result if isinstance(result, list) else [result]
                best = max(candidates, key=lambda x: float(x["score"]))
                label = str(best["label"]).lower()
                confidence = float(best["score"])

                if "positive" in label:
                    score = confidence
                    canonical = "positive"
                elif "negative" in label:
                    score = -confidence
                    canonical = "negative"
                else:
                    score = 0.0
                    canonical = "neutral"

                labels.append(canonical)
                confidences.append(confidence)
                scores.append(score)

        out["label"] = labels
        out["confidence"] = confidences
        out["sentiment_score"] = scores

        # Mantener tipos datetime internamente. Solo se convierten a texto
        # ISO cuando se persisten a CSV.
        out["published_at"] = pd.to_datetime(
            out["published_at"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)

        out["date"] = pd.to_datetime(
            out["date"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None).dt.normalize()

        return self._normalize_article_cache(out)

    def _write_daily_history(self, ticker: str, article_cache: pd.DataFrame):
        if article_cache.empty:
            return

        df = self._normalize_article_cache(article_cache)
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None).dt.normalize()
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
        df = df.dropna(subset=["date", "sentiment_score"])
        if df.empty:
            return

        df["is_positive"] = (df["sentiment_score"] > 0).astype(float)
        df["is_negative"] = (df["sentiment_score"] < 0).astype(float)

        daily = df.groupby("date", as_index=False).agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_positive_ratio=("is_positive", "mean"),
            sentiment_negative_ratio=("is_negative", "mean"),
            news_count=("article_id", "count"),
        )
        daily["sentiment_available"] = 1.0
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

        daily["sentiment_source"] = "FINBERT_GOOGLE_NEWS"
        daily["gdelt_tone"] = np.nan

        path = os.path.join(
            self.config.historical_sentiment_dir,
            f"{ticker.lower()}_sentiment.csv",
        )

        # Nunca sobrescribimos el histórico GDELT. Fusionamos y damos
        # prioridad a FinBERT en las fechas en las que existen noticias
        # recientes clasificadas por el transformer.
        if os.path.exists(path):
            try:
                old_history = pd.read_csv(path)
                if "sentiment_source" not in old_history.columns:
                    old_history["sentiment_source"] = "GDELT_TONE"
                if "gdelt_tone" not in old_history.columns:
                    old_history["gdelt_tone"] = np.nan

                combined = pd.concat(
                    [old_history, daily],
                    ignore_index=True,
                    sort=False,
                )
                combined["date"] = pd.to_datetime(
                    combined["date"], errors="coerce"
                )
                combined["_priority"] = combined["sentiment_source"].map(
                    {
                        "GDELT_TONE": 1,
                        "FINBERT_GOOGLE_NEWS": 2,
                    }
                ).fillna(1)
                combined = (
                    combined.dropna(subset=["date"])
                    .sort_values(["date", "_priority"])
                    .drop_duplicates("date", keep="last")
                    .drop(columns="_priority")
                    .sort_values("date")
                )
                combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
                combined.to_csv(path, index=False)
            except Exception:
                daily.to_csv(path, index=False)
        else:
            daily.to_csv(path, index=False)

        print(
            f"Histórico FinBERT fusionado con histórico NLP: "
            f"{path} ({len(daily)} días FinBERT recientes)."
        )

    @staticmethod
    def _aggregate_score(scored_news: pd.DataFrame) -> Dict[str, float]:
        if scored_news.empty:
            return NewsSentimentAnalyzer._empty_score()

        scores = pd.to_numeric(scored_news["sentiment_score"], errors="coerce").dropna()
        if scores.empty:
            return NewsSentimentAnalyzer._empty_score()

        return {
            "sentiment_mean": float(scores.mean()),
            "sentiment_positive_ratio": float((scores > 0).mean()),
            "sentiment_negative_ratio": float((scores < 0).mean()),
            "news_count": float(len(scores)),
            "sentiment_available": 1.0,
        }


# =============================================================================
# HISTÓRICO LARGO DE NOTICIAS: ALPHA VANTAGE + FINBERT
# =============================================================================

class AlphaVantageHistoricalNewsBackfiller:
    """
    Histórico largo y reanudable de noticias usando Alpha Vantage
    NEWS_SENTIMENT + FinBERT.

    MEJORAS PRINCIPALES
    -------------------
    1. Varias familias de consulta independientes por ETF.
       Alpha Vantage interpreta múltiples tickers/topics como filtros
       simultáneos, por lo que NO los agrupamos en una única llamada.

    2. Estado reanudable por:
         ticker + familia_query + ventana_temporal

    3. División adaptativa de ventanas:
       si una respuesta se acerca al límite de 1000 artículos,
       el bloque se divide en dos antes de marcarse como completo.

    4. Relevance score:
       combina la relevancia del filtro de búsqueda, la relevancia de ticker
       devuelta por Alpha Vantage y coincidencias semánticas simples en
       título/resumen.

    5. FinBERT:
       title + summary se clasifican con el mismo modelo que las noticias
       recientes.

    6. Sentimiento diario:
       se pondera por relevance_score * confianza_FinBERT.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    QUERY_FAMILIES = {
        "SXR8": [
            {
                "name": "benchmark_spy",
                "tickers": "SPY",
                "topics": None,
                "query_weight": 1.00,
            },
            {
                "name": "financial_markets",
                "tickers": None,
                "topics": "financial_markets",
                "query_weight": 0.82,
            },
            {
                "name": "economy_monetary",
                "tickers": None,
                "topics": "economy_monetary",
                "query_weight": 0.90,
            },
            {
                "name": "economy_macro",
                "tickers": None,
                "topics": "economy_macro",
                "query_weight": 0.78,
            },
        ],
        "SXRV": [
            {
                "name": "benchmark_qqq",
                "tickers": "QQQ",
                "topics": None,
                "query_weight": 1.00,
            },
            {
                "name": "technology",
                "tickers": None,
                "topics": "technology",
                "query_weight": 0.90,
            },
            {
                "name": "financial_markets",
                "tickers": None,
                "topics": "financial_markets",
                "query_weight": 0.80,
            },
            {
                "name": "economy_monetary",
                "tickers": None,
                "topics": "economy_monetary",
                "query_weight": 0.82,
            },
        ],
    }

    KEYWORDS = {
        "SXR8": [
            "s&p 500", "s&p500", "spy", "wall street", "us stocks",
            "federal reserve", "fed", "inflation", "treasury",
            "interest rates", "rate cut", "rate hike", "us economy",
            "earnings", "recession", "jobs report", "cpi", "gdp",
        ],
        "SXRV": [
            "nasdaq", "nasdaq 100", "qqq", "technology stocks",
            "semiconductor", "semiconductors", "artificial intelligence",
            "ai", "nvidia", "apple", "microsoft", "amazon", "meta",
            "alphabet", "google", "interest rates", "federal reserve",
            "fed", "earnings", "chips", "software", "cloud",
        ],
    }

    RELEVANT_TICKERS = {
        "SXR8": {"SPY", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG"},
        "SXRV": {"QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO"},
    }

    def __init__(
        self,
        config: Config,
        sentiment_analyzer: NewsSentimentAnalyzer,
    ):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.total_calls_this_run = 0

    def update(
        self,
        ticker: str,
        market_dates: pd.Series,
    ) -> Optional[pd.DataFrame]:

        if not self.config.use_alpha_vantage_history:
            return None

        ticker = ticker.upper()
        api_key = self.config.alpha_vantage_api_key

        history_path = os.path.join(
            self.config.alpha_vantage_history_dir,
            f"{ticker.lower()}_articles.csv",
        )
        state_path = os.path.join(
            self.config.alpha_vantage_history_dir,
            f"{ticker.lower()}_backfill_state.json",
        )

        history = self._load_history(history_path)

        if not api_key:
            if history.empty and self.config.require_alpha_vantage_history:
                raise RuntimeError(
                    "Falta ALPHAVANTAGE_API_KEY y no existe histórico local."
                )
            if not history.empty:
                self._build_daily_history(ticker, history)
            return history if not history.empty else None

        start = max(
            pd.Timestamp(self.config.alpha_vantage_history_start),
            pd.Timestamp(self.config.start_date),
        ).normalize()

        end = min(
            pd.Timestamp(self.config.end_date),
            pd.Series(pd.to_datetime(market_dates)).max(),
        ).normalize()

        state = self._load_or_initialize_state(
            ticker=ticker,
            path=state_path,
            start=start,
            end=end,
        )

        remaining_global = max(
            int(self.config.alpha_vantage_max_total_calls_per_run)
            - self.total_calls_this_run,
            0,
        )
        ticker_budget = min(
            int(self.config.alpha_vantage_max_calls_per_ticker_per_run),
            remaining_global,
        )

        print("\n" + "=" * 108)
        print(f"HISTÓRICO MULTIFUENTE ALPHA VANTAGE + FINBERT: {ticker}")
        print("=" * 108)
        print(f"Periodo:                   {start.date()} -> {end.date()}")
        print(f"Familias de consulta:      {len(self.QUERY_FAMILIES[ticker])}")
        print(f"Tareas pendientes:         {len(state['pending'])}")
        print(f"Tareas completadas:        {len(state['completed'])}")
        print(f"Presupuesto ETF esta vez:  {ticker_budget} llamadas")
        print(f"Presupuesto global usado:  {self.total_calls_this_run}/"
              f"{self.config.alpha_vantage_max_total_calls_per_run}")

        calls_for_ticker = 0

        while (
            state["pending"]
            and calls_for_ticker < ticker_budget
            and self.total_calls_this_run
            < self.config.alpha_vantage_max_total_calls_per_run
        ):
            task = state["pending"].pop(0)

            family = self._family_by_name(ticker, task["family"])
            block_start = pd.Timestamp(task["start"])
            block_end = pd.Timestamp(task["end"])

            print(
                f"\n[{ticker}] {family['name']}: "
                f"{block_start.date()} -> {block_end.date()}"
            )

            try:
                articles, quota_limited = self._fetch_task(
                    ticker=ticker,
                    family=family,
                    start=block_start,
                    end=block_end,
                    api_key=api_key,
                )

                calls_for_ticker += 1
                self.total_calls_this_run += 1

                if quota_limited:
                    # Reinsertar la tarea actual al principio para la próxima ejecución.
                    state["pending"].insert(0, task)
                    self._save_state(state_path, state)
                    print(
                        "Alpha Vantage ha alcanzado su cuota. "
                        "El proceso queda guardado y continuará en la siguiente ejecución."
                    )
                    break

                # Si estamos cerca del límite de resultados, la ventana puede
                # estar truncada. La dividimos antes de marcarla completa.
                if self._should_split(articles, block_start, block_end):
                    children = self._split_task(task)
                    state["pending"] = children + state["pending"]
                    self._save_state(state_path, state)
                    print(
                        f"Respuesta con {len(articles)} artículos: "
                        f"ventana dividida en {len(children)} subbloques."
                    )
                    # No añadimos todavía estos artículos para evitar sesgo por truncación.
                    continue

                if not articles.empty:
                    history = pd.concat(
                        [history, articles],
                        ignore_index=True,
                        sort=False,
                    )
                    history = self._normalize_history(history)
                    history = (
                        history.sort_values(
                            ["article_id", "raw_relevance_score"],
                            ascending=[True, False],
                        )
                        .drop_duplicates("article_id", keep="first")
                        .sort_values("published_at")
                        .reset_index(drop=True)
                    )
                    history.to_csv(
                        history_path,
                        index=False,
                        date_format="%Y-%m-%dT%H:%M:%S",
                    )

                state["completed"].append(task)
                self._save_state(state_path, state)

                if self.config.alpha_vantage_pause_seconds > 0:
                    time.sleep(self.config.alpha_vantage_pause_seconds)

            except Exception as exc:
                # No perder la tarea que falló.
                state["pending"].insert(0, task)
                self._save_state(state_path, state)
                print(
                    f"Error temporal Alpha Vantage ({type(exc).__name__}): {exc}"
                )
                break

        history = self._normalize_history(history)
        history = self._score_pending_with_finbert(history)

        if not history.empty:
            history = self._compute_final_relevance(ticker, history)
            history = history[
                history["relevance_score"]
                >= float(self.config.alpha_vantage_min_relevance)
            ].copy()

            history.to_csv(
                history_path,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%S",
            )
            self._build_daily_history(ticker, history)

        print("\nRESUMEN HISTÓRICO NLP")
        print(f"Artículos relevantes:        {len(history)}")
        print(f"Tareas pendientes:           {len(state['pending'])}")
        print(f"Tareas completadas:          {len(state['completed'])}")
        print(f"Llamadas {ticker} esta vez:  {calls_for_ticker}")
        print(f"Llamadas globales esta vez:  {self.total_calls_this_run}")

        return history if not history.empty else None

    def _load_or_initialize_state(
        self,
        ticker: str,
        path: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Dict[str, Any]:

        if os.path.exists(path):
            try:
                state = json.loads(
                    Path(path).read_text(encoding="utf-8")
                )
                if "pending" in state and "completed" in state:
                    return state
            except Exception:
                pass

        tasks = []
        initial_days = int(
            self.config.alpha_vantage_initial_chunk_days
        )

        for family in self.QUERY_FAMILIES[ticker]:
            current = start

            while current <= end:
                block_end = min(
                    current + pd.Timedelta(days=initial_days - 1),
                    end,
                )
                tasks.append({
                    "family": family["name"],
                    "start": current.strftime("%Y-%m-%d"),
                    "end": block_end.strftime("%Y-%m-%d"),
                })
                current = block_end + pd.Timedelta(days=1)

        # Prioridad: benchmark primero, después temas.
        tasks.sort(
            key=lambda x: (
                0 if x["family"].startswith("benchmark") else 1,
                x["start"],
                x["family"],
            )
        )

        state = {
            "ticker": ticker,
            "created_at": datetime.now().isoformat(),
            "pending": tasks,
            "completed": [],
        }
        self._save_state(path, state)
        return state

    @staticmethod
    def _save_state(path: str, state: Dict[str, Any]):
        Path(path).write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _family_by_name(
        self,
        ticker: str,
        name: str,
    ) -> Dict[str, Any]:
        for family in self.QUERY_FAMILIES[ticker]:
            if family["name"] == name:
                return family
        raise KeyError(
            f"Familia de consulta desconocida {ticker}/{name}"
        )

    def _fetch_task(
        self,
        ticker: str,
        family: Dict[str, Any],
        start: pd.Timestamp,
        end: pd.Timestamp,
        api_key: str,
    ) -> Tuple[pd.DataFrame, bool]:

        params = {
            "function": "NEWS_SENTIMENT",
            "time_from": start.strftime("%Y%m%dT0000"),
            "time_to": (
                end + pd.Timedelta(days=1)
            ).strftime("%Y%m%dT0000"),
            "sort": "EARLIEST",
            "limit": int(self.config.alpha_vantage_limit),
            "apikey": api_key,
        }

        if family.get("tickers"):
            params["tickers"] = family["tickers"]

        if family.get("topics"):
            params["topics"] = family["topics"]

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=self.config.alpha_vantage_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        if "Note" in payload or "Information" in payload:
            message = payload.get("Note") or payload.get("Information")
            print(f"Alpha Vantage: {message}")
            return pd.DataFrame(), True

        feed = payload.get("feed", [])
        rows = []

        for item in feed:
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            url = str(item.get("url") or "").strip()

            published = pd.to_datetime(
                str(item.get("time_published") or ""),
                format="%Y%m%dT%H%M%S",
                errors="coerce",
                utc=True,
            )

            if pd.isna(published):
                continue

            published = published.tz_convert(None)

            raw_id = (
                f"{title}|{url}|{published.isoformat()}"
            ).encode("utf-8", errors="ignore")
            article_id = hashlib.sha256(raw_id).hexdigest()[:24]

            ticker_relevance = self._max_ticker_relevance(
                ticker,
                item.get("ticker_sentiment", []),
            )

            keyword_score = self._keyword_score(
                ticker,
                f"{title}. {summary}",
            )

            query_weight = float(
                family.get("query_weight", 0.5)
            )

            raw_relevance = np.clip(
                0.45 * query_weight
                + 0.35 * ticker_relevance
                + 0.20 * keyword_score,
                0.0,
                1.0,
            )

            rows.append({
                "article_id": article_id,
                "published_at": published,
                "date": published.normalize(),
                "title": title,
                "summary": summary,
                "url": url,
                "source": str(item.get("source") or ""),
                "query_family": family["name"],
                "query_weight": query_weight,
                "ticker_relevance": ticker_relevance,
                "keyword_score": keyword_score,
                "raw_relevance_score": raw_relevance,
                "alpha_overall_sentiment": pd.to_numeric(
                    item.get("overall_sentiment_score"),
                    errors="coerce",
                ),
                "alpha_overall_label": str(
                    item.get("overall_sentiment_label") or ""
                ),
                "finbert_label": np.nan,
                "finbert_confidence": np.nan,
                "finbert_sentiment_score": np.nan,
                "relevance_score": np.nan,
            })

        return self._normalize_history(pd.DataFrame(rows)), False

    def _should_split(
        self,
        articles: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> bool:
        days = int((end - start).days) + 1

        if days <= int(self.config.alpha_vantage_min_chunk_days):
            return False

        threshold = int(
            self.config.alpha_vantage_limit
            * self.config.alpha_vantage_truncation_ratio
        )

        return len(articles) >= threshold

    @staticmethod
    def _split_task(task: Dict[str, str]) -> List[Dict[str, str]]:
        start = pd.Timestamp(task["start"])
        end = pd.Timestamp(task["end"])
        midpoint = start + (end - start) / 2
        midpoint = pd.Timestamp(midpoint.date())

        left = {
            "family": task["family"],
            "start": start.strftime("%Y-%m-%d"),
            "end": midpoint.strftime("%Y-%m-%d"),
        }
        right_start = midpoint + pd.Timedelta(days=1)
        right = {
            "family": task["family"],
            "start": right_start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        }

        return [left, right]

    def _max_ticker_relevance(
        self,
        ticker: str,
        ticker_sentiment: Any,
    ) -> float:

        relevant = self.RELEVANT_TICKERS[ticker]
        best = 0.0

        if not isinstance(ticker_sentiment, list):
            return best

        for row in ticker_sentiment:
            symbol = str(row.get("ticker") or "").upper()

            if symbol not in relevant:
                continue

            value = pd.to_numeric(
                row.get("relevance_score"),
                errors="coerce",
            )
            if pd.notna(value):
                best = max(best, float(value))

        return float(np.clip(best, 0.0, 1.0))

    def _keyword_score(
        self,
        ticker: str,
        text: str,
    ) -> float:
        normalized = str(text).lower()
        keywords = self.KEYWORDS[ticker]

        hits = sum(
            1 for keyword in keywords
            if keyword.lower() in normalized
        )

        # 4 coincidencias ya se consideran relevancia léxica máxima.
        return float(min(hits / 4.0, 1.0))

    def _compute_final_relevance(
        self,
        ticker: str,
        history: pd.DataFrame,
    ) -> pd.DataFrame:

        out = history.copy()

        out["query_weight"] = pd.to_numeric(
            out["query_weight"],
            errors="coerce",
        ).fillna(0.5)

        out["ticker_relevance"] = pd.to_numeric(
            out["ticker_relevance"],
            errors="coerce",
        ).fillna(0.0)

        out["keyword_score"] = pd.to_numeric(
            out["keyword_score"],
            errors="coerce",
        ).fillna(0.0)

        out["finbert_confidence"] = pd.to_numeric(
            out["finbert_confidence"],
            errors="coerce",
        ).fillna(0.0)

        # La confianza de FinBERT se usa para la ponderación del sentimiento
        # diario, pero no debe eliminar una noticia relevante solo por ser
        # clasificada como neutral.
        out["relevance_score"] = np.clip(
            0.45 * out["query_weight"]
            + 0.35 * out["ticker_relevance"]
            + 0.20 * out["keyword_score"],
            0.0,
            1.0,
        )

        return out

    @staticmethod
    def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "article_id",
            "published_at",
            "date",
            "title",
            "summary",
            "url",
            "source",
            "query_family",
            "query_weight",
            "ticker_relevance",
            "keyword_score",
            "raw_relevance_score",
            "alpha_overall_sentiment",
            "alpha_overall_label",
            "finbert_label",
            "finbert_confidence",
            "finbert_sentiment_score",
            "relevance_score",
        ]

        if df is None or df.empty:
            return pd.DataFrame(columns=columns)

        out = df.copy()

        for col in columns:
            if col not in out.columns:
                out[col] = np.nan

        out["published_at"] = pd.to_datetime(
            out["published_at"],
            errors="coerce",
            format="mixed",
            utc=True,
        ).dt.tz_convert(None)

        out["date"] = pd.to_datetime(
            out["date"],
            errors="coerce",
            format="mixed",
            utc=True,
        ).dt.tz_convert(None).dt.normalize()

        numeric_cols = [
            "query_weight",
            "ticker_relevance",
            "keyword_score",
            "raw_relevance_score",
            "alpha_overall_sentiment",
            "finbert_confidence",
            "finbert_sentiment_score",
            "relevance_score",
        ]

        for col in numeric_cols:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

        return (
            out.dropna(
                subset=["article_id", "published_at"]
            )[columns]
            .reset_index(drop=True)
        )

    def _load_history(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return self._normalize_history(pd.DataFrame())

        try:
            return self._normalize_history(pd.read_csv(path))
        except Exception as exc:
            print(
                f"No se pudo leer el histórico Alpha Vantage: {exc}"
            )
            return self._normalize_history(pd.DataFrame())

    def _score_pending_with_finbert(
        self,
        history: pd.DataFrame,
    ) -> pd.DataFrame:

        if history.empty:
            return history

        if self.sentiment_analyzer.finbert is None:
            raise RuntimeError(
                "FinBERT no está disponible para puntuar el histórico."
            )

        pending_idx = history.index[
            history["finbert_sentiment_score"].isna()
        ].tolist()

        if not pending_idx:
            print("FinBERT histórico: no hay artículos pendientes.")
            return history

        print(
            f"FinBERT histórico: {len(pending_idx)} artículos pendientes."
        )

        batch_size = 16

        for start_idx in range(
            0,
            len(pending_idx),
            batch_size,
        ):
            batch_indices = pending_idx[
                start_idx:start_idx + batch_size
            ]

            texts = [
                (
                    f"{history.at[idx, 'title']}. "
                    f"{history.at[idx, 'summary']}"
                )[:1200]
                for idx in batch_indices
            ]

            try:
                results = self.sentiment_analyzer.finbert(
                    texts,
                    batch_size=batch_size,
                    truncation=True,
                )
            except TypeError:
                results = self.sentiment_analyzer.finbert(texts)

            for idx, result in zip(
                batch_indices,
                results,
            ):
                if result and isinstance(result[0], list):
                    result = result[0]

                best = max(
                    result,
                    key=lambda x: x["score"],
                )

                label = str(best["label"]).lower()
                confidence = float(best["score"])

                if "positive" in label:
                    score = confidence
                elif "negative" in label:
                    score = -confidence
                else:
                    score = 0.0

                history.at[
                    idx, "finbert_label"
                ] = label
                history.at[
                    idx, "finbert_confidence"
                ] = confidence
                history.at[
                    idx, "finbert_sentiment_score"
                ] = score

        return history

    def _build_daily_history(
        self,
        ticker: str,
        history: pd.DataFrame,
    ):

        if history.empty:
            return

        scored = history.dropna(
            subset=[
                "date",
                "finbert_sentiment_score",
                "relevance_score",
            ]
        ).copy()

        if scored.empty:
            return

        scored["date"] = pd.to_datetime(
            scored["date"]
        ).dt.normalize()

        scored["finbert_confidence"] = pd.to_numeric(
            scored["finbert_confidence"],
            errors="coerce",
        ).fillna(0.0)

        scored["article_weight"] = (
            scored["relevance_score"]
            * np.maximum(
                scored["finbert_confidence"],
                0.25,
            )
        )

        scored["weighted_sentiment"] = (
            scored["finbert_sentiment_score"]
            * scored["article_weight"]
        )

        scored["positive_weight"] = np.where(
            scored["finbert_sentiment_score"] > 0,
            scored["article_weight"],
            0.0,
        )
        scored["negative_weight"] = np.where(
            scored["finbert_sentiment_score"] < 0,
            scored["article_weight"],
            0.0,
        )

        rows = []

        for date, group in scored.groupby("date"):
            total_weight = float(
                group["article_weight"].sum()
            )

            if total_weight <= 0:
                sentiment_mean = 0.0
                positive_ratio = 0.0
                negative_ratio = 0.0
            else:
                sentiment_mean = float(
                    group["weighted_sentiment"].sum()
                    / total_weight
                )
                positive_ratio = float(
                    group["positive_weight"].sum()
                    / total_weight
                )
                negative_ratio = float(
                    group["negative_weight"].sum()
                    / total_weight
                )

            rows.append({
                "date": date,
                "sentiment_mean": sentiment_mean,
                "sentiment_positive_ratio": positive_ratio,
                "sentiment_negative_ratio": negative_ratio,
                "news_count": int(len(group)),
                "relevance_weight_sum": total_weight,
                "relevance_mean": float(
                    group["relevance_score"].mean()
                ),
                "sentiment_available": 1.0,
                "sentiment_source": "FINBERT_ALPHA_MULTIFAMILY",
            })

        daily = pd.DataFrame(rows).sort_values("date")

        out_path = os.path.join(
            self.config.historical_sentiment_dir,
            f"{ticker.lower()}_sentiment.csv",
        )

        # Google News reciente tiene prioridad en fechas coincidentes.
        if os.path.exists(out_path):
            try:
                old = pd.read_csv(out_path)
                old["date"] = pd.to_datetime(
                    old["date"],
                    errors="coerce",
                )

                if "sentiment_source" not in old.columns:
                    old["sentiment_source"] = (
                        "FINBERT_GOOGLE_NEWS"
                    )

                combined = pd.concat(
                    [daily, old],
                    ignore_index=True,
                    sort=False,
                )

                combined["_priority"] = combined[
                    "sentiment_source"
                ].map({
                    "GDELT_TONE": 1,
                    "FINBERT_ALPHA_VANTAGE": 2,
                    "FINBERT_ALPHA_MULTIFAMILY": 3,
                    "FINBERT_GOOGLE_NEWS": 4,
                }).fillna(1)

                combined = (
                    combined.dropna(subset=["date"])
                    .sort_values(
                        ["date", "_priority"]
                    )
                    .drop_duplicates(
                        "date",
                        keep="last",
                    )
                    .drop(columns="_priority")
                    .sort_values("date")
                )
            except Exception:
                combined = daily.copy()
        else:
            combined = daily.copy()

        combined.to_csv(
            out_path,
            index=False,
            date_format="%Y-%m-%d",
        )

        print(
            f"Histórico diario FinBERT: {out_path} "
            f"({len(daily)} días Alpha multifamilia)."
        )


# =============================================================================
# SEGUNDA FUENTE HISTÓRICA: THE GUARDIAN OPEN PLATFORM + FINBERT
# =============================================================================

class GuardianHistoricalNewsBackfiller:
    """
    Segunda fuente histórica independiente de Alpha Vantage.

    The Guardian Open Platform permite búsquedas por texto, filtros de fecha,
    paginación y recuperación de campos como headline/trailText. El proceso es
    reanudable y divide ventanas demasiado densas antes de aceptar truncación.
    """

    BASE_URL = "https://content.guardianapis.com/search"

    QUERY_FAMILIES = {
        "SXR8": [
            {"name": "sp500_wallstreet", "q": '"S&P 500" OR "Wall Street"', "weight": 1.00},
            {"name": "fed_markets", "q": '"Federal Reserve" AND (markets OR stocks OR rates)', "weight": 0.92},
            {"name": "inflation_macro", "q": '(inflation OR CPI OR recession OR "US economy") AND (markets OR stocks)', "weight": 0.82},
            {"name": "earnings_market", "q": '(earnings OR profits) AND ("S&P 500" OR "US stocks")', "weight": 0.78},
        ],
        "SXRV": [
            {"name": "nasdaq", "q": 'Nasdaq OR "Nasdaq 100"', "weight": 1.00},
            {"name": "big_tech", "q": '(Nvidia OR Apple OR Microsoft OR Amazon OR Meta OR Alphabet) AND (technology OR stocks OR earnings)', "weight": 0.95},
            {"name": "semis_ai", "q": '(semiconductors OR chips OR "artificial intelligence" OR AI) AND (stocks OR market OR earnings)', "weight": 0.90},
            {"name": "fed_tech", "q": '"Federal Reserve" AND (technology OR Nasdaq OR stocks OR rates)', "weight": 0.82},
        ],
    }

    KEYWORDS = AlphaVantageHistoricalNewsBackfiller.KEYWORDS

    def __init__(self, config: Config, sentiment_analyzer: NewsSentimentAnalyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.total_calls_this_run = 0

    def update(self, ticker: str, market_dates: pd.Series) -> Optional[pd.DataFrame]:
        if not self.config.use_guardian_history:
            return None

        ticker = ticker.upper()
        history_path = os.path.join(
            self.config.guardian_history_dir,
            f"{ticker.lower()}_articles.csv",
        )
        state_path = os.path.join(
            self.config.guardian_history_dir,
            f"{ticker.lower()}_backfill_state.json",
        )

        history = self._load_history(history_path)
        api_key = self.config.guardian_api_key

        if not api_key:
            if history.empty and self.config.require_guardian_history:
                raise RuntimeError(
                    "Falta GUARDIAN_API_KEY y no existe histórico Guardian local."
                )
            return history if not history.empty else None

        start = max(
            pd.Timestamp(self.config.guardian_history_start),
            pd.Timestamp(self.config.start_date),
        ).normalize()
        end = min(
            pd.Timestamp(self.config.end_date),
            pd.Series(pd.to_datetime(market_dates)).max(),
        ).normalize()

        state = self._load_or_initialize_state(ticker, state_path, start, end)
        budget = min(
            int(self.config.guardian_max_calls_per_ticker_per_run),
            int(self.config.guardian_max_total_calls_per_run) - self.total_calls_this_run,
        )
        budget = max(budget, 0)

        print("\n" + "=" * 108)
        print(f"HISTÓRICO THE GUARDIAN + FINBERT: {ticker}")
        print("=" * 108)
        print(f"Periodo:                   {start.date()} -> {end.date()}")
        print(f"Familias de consulta:      {len(self.QUERY_FAMILIES[ticker])}")
        print(f"Tareas pendientes:         {len(state['pending'])}")
        print(f"Tareas completadas:        {len(state['completed'])}")
        print(f"Presupuesto esta ejecución:{budget} llamadas")

        calls = 0
        while (
            state["pending"]
            and calls < budget
            and self.total_calls_this_run < self.config.guardian_max_total_calls_per_run
        ):
            task = state["pending"].pop(0)
            family = self._family(ticker, task["family"])
            start_t = pd.Timestamp(task["start"])
            end_t = pd.Timestamp(task["end"])
            page = int(task.get("page", 1))

            print(
                f"\n[Guardian {ticker}] {family['name']} "
                f"{start_t.date()} -> {end_t.date()} | page={page}"
            )

            try:
                articles, pages_total = self._fetch_page(
                    ticker=ticker,
                    family=family,
                    start=start_t,
                    end=end_t,
                    page=page,
                    api_key=api_key,
                )
                calls += 1
                self.total_calls_this_run += 1

                days = int((end_t - start_t).days) + 1
                if (
                    page == 1
                    and pages_total > self.config.guardian_max_pages_per_window
                    and days > self.config.guardian_min_chunk_days
                ):
                    children = self._split_task(task)
                    state["pending"] = children + state["pending"]
                    self._save_state(state_path, state)
                    print(
                        f"Ventana demasiado densa ({pages_total} páginas): "
                        "se divide para evitar truncación."
                    )
                    continue

                if not articles.empty:
                    history = pd.concat([history, articles], ignore_index=True, sort=False)
                    history = self._normalize_history(history)
                    history = (
                        history.sort_values(
                            ["dedupe_key", "relevance_score"],
                            ascending=[True, False],
                        )
                        .drop_duplicates("dedupe_key", keep="first")
                        .sort_values("published_at")
                        .reset_index(drop=True)
                    )
                    history.to_csv(
                        history_path,
                        index=False,
                        date_format="%Y-%m-%dT%H:%M:%S",
                    )

                # Si es la primera página y el número total cabe en el límite,
                # programamos las páginas restantes como tareas reanudables.
                if page == 1 and pages_total > 1:
                    max_page = min(
                        int(pages_total),
                        int(self.config.guardian_max_pages_per_window),
                    )
                    extra = []
                    for p in range(2, max_page + 1):
                        extra.append({
                            "family": task["family"],
                            "start": task["start"],
                            "end": task["end"],
                            "page": p,
                        })
                    state["pending"] = extra + state["pending"]

                state["completed"].append(task)
                self._save_state(state_path, state)

                if self.config.guardian_pause_seconds > 0:
                    time.sleep(self.config.guardian_pause_seconds)

            except Exception as exc:
                state["pending"].insert(0, task)
                self._save_state(state_path, state)
                print(f"Guardian error temporal ({type(exc).__name__}): {exc}")
                break

        history = self._normalize_history(history)
        history = self._score_pending(history)
        if not history.empty:
            history = history[
                history["relevance_score"] >= self.config.guardian_min_relevance
            ].copy()
            history.to_csv(
                history_path,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%S",
            )

        print("\nRESUMEN GUARDIAN")
        print(f"Artículos relevantes:  {len(history)}")
        print(f"Tareas pendientes:     {len(state['pending'])}")
        print(f"Tareas completadas:    {len(state['completed'])}")
        print(f"Llamadas esta vez:     {calls}")
        return history if not history.empty else None

    def _load_or_initialize_state(self, ticker, path, start, end):
        if os.path.exists(path):
            try:
                state = json.loads(Path(path).read_text(encoding="utf-8"))
                if "pending" in state and "completed" in state:
                    return state
            except Exception:
                pass

        tasks = []
        chunk = int(self.config.guardian_initial_chunk_days)
        for family in self.QUERY_FAMILIES[ticker]:
            current = start
            while current <= end:
                block_end = min(current + pd.Timedelta(days=chunk - 1), end)
                tasks.append({
                    "family": family["name"],
                    "start": current.strftime("%Y-%m-%d"),
                    "end": block_end.strftime("%Y-%m-%d"),
                    "page": 1,
                })
                current = block_end + pd.Timedelta(days=1)

        state = {
            "ticker": ticker,
            "created_at": datetime.now().isoformat(),
            "pending": tasks,
            "completed": [],
        }
        self._save_state(path, state)
        return state

    @staticmethod
    def _save_state(path, state):
        Path(path).write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _family(self, ticker, name):
        for family in self.QUERY_FAMILIES[ticker]:
            if family["name"] == name:
                return family
        raise KeyError(name)

    def _fetch_page(self, ticker, family, start, end, page, api_key):
        params = {
            "api-key": api_key,
            "format": "json",
            "q": family["q"],
            "from-date": start.strftime("%Y-%m-%d"),
            "to-date": end.strftime("%Y-%m-%d"),
            "use-date": "published",
            "order-by": "oldest",
            "page-size": int(self.config.guardian_page_size),
            "page": int(page),
            "show-fields": "headline,trailText,shortUrl",
        }
        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=self.config.guardian_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json().get("response", {})
        if payload.get("status") != "ok":
            raise RuntimeError(f"Guardian status={payload.get('status')}")

        pages_total = int(payload.get("pages", 1) or 1)
        rows = []
        for item in payload.get("results", []):
            published = pd.to_datetime(
                item.get("webPublicationDate"),
                errors="coerce",
                utc=True,
            )
            if pd.isna(published):
                continue
            published = published.tz_convert(None)

            fields = item.get("fields") or {}
            title = str(fields.get("headline") or item.get("webTitle") or "").strip()
            summary_html = str(fields.get("trailText") or "")
            summary = unescape(re.sub(r"<[^>]+>", " ", summary_html))
            summary = re.sub(r"\\s+", " ", summary).strip()
            url = str(fields.get("shortUrl") or item.get("webUrl") or "").strip()
            section = str(item.get("sectionName") or "")

            keyword_score = self._keyword_score(ticker, f"{title}. {summary}")
            section_bonus = 1.0 if section.lower() in {"business", "technology"} else 0.5
            relevance = float(np.clip(
                0.65 * float(family["weight"])
                + 0.25 * keyword_score
                + 0.10 * section_bonus,
                0.0,
                1.0,
            ))

            normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            dedupe_raw = f"{published.date()}|{normalized_title}".encode("utf-8")
            dedupe_key = hashlib.sha256(dedupe_raw).hexdigest()[:24]
            article_id = str(item.get("id") or dedupe_key)

            rows.append({
                "article_id": article_id,
                "dedupe_key": dedupe_key,
                "published_at": published,
                "date": published.normalize(),
                "title": title,
                "summary": summary,
                "url": url,
                "publisher": "The Guardian",
                "section": section,
                "query_family": family["name"],
                "query_weight": float(family["weight"]),
                "keyword_score": keyword_score,
                "relevance_score": relevance,
                "finbert_label": np.nan,
                "finbert_confidence": np.nan,
                "finbert_sentiment_score": np.nan,
            })

        return self._normalize_history(pd.DataFrame(rows)), pages_total

    def _keyword_score(self, ticker, text):
        normalized = str(text).lower()
        hits = sum(1 for kw in self.KEYWORDS[ticker] if kw.lower() in normalized)
        return float(min(hits / 4.0, 1.0))

    @staticmethod
    def _split_task(task):
        start = pd.Timestamp(task["start"])
        end = pd.Timestamp(task["end"])
        mid = start + (end - start) / 2
        mid = pd.Timestamp(mid.date())
        return [
            {
                "family": task["family"],
                "start": start.strftime("%Y-%m-%d"),
                "end": mid.strftime("%Y-%m-%d"),
                "page": 1,
            },
            {
                "family": task["family"],
                "start": (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "page": 1,
            },
        ]

    @staticmethod
    def _normalize_history(df):
        columns = [
            "article_id", "dedupe_key", "published_at", "date", "title",
            "summary", "url", "publisher", "section", "query_family",
            "query_weight", "keyword_score", "relevance_score",
            "finbert_label", "finbert_confidence", "finbert_sentiment_score",
        ]
        if df is None or df.empty:
            return pd.DataFrame(columns=columns)
        out = df.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = np.nan
        out["published_at"] = pd.to_datetime(
            out["published_at"], errors="coerce", format="mixed", utc=True
        ).dt.tz_convert(None)
        out["date"] = pd.to_datetime(
            out["date"], errors="coerce", format="mixed", utc=True
        ).dt.tz_convert(None).dt.normalize()
        for col in ["query_weight", "keyword_score", "relevance_score", "finbert_confidence", "finbert_sentiment_score"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.dropna(subset=["article_id", "published_at"])[columns].reset_index(drop=True)

    def _load_history(self, path):
        if not os.path.exists(path):
            return self._normalize_history(pd.DataFrame())
        try:
            return self._normalize_history(pd.read_csv(path))
        except Exception as exc:
            print(f"Guardian histórico no legible: {exc}")
            return self._normalize_history(pd.DataFrame())

    def _score_pending(self, history):
        if history.empty:
            return history
        if self.sentiment_analyzer.finbert is None:
            raise RuntimeError("FinBERT no disponible para Guardian.")

        pending = history.index[history["finbert_sentiment_score"].isna()].tolist()
        if not pending:
            print("FinBERT Guardian: no hay artículos pendientes.")
            return history

        print(f"FinBERT Guardian: {len(pending)} artículos pendientes.")
        batch_size = 16
        for start in range(0, len(pending), batch_size):
            idxs = pending[start:start + batch_size]
            texts = [
                f"{history.at[i, 'title']}. {history.at[i, 'summary']}"[:1200]
                for i in idxs
            ]
            try:
                results = self.sentiment_analyzer.finbert(
                    texts, batch_size=batch_size, truncation=True
                )
            except TypeError:
                results = self.sentiment_analyzer.finbert(texts)

            for i, result in zip(idxs, results):
                if result and isinstance(result[0], list):
                    result = result[0]
                best = max(result, key=lambda x: x["score"])
                label = str(best["label"]).lower()
                conf = float(best["score"])
                score = conf if "positive" in label else (-conf if "negative" in label else 0.0)
                history.at[i, "finbert_label"] = label
                history.at[i, "finbert_confidence"] = conf
                history.at[i, "finbert_sentiment_score"] = score
        return history


class MultiSourceHistoricalSentimentBuilder:
    """Fusiona Alpha Vantage + Guardian a nivel de artículo y crea el diario."""

    def __init__(self, config: Config):
        self.config = config

    def build(self, ticker: str):
        ticker = ticker.upper()
        frames = []

        alpha_path = os.path.join(
            self.config.alpha_vantage_history_dir,
            f"{ticker.lower()}_articles.csv",
        )
        if os.path.exists(alpha_path):
            try:
                a = pd.read_csv(alpha_path)
                if not a.empty:
                    a["source_system"] = "ALPHA_VANTAGE"
                    a["published_at"] = pd.to_datetime(a["published_at"], errors="coerce", format="mixed", utc=True).dt.tz_convert(None)
                    a["date"] = pd.to_datetime(a.get("date", a["published_at"]), errors="coerce", format="mixed", utc=True).dt.tz_convert(None).dt.normalize()
                    a["title"] = a.get("title", "").fillna("").astype(str)
                    a["summary"] = a.get("summary", "").fillna("").astype(str)
                    a["relevance_score"] = pd.to_numeric(a.get("relevance_score", a.get("raw_relevance_score", 0.5)), errors="coerce").fillna(0.5)
                    a["finbert_confidence"] = pd.to_numeric(a.get("finbert_confidence"), errors="coerce")
                    a["finbert_sentiment_score"] = pd.to_numeric(a.get("finbert_sentiment_score"), errors="coerce")
                    frames.append(a[["published_at", "date", "title", "summary", "relevance_score", "finbert_confidence", "finbert_sentiment_score", "source_system"]])
            except Exception as exc:
                print(f"No se pudo combinar Alpha {ticker}: {exc}")

        guardian_path = os.path.join(
            self.config.guardian_history_dir,
            f"{ticker.lower()}_articles.csv",
        )
        if os.path.exists(guardian_path):
            try:
                g = pd.read_csv(guardian_path)
                if not g.empty:
                    g["source_system"] = "GUARDIAN"
                    g["published_at"] = pd.to_datetime(g["published_at"], errors="coerce", format="mixed", utc=True).dt.tz_convert(None)
                    g["date"] = pd.to_datetime(g.get("date", g["published_at"]), errors="coerce", format="mixed", utc=True).dt.tz_convert(None).dt.normalize()
                    g["title"] = g.get("title", "").fillna("").astype(str)
                    g["summary"] = g.get("summary", "").fillna("").astype(str)
                    g["relevance_score"] = pd.to_numeric(g.get("relevance_score", 0.5), errors="coerce").fillna(0.5)
                    g["finbert_confidence"] = pd.to_numeric(g.get("finbert_confidence"), errors="coerce")
                    g["finbert_sentiment_score"] = pd.to_numeric(g.get("finbert_sentiment_score"), errors="coerce")
                    frames.append(g[["published_at", "date", "title", "summary", "relevance_score", "finbert_confidence", "finbert_sentiment_score", "source_system"]])
            except Exception as exc:
                print(f"No se pudo combinar Guardian {ticker}: {exc}")

        if not frames:
            print(f"Histórico multifuente {ticker}: sin artículos disponibles.")
            return None

        articles = pd.concat(frames, ignore_index=True, sort=False)
        articles = articles.dropna(subset=["date", "finbert_sentiment_score"]).copy()
        if articles.empty:
            return None

        # Deduplicación entre proveedores por fecha + título normalizado.
        normalized_title = (
            articles["title"].str.lower()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
            .str.strip()
        )
        articles["cross_source_key"] = [
            hashlib.sha256(f"{d.date()}|{t}".encode("utf-8")).hexdigest()[:24]
            for d, t in zip(articles["date"], normalized_title)
        ]
        articles["quality"] = (
            articles["relevance_score"].fillna(0.0)
            * articles["finbert_confidence"].fillna(0.25).clip(lower=0.25)
        )
        articles = (
            articles.sort_values(["cross_source_key", "quality"], ascending=[True, False])
            .drop_duplicates("cross_source_key", keep="first")
            .sort_values("published_at")
        )

        articles["weight"] = (
            articles["relevance_score"].fillna(0.0)
            * articles["finbert_confidence"].fillna(0.25).clip(lower=0.25)
        )
        articles["weighted_sentiment"] = articles["finbert_sentiment_score"] * articles["weight"]
        articles["positive_weight"] = np.where(articles["finbert_sentiment_score"] > 0, articles["weight"], 0.0)
        articles["negative_weight"] = np.where(articles["finbert_sentiment_score"] < 0, articles["weight"], 0.0)

        rows = []
        for date, group in articles.groupby("date"):
            weight = float(group["weight"].sum())
            if weight <= 0:
                continue
            rows.append({
                "date": date,
                "sentiment_mean": float(group["weighted_sentiment"].sum() / weight),
                "sentiment_positive_ratio": float(group["positive_weight"].sum() / weight),
                "sentiment_negative_ratio": float(group["negative_weight"].sum() / weight),
                "news_count": int(len(group)),
                "relevance_weight_sum": weight,
                "sentiment_available": 1.0,
                "sentiment_source": "FINBERT_ALPHA_GUARDIAN",
                "n_alpha": int((group["source_system"] == "ALPHA_VANTAGE").sum()),
                "n_guardian": int((group["source_system"] == "GUARDIAN").sum()),
            })

        daily = pd.DataFrame(rows).sort_values("date")
        path = os.path.join(
            self.config.historical_sentiment_dir,
            f"{ticker.lower()}_sentiment.csv",
        )
        daily.to_csv(path, index=False, date_format="%Y-%m-%d")

        merged_articles_path = os.path.join(
            self.config.output_dir,
            ticker,
            "historical_news_multisource.csv",
        )
        os.makedirs(os.path.dirname(merged_articles_path), exist_ok=True)
        articles.to_csv(merged_articles_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")

        print(
            f"Histórico multifuente {ticker}: {len(articles)} artículos únicos, "
            f"{len(daily)} días reales."
        )
        return daily


# =============================================================================
# 8. VARIABLES EXTERNAS: FRED
# =============================================================================

class ExternalMarketDataLoader:
    """
    Descarga y cachea variables externas desde FRED sin necesidad de API key.

    Series:
        SP500      S&P 500
        NASDAQ100  Nasdaq 100
        VIXCLS     CBOE VIX
        DEXUSEU    EUR/USD (USD por EUR)
        DGS10      Treasury 10Y
        DGS2       Treasury 2Y
        DFF        Effective Federal Funds Rate

    Para evitar look-ahead bias, las features finales se desplazan
    `external_feature_lag_sessions` sesiones antes de fusionarse con el ETF.
    """

    FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    FEATURE_COLUMNS = [
        "ext_sp500_return_1d",
        "ext_sp500_return_5d",
        "ext_nasdaq100_return_1d",
        "ext_nasdaq100_return_5d",
        "ext_vix_level",
        "ext_vix_change_1d",
        "ext_vix_change_5d",
        "ext_eurusd_return_1d",
        "ext_eurusd_return_5d",
        "ext_us10y_level",
        "ext_us10y_change_1d",
        "ext_us2y_level",
        "ext_us2y_change_1d",
        "ext_yield_curve_10y2y",
        "ext_fedfunds_level",
        "ext_fedfunds_change_5d",
    ]

    def __init__(self, config: Config):
        self.config = config

    def build(self, market_dates: pd.Series) -> Optional[pd.DataFrame]:
        if not self.config.use_external_features:
            return None

        dates = pd.Series(pd.to_datetime(market_dates)).dropna().sort_values()
        if dates.empty:
            return None

        start = dates.min() - pd.Timedelta(days=120)
        end = dates.max() + pd.Timedelta(days=3)

        raw = {}
        failures = []

        print("\nVARIABLES EXTERNAS FRED")
        print("-" * 90)

        for alias, fred_id in self.config.fred_series.items():
            try:
                series = self._load_series(alias, fred_id, start, end)
                raw[alias] = series
                print(
                    f"{alias:15s} {fred_id:12s} OK "
                    f"({series.notna().sum()} observaciones)"
                )
            except Exception as exc:
                failures.append((alias, fred_id, str(exc)))
                print(f"{alias:15s} {fred_id:12s} ERROR: {exc}")

        if failures and self.config.require_external_features:
            missing = ", ".join(f"{a}/{i}" for a, i, _ in failures)
            raise RuntimeError(
                "No se han podido obtener todas las variables externas "
                f"obligatorias: {missing}"
            )

        if not raw:
            return None

        # Calendario diario para poder forward-fill fines de semana y días
        # sin observación. No interpolamos hacia atrás.
        full_index = pd.date_range(start.normalize(), end.normalize(), freq="D")
        ext = pd.DataFrame(index=full_index)

        for alias, s in raw.items():
            ext[alias] = s.reindex(full_index)

        # Límite de forward fill razonable: evita propagar datos demasiado
        # antiguos ante una interrupción prolongada de la fuente.
        ext = ext.ffill(limit=10)

        # Features de índices y FX.
        if "sp500" in ext:
            ext["ext_sp500_return_1d"] = ext["sp500"].pct_change(1)
            ext["ext_sp500_return_5d"] = ext["sp500"].pct_change(5)

        if "nasdaq100" in ext:
            ext["ext_nasdaq100_return_1d"] = ext["nasdaq100"].pct_change(1)
            ext["ext_nasdaq100_return_5d"] = ext["nasdaq100"].pct_change(5)

        if "vix" in ext:
            ext["ext_vix_level"] = ext["vix"]
            ext["ext_vix_change_1d"] = ext["vix"].pct_change(1)
            ext["ext_vix_change_5d"] = ext["vix"].pct_change(5)

        if "eurusd" in ext:
            ext["ext_eurusd_return_1d"] = ext["eurusd"].pct_change(1)
            ext["ext_eurusd_return_5d"] = ext["eurusd"].pct_change(5)

        # Tipos de interés: cambios en puntos porcentuales, no pct_change.
        if "us10y" in ext:
            ext["ext_us10y_level"] = ext["us10y"]
            ext["ext_us10y_change_1d"] = ext["us10y"].diff(1)

        if "us2y" in ext:
            ext["ext_us2y_level"] = ext["us2y"]
            ext["ext_us2y_change_1d"] = ext["us2y"].diff(1)

        if "us10y" in ext and "us2y" in ext:
            ext["ext_yield_curve_10y2y"] = ext["us10y"] - ext["us2y"]

        if "fedfunds" in ext:
            ext["ext_fedfunds_level"] = ext["fedfunds"]
            ext["ext_fedfunds_change_5d"] = ext["fedfunds"].diff(5)

        # Garantizar todas las columnas previstas.
        for col in self.FEATURE_COLUMNS:
            if col not in ext.columns:
                ext[col] = np.nan

        feature_df = ext[self.FEATURE_COLUMNS].copy()

        # Anti-look-ahead: las variables del día t solo se utilizan desde
        # t + lag. Por defecto lag=1.
        lag = max(int(self.config.external_feature_lag_sessions), 0)
        if lag:
            feature_df = feature_df.shift(lag)

        feature_df = (
            feature_df.replace([np.inf, -np.inf], np.nan)
            .reset_index()
            .rename(columns={"index": "date"})
        )

        target_dates = pd.DataFrame(
            {"date": pd.Series(pd.to_datetime(market_dates)).dt.normalize()}
        ).drop_duplicates()

        merged = target_dates.merge(feature_df, on="date", how="left")
        coverage = merged[self.FEATURE_COLUMNS].notna().mean().mean()

        out_path = os.path.join(
            self.config.external_data_dir,
            "external_features_merged.csv",
        )
        feature_df.to_csv(out_path, index=False, date_format="%Y-%m-%d")

        print("-" * 90)
        print(f"Cobertura media de features externas: {coverage:.1%}")
        print(f"Lag anti-look-ahead: {lag} sesión/sesiones")
        print(f"Cache features: {out_path}")
        print("-" * 90)

        return feature_df

    def _load_series(
        self,
        alias: str,
        fred_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        cache_path = os.path.join(
            self.config.external_data_dir,
            f"fred_{fred_id}.csv",
        )

        params = {
            "id": fred_id,
            "cosd": start.strftime("%Y-%m-%d"),
            "coed": end.strftime("%Y-%m-%d"),
        }

        try:
            response = requests.get(
                self.FRED_CSV_URL,
                params=params,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0 TFM-ETF-Research/1.0"},
            )
            response.raise_for_status()
            cache_path_tmp = cache_path + ".tmp"
            with open(cache_path_tmp, "wb") as f:
                f.write(response.content)

            df = pd.read_csv(cache_path_tmp)
            os.replace(cache_path_tmp, cache_path)
        except Exception as exc:
            if not os.path.exists(cache_path):
                raise RuntimeError(
                    f"FRED {fred_id} no disponible y no existe caché: {exc}"
                )
            df = pd.read_csv(cache_path)
            print(f"  {fred_id}: usando caché local por fallo de red.")

        date_col = None
        for candidate in ["DATE", "observation_date", "date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            date_col = df.columns[0]

        value_cols = [c for c in df.columns if c != date_col]
        if not value_cols:
            raise ValueError(f"FRED {fred_id}: no hay columna de valores.")

        value_col = fred_id if fred_id in df.columns else value_cols[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        s = df.set_index(date_col)[value_col]
        s.index = pd.DatetimeIndex(s.index).normalize()
        s.name = alias
        return s


# =============================================================================
# 9. FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    SENTIMENT_WINDOWS = [1, 5, 10, 15, 20, 30, 45, 60]

    SENTIMENT_FEATURES = []
    for _w in SENTIMENT_WINDOWS:
        SENTIMENT_FEATURES.extend([
            f"sentiment_mean_{_w}d",
            f"sentiment_positive_ratio_{_w}d",
            f"sentiment_negative_ratio_{_w}d",
            f"news_count_{_w}d",
            f"news_count_log_{_w}d",
            f"sentiment_available_{_w}d",
        ])

    SENTIMENT_FEATURES.extend([
        "sentiment_momentum_5_20",
        "sentiment_momentum_10_30",
        "sentiment_momentum_15_45",
        "sentiment_momentum_20_60",
    ])


    EXTERNAL_FEATURES = list(ExternalMarketDataLoader.FEATURE_COLUMNS)

    TECHNICAL_WINDOWS = [5, 10, 15, 20, 30, 45, 60]

    TECHNICAL_FEATURES = [
        "return_1d", "return_2d", "return_3d",
        *[f"return_{w}d" for w in TECHNICAL_WINDOWS],
        "intraday_return", "range_pct", "gap_return",
        *[f"volatility_{w}d" for w in TECHNICAL_WINDOWS],
        *[f"close_vs_ma_{w}" for w in TECHNICAL_WINDOWS],
        "close_vs_ma_50", "close_vs_ma_100", "close_vs_ma_200",
        "ma_20_vs_50", "ma_50_vs_200",
        "rsi_14", "rsi_28",
        "macd", "macd_signal", "macd_diff",
        "bollinger_position",
        "volume_change_1d", "volume_vs_ma20",
        "day_of_week", "month",
    ]


    @staticmethod
    def build(
        market_df: pd.DataFrame,
        historical_sentiment: Optional[pd.DataFrame] = None,
        external_features: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        df = market_df.copy().sort_values("date").reset_index(drop=True)

        for p in [1, 2, 3] + FeatureEngineer.TECHNICAL_WINDOWS:
            df[f"return_{p}d"] = df["Close"].pct_change(p)

        df["intraday_return"] = df["Close"] / df["Open"] - 1
        df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
        df["gap_return"] = df["Open"] / df["Close"].shift(1) - 1

        for p in FeatureEngineer.TECHNICAL_WINDOWS:
            df[f"volatility_{p}d"] = df["return_1d"].rolling(p).std()

        ma_periods = sorted(set(FeatureEngineer.TECHNICAL_WINDOWS + [50, 100, 200]))
        for p in ma_periods:
            df[f"ma_{p}"] = df["Close"].rolling(p).mean()
            df[f"close_vs_ma_{p}"] = df["Close"] / df[f"ma_{p}"] - 1

        df["ma_20_vs_50"] = df["ma_20"] / df["ma_50"] - 1
        df["ma_50_vs_200"] = df["ma_50"] / df["ma_200"] - 1

        df["rsi_14"] = FeatureEngineer._rsi(df["Close"], 14)
        df["rsi_28"] = FeatureEngineer._rsi(df["Close"], 28)
        macd, signal = FeatureEngineer._macd(df["Close"])
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_diff"] = macd - signal

        ma20 = df["Close"].rolling(20).mean()
        sd20 = df["Close"].rolling(20).std()
        upper = ma20 + 2 * sd20
        lower = ma20 - 2 * sd20
        df["bollinger_position"] = (df["Close"] - lower) / (upper - lower)

        # Volumen robusto: si no hay volumen, las features quedan a cero.
        if df["Volume"].fillna(0).abs().sum() == 0:
            df["volume_change_1d"] = 0.0
            df["volume_vs_ma20"] = 0.0
        else:
            previous_volume = df["Volume"].shift(1)
            df["volume_change_1d"] = np.where(
                previous_volume > 0,
                df["Volume"] / previous_volume - 1,
                0.0,
            )
            volume_ma20 = df["Volume"].rolling(20).mean()
            df["volume_vs_ma20"] = np.where(
                volume_ma20 > 0,
                df["Volume"] / volume_ma20 - 1,
                0.0,
            )

        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month

        feature_cols = list(FeatureEngineer.TECHNICAL_FEATURES)

        # Variables externas de mercado/macro.
        if external_features is not None and not external_features.empty:
            external = external_features.copy()
            external["date"] = pd.to_datetime(
                external["date"], errors="coerce"
            ).dt.normalize()

            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.merge(external, on="date", how="left")

            for col in FeatureEngineer.EXTERNAL_FEATURES:
                if col not in df.columns:
                    df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors="coerce")

            feature_cols += FeatureEngineer.EXTERNAL_FEATURES
            print(
                f"Variables externas: ACTIVADAS "
                f"({len(FeatureEngineer.EXTERNAL_FEATURES)} features)."
            )
        else:
            print("Variables externas: DESACTIVADAS.")

        # El sentimiento solo entra si existe histórico real.
        if historical_sentiment is not None and not historical_sentiment.empty:
            df = df.merge(historical_sentiment, on="date", how="left")
            for col in FeatureEngineer.SENTIMENT_FEATURES:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            feature_cols += FeatureEngineer.SENTIMENT_FEATURES
            print("NLP histórico: ACTIVADO dentro del modelo.")
        else:
            print("NLP histórico: DESACTIVADO dentro del modelo (evita covariate shift/leakage).")

        df = df.replace([np.inf, -np.inf], np.nan)
        return df, feature_cols

    @staticmethod
    def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        out = df.copy()
        out[f"future_close_{horizon}d"] = out["Close"].shift(-horizon)
        out[f"future_return_{horizon}d"] = out[f"future_close_{horizon}d"] / out["Close"] - 1
        out[f"target_{horizon}d"] = (out[f"future_return_{horizon}d"] > 0).astype(float)
        out.loc[out[f"future_close_{horizon}d"].isna(), f"target_{horizon}d"] = np.nan
        return out

    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal


# =============================================================================
# 8. MÉTRICAS Y MODELOS CLÁSICOS
# =============================================================================

class Metrics:
    @staticmethod
    def classification(y_true, y_pred, y_prob) -> Dict[str, float]:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.5
        try:
            ll = log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))
        except Exception:
            ll = np.nan
        try:
            brier = brier_score_loss(y_true, y_prob)
        except Exception:
            brier = np.nan

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": auc,
            "log_loss": ll,
            "brier": brier,
            "positive_rate": float(np.mean(y_true)),
        }

    @staticmethod
    def model_weight_score(metrics: Dict[str, float]) -> float:
        # AUC por debajo de 0.5 no aporta score positivo; F1 pesa menos.
        auc_component = max(float(metrics.get("roc_auc", 0.5)) - 0.5, 0.0)
        bal_component = max(float(metrics.get("balanced_accuracy", 0.5)) - 0.5, 0.0)
        f1_component = max(float(metrics.get("f1", 0.0)), 0.0)
        return 0.50 * auc_component + 0.25 * bal_component + 0.25 * f1_component


class ClassicalModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _prepare_for_fit(name: str, model, y_train: pd.Series):
        """Ajusta parámetros dependientes del fold sin mirar validación/test."""
        if name == "XGBoost":
            positives = int((y_train == 1).sum())
            negatives = int((y_train == 0).sum())
            scale = negatives / positives if positives > 0 else 1.0
            model.set_params(scale_pos_weight=float(scale))
        return model

    def build_models(self) -> Dict[str, object]:
        models: Dict[str, object] = {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=self.config.random_seed,
                )),
            ]),
            "RandomForest": RandomForestClassifier(
                n_estimators=500,
                max_depth=7,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=self.config.random_seed,
                n_jobs=1 if self.config.single_thread_models else -1,
            ),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=600,
                max_depth=3,
                learning_rate=0.02,
                min_child_weight=4,
                gamma=0.05,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=self.config.random_seed,
                n_jobs=1 if self.config.single_thread_models else -1,
            )
        return models

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        models = self.build_models()
        n_splits = min(self.config.cv_splits, max(2, len(X) // 150))
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

        rows = []
        raw_weights = {}

        for name, model in models.items():
            print(f"\nValidación cruzada temporal: {name}")
            fold_rows = []

            for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_valid = X.iloc[valid_idx]
                y_valid = y.iloc[valid_idx]

                if y_train.nunique() < 2 or y_valid.nunique() < 2:
                    print(f"  Fold {fold}: omitido por clase única.")
                    continue

                fold_seed = Reproducibility.stable_seed(
                    self.config.random_seed, "cv", name, horizon, fold
                )
                Reproducibility.apply(
                    fold_seed, self.config.deterministic_tensorflow
                )
                fitted = clone(model)
                fitted = self._prepare_for_fit(name, fitted, y_train)
                fitted.fit(X_train, y_train)
                prob = fitted.predict_proba(X_valid)[:, 1]
                pred = (prob >= 0.5).astype(int)
                m = Metrics.classification(y_valid, pred, prob)
                m["fold"] = fold
                fold_rows.append(m)

                print(
                    f"  Fold {fold}: Acc={m['accuracy']:.3f} "
                    f"BalAcc={m['balanced_accuracy']:.3f} F1={m['f1']:.3f} "
                    f"AUC={m['roc_auc']:.3f} Brier={m['brier']:.3f}"
                )

            if not fold_rows:
                continue

            avg = pd.DataFrame(fold_rows).mean(numeric_only=True).to_dict()
            avg["model"] = name
            rows.append(avg)
            raw_weights[name] = Metrics.model_weight_score(avg)

        if not rows:
            raise RuntimeError("No se ha podido completar ningún fold de validación temporal.")

        results = pd.DataFrame(rows)
        order = [
            "model", "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "roc_auc", "log_loss", "brier", "positive_rate",
        ]
        results = results[order].sort_values(["roc_auc", "balanced_accuracy"], ascending=False)
        return results, self.normalize_weights(raw_weights)

    @staticmethod
    def normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
        if not raw:
            return {}
        total = sum(max(v, 0.0) for v in raw.values())
        if total <= 0:
            return {k: 1.0 / len(raw) for k in raw}
        return {k: max(v, 0.0) / total for k, v in raw.items()}

    def fit_final(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
        fitted = {}
        for name, model in self.build_models().items():
            print(f"Entrenando modelo final: {name}")
            final_seed = Reproducibility.stable_seed(
                self.config.random_seed, "final", name, len(X_train), X_train.shape[1]
            )
            Reproducibility.apply(
                final_seed, self.config.deterministic_tensorflow
            )
            m = clone(model)
            m = self._prepare_for_fit(name, m, y_train)
            m.fit(X_train, y_train)
            fitted[name] = m
        return fitted



# =============================================================================
# 9. REGRESIÓN DE PRECIO FUTURO / RMSE
# =============================================================================

class RegressionMetrics:
    """Métricas de regresión expresadas finalmente en unidades de precio."""

    @staticmethod
    def evaluate(
        actual_price: np.ndarray,
        predicted_price: np.ndarray,
        current_price: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        actual = np.asarray(actual_price, dtype=float)
        predicted = np.asarray(predicted_price, dtype=float)

        mask = np.isfinite(actual) & np.isfinite(predicted)
        actual = actual[mask]
        predicted = predicted[mask]

        if current_price is not None:
            current = np.asarray(current_price, dtype=float)[mask]
        else:
            current = None

        if len(actual) == 0:
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "mape_pct": np.nan,
                "r2": np.nan,
                "direction_accuracy": np.nan,
            }

        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        safe_actual = np.where(np.abs(actual) > 1e-12, np.abs(actual), np.nan)
        mape = float(np.nanmean(np.abs(actual - predicted) / safe_actual) * 100.0)

        try:
            r2 = float(r2_score(actual, predicted))
        except Exception:
            r2 = np.nan

        if current is not None:
            actual_dir = actual > current
            pred_dir = predicted > current
            direction_accuracy = float(np.mean(actual_dir == pred_dir))
        else:
            direction_accuracy = np.nan

        return {
            "rmse": rmse,
            "mae": mae,
            "mape_pct": mape,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
        }


class PriceRegressionTrainer:
    """
    Segundo problema supervisado del TFM.

    En vez de aprender directamente el nivel de precio (no estacionario), los
    modelos aprenden el log-retorno futuro:

        y_t = log(Close[t+h] / Close[t])

    y posteriormente se reconstruye el precio:

        Close_pred[t+h] = Close[t] * exp(y_pred)

    De esta forma el RMSE se calcula en unidades monetarias pero el objetivo de
    entrenamiento es más estable que el precio bruto.
    """

    def __init__(self, config: Config):
        self.config = config

    def build_models(self) -> Dict[str, object]:
        models: Dict[str, object] = {}

        if "Ridge" in self.config.regression_models:
            models["Ridge"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=2.0)),
            ])

        if "RandomForestRegressor" in self.config.regression_models:
            models["RandomForestRegressor"] = RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=8,
                max_features="sqrt",
                random_state=self.config.random_seed,
                n_jobs=1 if self.config.single_thread_models else -1,
            )

        if "XGBRegressor" in self.config.regression_models and XGBOOST_AVAILABLE:
            models["XGBRegressor"] = XGBRegressor(
                n_estimators=650,
                max_depth=3,
                learning_rate=0.02,
                min_child_weight=4,
                gamma=0.02,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                random_state=self.config.random_seed,
                n_jobs=1 if self.config.single_thread_models else -1,
            )

        if not models:
            raise RuntimeError("No hay modelos de regresión disponibles.")
        return models

    @staticmethod
    def _log_return_target(df: pd.DataFrame, horizon: int) -> pd.Series:
        future = pd.to_numeric(df[f"future_close_{horizon}d"], errors="coerce")
        current = pd.to_numeric(df["Close"], errors="coerce")
        return np.log(future / current)

    @staticmethod
    def _prices_from_log_return(current_price, predicted_log_return):
        current = np.asarray(current_price, dtype=float)
        pred_ret = np.asarray(predicted_log_return, dtype=float)
        # Protección frente a extrapolaciones patológicas.
        pred_ret = np.clip(pred_ret, -1.5, 1.5)
        return current * np.exp(pred_ret)

    @staticmethod
    def normalize_inverse_rmse(rmses: Dict[str, float]) -> Dict[str, float]:
        raw = {}
        for name, rmse in rmses.items():
            if rmse is not None and np.isfinite(rmse) and rmse > 0:
                raw[name] = 1.0 / rmse
        if not raw:
            n = len(rmses)
            return {k: 1.0 / n for k in rmses} if n else {}
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def cross_validate(
        self,
        X: pd.DataFrame,
        y_log_return: pd.Series,
        current_price: pd.Series,
        future_price: pd.Series,
        horizon: int,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Validación temporal de los regresores y del ensemble.

        El ensemble se evalúa de forma OOF por fold: para evaluar un fold,
        sus pesos se calculan usando exclusivamente los RMSE de los OTROS
        folds. Así el RMSE CV del ensemble no utiliza el resultado del propio
        fold para decidir sus pesos.

        Los pesos finales que después se aplican al holdout sí se estiman con
        todos los folds de train, pero nunca con el holdout.
        """
        models = self.build_models()
        n_splits = min(self.config.cv_splits, max(2, len(X) // 150))
        splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
        folds = list(splitter.split(X))

        rows = []
        fold_metrics_by_model: Dict[str, Dict[int, Dict[str, float]]] = {
            name: {} for name in models
        }
        fold_predictions: Dict[int, Dict[str, np.ndarray]] = {
            fold: {} for fold in range(1, len(folds) + 1)
        }
        fold_context: Dict[int, Dict[str, np.ndarray]] = {}
        mean_rmses: Dict[str, float] = {}

        for name, model in models.items():
            fold_rows = []
            print(f"\nCV REGRESIÓN TEMPORAL: {name}")

            for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
                seed = Reproducibility.stable_seed(
                    self.config.random_seed, "reg_cv", name, horizon, fold
                )
                Reproducibility.apply(seed, self.config.deterministic_tensorflow)

                fitted = clone(model)
                fitted.fit(X.iloc[train_idx], y_log_return.iloc[train_idx])
                pred_lr = fitted.predict(X.iloc[valid_idx])
                pred_price = self._prices_from_log_return(
                    current_price.iloc[valid_idx].values, pred_lr
                )

                m = RegressionMetrics.evaluate(
                    actual_price=future_price.iloc[valid_idx].values,
                    predicted_price=pred_price,
                    current_price=current_price.iloc[valid_idx].values,
                )
                m["fold"] = fold
                fold_rows.append(m)
                fold_metrics_by_model[name][fold] = dict(m)
                fold_predictions[fold][name] = np.asarray(pred_price, dtype=float)
                fold_context[fold] = {
                    "actual": future_price.iloc[valid_idx].values.astype(float),
                    "current": current_price.iloc[valid_idx].values.astype(float),
                }

                print(
                    f"  Fold {fold}: RMSE={m['rmse']:.4f} "
                    f"MAE={m['mae']:.4f} MAPE={m['mape_pct']:.2f}% "
                    f"DirAcc={m['direction_accuracy']:.3f}"
                )

            avg = pd.DataFrame(fold_rows).mean(numeric_only=True).to_dict()
            avg["model"] = name
            rows.append(avg)
            mean_rmses[name] = float(avg["rmse"])

        # Pesos finales del ensemble: solo CV de train.
        final_weights = self.normalize_inverse_rmse(mean_rmses)

        # Evaluación CV OOF del ensemble. Para cada fold, calculamos sus pesos
        # con los RMSE observados en los restantes folds.
        ensemble_fold_rows = []
        for fold in range(1, len(folds) + 1):
            rmses_without_fold = {}
            for name in models:
                other_rmses = [
                    metrics["rmse"]
                    for f, metrics in fold_metrics_by_model[name].items()
                    if f != fold and np.isfinite(metrics.get("rmse", np.nan))
                ]
                if other_rmses:
                    rmses_without_fold[name] = float(np.mean(other_rmses))

            fold_weights = self.normalize_inverse_rmse(rmses_without_fold)
            if not fold_weights:
                fold_weights = {name: 1.0 / len(models) for name in models}

            ensemble_pred = np.zeros_like(
                fold_context[fold]["actual"], dtype=float
            )
            for name in models:
                ensemble_pred += (
                    float(fold_weights.get(name, 0.0))
                    * fold_predictions[fold][name]
                )

            m = RegressionMetrics.evaluate(
                actual_price=fold_context[fold]["actual"],
                predicted_price=ensemble_pred,
                current_price=fold_context[fold]["current"],
            )
            m["fold"] = fold
            ensemble_fold_rows.append(m)

        if ensemble_fold_rows:
            ensemble_avg = (
                pd.DataFrame(ensemble_fold_rows)
                .mean(numeric_only=True)
                .to_dict()
            )
            ensemble_avg["model"] = "ENSEMBLE_REGRESSION"
            rows.append(ensemble_avg)

        result = pd.DataFrame(rows)
        order = [
            "model", "rmse", "mae", "mape_pct", "r2",
            "direction_accuracy",
        ]
        result = result[
            [c for c in order if c in result.columns]
        ].sort_values("rmse").reset_index(drop=True)

        return result, final_weights

    def run(
        self,
        ticker: str,
        horizon: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        latest_df: pd.DataFrame,
        horizon_dir: str,
    ) -> Dict[str, Any]:
        if not self.config.use_price_regression:
            return {}

        future_col = f"future_close_{horizon}d"
        required = feature_cols + ["Close", future_col]
        train = train_df.dropna(subset=required).copy()
        test = test_df.dropna(subset=required).copy()

        if len(train) < self.config.regression_min_train_rows or len(test) < 20:
            print("REGRESIÓN: dataset insuficiente; se omite.")
            return {}

        X_train = train[feature_cols]
        X_test = test[feature_cols]
        y_train_lr = self._log_return_target(train, horizon)
        y_test_lr = self._log_return_target(test, horizon)
        current_train = train["Close"].astype(float)
        current_test = test["Close"].astype(float)
        future_train = train[future_col].astype(float)
        future_test = test[future_col].astype(float)

        cv_results, weights = self.cross_validate(
            X_train,
            y_train_lr,
            current_train,
            future_train,
            horizon,
        )

        models = self.build_models()
        holdout_rows = []
        prediction_table = test[["date", "Close", future_col]].copy().reset_index(drop=True)
        prediction_table = prediction_table.rename(columns={future_col: "actual_future_price"})
        predicted_prices = {}
        fitted_models = {}

        for name, model in models.items():
            seed = Reproducibility.stable_seed(
                self.config.random_seed, "reg_final", ticker, horizon, name
            )
            Reproducibility.apply(seed, self.config.deterministic_tensorflow)
            fitted = clone(model)
            fitted.fit(X_train, y_train_lr)
            fitted_models[name] = fitted

            pred_lr = fitted.predict(X_test)
            pred_price = self._prices_from_log_return(current_test.values, pred_lr)
            predicted_prices[name] = pred_price
            prediction_table[f"pred_price_{name}"] = pred_price

            m = RegressionMetrics.evaluate(
                future_test.values,
                pred_price,
                current_test.values,
            )
            m["model"] = name
            holdout_rows.append(m)

        # Ensemble ponderado por 1/RMSE de CV, nunca por holdout.
        valid_weights = {k: weights.get(k, 0.0) for k in predicted_prices}
        total = sum(valid_weights.values())
        if total <= 0:
            valid_weights = {k: 1 / len(predicted_prices) for k in predicted_prices}
        else:
            valid_weights = {k: v / total for k, v in valid_weights.items()}

        ensemble_price = np.zeros(len(test), dtype=float)
        for name, arr in predicted_prices.items():
            ensemble_price += valid_weights[name] * arr
        prediction_table["pred_price_ENSEMBLE"] = ensemble_price

        ensemble_metrics = RegressionMetrics.evaluate(
            future_test.values,
            ensemble_price,
            current_test.values,
        )
        ensemble_metrics["model"] = "ENSEMBLE_REGRESSION"
        holdout_rows.append(dict(ensemble_metrics))

        # Baseline de persistencia: el precio futuro será igual al precio actual.
        persistence_price = current_test.values
        persistence_metrics = RegressionMetrics.evaluate(
            future_test.values,
            persistence_price,
            current_test.values,
        )
        persistence_metrics["model"] = "PERSISTENCE_BASELINE"
        holdout_rows.append(dict(persistence_metrics))

        holdout_metrics = pd.DataFrame(holdout_rows).sort_values("rmse").reset_index(drop=True)

        # ------------------------------------------------------------------
        # SELECCIÓN FINAL DEL REGRESOR: SOLO MEDIANTE RMSE DE CV TEMPORAL
        # ------------------------------------------------------------------
        # No usamos el holdout para elegir el predictor final. Los candidatos
        # prioritarios son exactamente los acordados: RF, XGBoost y ensemble.
        preferred_candidates = [
            "RandomForestRegressor",
            "XGBRegressor",
            "ENSEMBLE_REGRESSION",
        ]
        candidate_cv = cv_results[
            cv_results["model"].isin(preferred_candidates)
        ].copy()

        # Fallback defensivo si algún modelo no está disponible.
        if candidate_cv.empty:
            candidate_cv = cv_results[
                cv_results["model"] != "PERSISTENCE_BASELINE"
            ].copy()

        candidate_cv["rmse"] = pd.to_numeric(
            candidate_cv["rmse"], errors="coerce"
        )
        candidate_cv = candidate_cv.dropna(subset=["rmse"])
        if candidate_cv.empty:
            raise RuntimeError(
                "No hay RMSE CV válido para seleccionar el regresor final."
            )

        selected_model_name = str(
            candidate_cv.sort_values("rmse").iloc[0]["model"]
        )
        selected_cv_rmse = float(
            candidate_cv.sort_values("rmse").iloc[0]["rmse"]
        )

        if selected_model_name == "ENSEMBLE_REGRESSION":
            selected_holdout_price = np.asarray(ensemble_price, dtype=float)
            selected_holdout_metrics = dict(ensemble_metrics)
        else:
            selected_holdout_price = np.asarray(
                predicted_prices[selected_model_name], dtype=float
            )
            selected_row = holdout_metrics[
                holdout_metrics["model"] == selected_model_name
            ]
            if selected_row.empty:
                raise RuntimeError(
                    f"No se encuentran métricas holdout para {selected_model_name}."
                )
            selected_holdout_metrics = selected_row.iloc[0].to_dict()

        selected_holdout_rmse = float(selected_holdout_metrics["rmse"])
        persistence_rmse = float(persistence_metrics["rmse"])
        regression_rmse_skill = (
            1.0 - selected_holdout_rmse / persistence_rmse
            if np.isfinite(persistence_rmse) and persistence_rmse > 0
            else np.nan
        )
        ensemble_rmse_skill = (
            1.0 - float(ensemble_metrics["rmse"]) / persistence_rmse
            if np.isfinite(persistence_rmse) and persistence_rmse > 0
            else np.nan
        )

        # Tabla explícita de selección para auditoría/reproducibilidad.
        selection_table = cv_results.copy()
        selection_table = selection_table.rename(columns={
            "rmse": "cv_rmse",
            "mae": "cv_mae",
            "mape_pct": "cv_mape_pct",
            "r2": "cv_r2",
            "direction_accuracy": "cv_direction_accuracy",
        })
        holdout_for_merge = holdout_metrics.rename(columns={
            "rmse": "holdout_rmse",
            "mae": "holdout_mae",
            "mape_pct": "holdout_mape_pct",
            "r2": "holdout_r2",
            "direction_accuracy": "holdout_direction_accuracy",
        })
        selection_table = selection_table.merge(
            holdout_for_merge, on="model", how="left"
        )
        selection_table["eligible_for_final_selection"] = (
            selection_table["model"].isin(preferred_candidates)
        )
        selection_table["selected_by_cv"] = (
            selection_table["model"] == selected_model_name
        )

        # Residuos OOS DEL MODELO SELECCIONADO para intervalos empíricos.
        residual = future_test.values - selected_holdout_price
        q_low = float(np.nanquantile(residual, self.config.regression_interval_low))
        q_high = float(np.nanquantile(residual, self.config.regression_interval_high))
        q025 = float(np.nanquantile(residual, 0.025))
        q975 = float(np.nanquantile(residual, 0.975))

        X_latest = latest_df[feature_cols]
        current_latest = float(latest_df["Close"].iloc[0])
        latest_prices = {}
        latest_log_returns = {}

        for name, model in fitted_models.items():
            pred_lr = float(model.predict(X_latest)[0])
            latest_log_returns[name] = pred_lr
            latest_prices[name] = float(
                self._prices_from_log_return([current_latest], [pred_lr])[0]
            )

        ensemble_latest_price = float(sum(
            latest_prices[name] * valid_weights[name] for name in latest_prices
        ))

        if selected_model_name == "ENSEMBLE_REGRESSION":
            predicted_future_price = ensemble_latest_price
        else:
            predicted_future_price = float(latest_prices[selected_model_name])

        expected_return = predicted_future_price / current_latest - 1.0

        interval80_low = max(0.0, predicted_future_price + q_low)
        interval80_high = max(0.0, predicted_future_price + q_high)
        interval95_low = max(0.0, predicted_future_price + q025)
        interval95_high = max(0.0, predicted_future_price + q975)

        # Soporte experimental para decisión: no se presenta como precio garantizado.
        edge_required = max(
            float(self.config.regression_min_expected_edge),
            float(self.config.transaction_cost) * 2.0,
        )
        if expected_return > edge_required:
            price_bias = "ALCISTA"
        elif expected_return < -edge_required:
            price_bias = "BAJISTA"
        else:
            price_bias = "NEUTRAL"

        # Precio máximo de entrada que conservaría el margen mínimo exigido
        # si el objetivo estimado llegara a materializarse. Es una referencia
        # experimental, no una orden de compra.
        max_entry_price_for_edge = (
            predicted_future_price / (1.0 + edge_required)
            if predicted_future_price > 0 else np.nan
        )

        latest = {
            "ticker": ticker,
            "horizon": horizon,
            "current_close": current_latest,
            "predicted_future_price": predicted_future_price,
            "expected_return": expected_return,
            "price_bias": price_bias,
            "entry_reference": current_latest,
            "max_entry_price_for_edge": max_entry_price_for_edge,
            "target_price_reference": predicted_future_price,
            "risk_reference_80_low": interval80_low,
            "interval80_low": interval80_low,
            "interval80_high": interval80_high,
            "interval95_low": interval95_low,
            "interval95_high": interval95_high,
            "selected_regression_model": selected_model_name,
            "selected_cv_rmse": selected_cv_rmse,
            "holdout_rmse": selected_holdout_rmse,
            "holdout_mae": float(selected_holdout_metrics["mae"]),
            "holdout_mape_pct": float(selected_holdout_metrics["mape_pct"]),
            "holdout_r2": float(selected_holdout_metrics["r2"]),
            "direction_accuracy_from_regression": float(
                selected_holdout_metrics["direction_accuracy"]
            ),
            "persistence_holdout_rmse": persistence_rmse,
            "regression_rmse_skill_vs_persistence": regression_rmse_skill,
            "ensemble_holdout_rmse": float(ensemble_metrics["rmse"]),
            "ensemble_rmse_skill_vs_persistence": ensemble_rmse_skill,
            "predicted_price_ENSEMBLE": ensemble_latest_price,
            **{f"weight_{k}": v for k, v in valid_weights.items()},
            **{f"predicted_price_{k}": v for k, v in latest_prices.items()},
        }

        print("\n" + "=" * 118)
        print(f"REGRESIÓN DE PRECIO {ticker} | horizonte {horizon} sesiones")
        print("=" * 118)
        print(holdout_metrics.to_string(index=False))
        print(f"Modelo regresión seleccionado:{selected_model_name}")
        print(f"Criterio selección:          menor RMSE de CV temporal")
        print(f"RMSE CV seleccionado:        {selected_cv_rmse:.4f}")
        print(f"Precio actual:               {current_latest:.4f}")
        print(f"Precio futuro estimado:      {predicted_future_price:.4f}")
        print(f"Retorno esperado estimado:   {expected_return * 100:+.2f}%")
        print(f"Entrada máxima ref. edge:    {max_entry_price_for_edge:.4f}")
        print(f"RMSE holdout seleccionado:   {selected_holdout_rmse:.4f}")
        print(f"RMSE persistencia:           {persistence_rmse:.4f}")
        print(f"Skill RMSE vs persistencia:  {regression_rmse_skill:+.4f}")
        print(f"MAE holdout:                 {float(selected_holdout_metrics['mae']):.4f}")
        print(f"MAPE holdout:                {float(selected_holdout_metrics['mape_pct']):.2f}%")
        print(f"R² holdout:                  {float(selected_holdout_metrics['r2']):.4f}")
        print(f"Intervalo empírico 80%:      [{interval80_low:.4f}, {interval80_high:.4f}]")
        print(f"Intervalo empírico 95%:      [{interval95_low:.4f}, {interval95_high:.4f}]")
        print(f"Sesgo de precio experimental:{price_bias}")
        print(
            "NOTA: target_price_reference es una estimación estadística, "
            "no un precio garantizado ni una recomendación de inversión."
        )

        regression_dir = os.path.join(horizon_dir, "price_regression")
        os.makedirs(regression_dir, exist_ok=True)
        cv_results.to_csv(os.path.join(regression_dir, "regression_cv.csv"), index=False)
        holdout_metrics.to_csv(
            os.path.join(regression_dir, "regression_holdout_metrics.csv"),
            index=False,
        )
        selection_table.to_csv(
            os.path.join(regression_dir, "regression_model_selection.csv"),
            index=False,
        )
        prediction_table["pred_price_SELECTED"] = selected_holdout_price
        prediction_table.to_csv(
            os.path.join(regression_dir, "regression_holdout_predictions.csv"),
            index=False,
        )
        pd.DataFrame([latest]).to_csv(
            os.path.join(regression_dir, "latest_price_prediction.csv"),
            index=False,
        )

        for name, model in fitted_models.items():
            joblib.dump(model, os.path.join(regression_dir, f"model_{name}.joblib"))

        # Gráfico real vs predicho.
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(prediction_table["date"], prediction_table["actual_future_price"], label="Precio futuro real")
            plt.plot(prediction_table["date"], prediction_table["pred_price_SELECTED"], label=f"Precio estimado ({selected_model_name})")
            plt.title(f"{ticker} - Regresión de precio - Horizonte {horizon}d")
            plt.xlabel("Fecha de predicción")
            plt.ylabel("Precio")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(regression_dir, "actual_vs_predicted_price.png"),
                dpi=250,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as exc:
            print(f"No se pudo crear gráfico de regresión: {exc}")

        return {
            "cv_results": cv_results,
            "holdout_metrics": holdout_metrics,
            "ensemble_metrics": ensemble_metrics,
            "selected_model": selected_model_name,
            "selected_metrics": selected_holdout_metrics,
            "selected_cv_rmse": selected_cv_rmse,
            "persistence_metrics": persistence_metrics,
            "regression_rmse_skill_vs_persistence": regression_rmse_skill,
            "latest": latest,
            "weights": valid_weights,
        }


# =============================================================================
# 9. LSTM CON VALIDACIÓN INTERNA (NO USA HOLDOUT PARA SU PESO)
# =============================================================================

class LSTMTrainer:
    """LSTM temporal con validación interna, early stopping y balance de clases."""

    def __init__(self, config: Config):
        self.config = config

    def enabled(self) -> bool:
        return self.config.use_lstm and TENSORFLOW_AVAILABLE

    @staticmethod
    def _make_sequences(X_scaled: np.ndarray, y: np.ndarray, lookback: int):
        xs, ys = [], []
        for i in range(lookback, len(X_scaled)):
            xs.append(X_scaled[i - lookback:i])
            ys.append(y[i])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    @staticmethod
    def _class_weights(y: np.ndarray) -> Dict[int, float]:
        y = np.asarray(y).astype(int)
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        total = max(n0 + n1, 1)
        if n0 == 0 or n1 == 0:
            return {0: 1.0, 1: 1.0}
        return {
            0: total / (2.0 * n0),
            1: total / (2.0 * n1),
        }

    def _build_model(self, input_shape: Tuple[int, int], seed: int):
        # Limpiamos grafo y restauramos la semilla ANTES de crear pesos y Dropout.
        try:
            K.clear_session()
        except Exception:
            pass

        Reproducibility.apply(
            seed, self.config.deterministic_tensorflow
        )

        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.25),
            LSTM(32, return_sequences=False),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dropout(0.10),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train_validate_and_test(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        if not self.enabled():
            return None, None, None, None, None

        lookback = self.config.lstm_lookback
        if len(X_train) < max(350, lookback * 6):
            print("LSTM: datos insuficientes. Se omite.")
            return None, None, None, None, None

        # Split temporal interno: el holdout final sigue completamente intacto.
        inner_split = int(len(X_train) * 0.85)
        inner_train_X = X_train.iloc[:inner_split]
        inner_train_y = y_train.iloc[:inner_split]
        inner_val_X = X_train.iloc[inner_split:]
        inner_val_y = y_train.iloc[inner_split:]

        scaler_inner = StandardScaler()
        inner_train_scaled = scaler_inner.fit_transform(inner_train_X)
        inner_val_scaled = scaler_inner.transform(inner_val_X)

        X_seq_train, y_seq_train = self._make_sequences(
            inner_train_scaled, inner_train_y.values, lookback
        )

        combined_val_X = np.vstack([inner_train_scaled[-lookback:], inner_val_scaled])
        combined_val_y = np.concatenate([
            inner_train_y.values[-lookback:], inner_val_y.values
        ])
        X_seq_val, y_seq_val = self._make_sequences(
            combined_val_X, combined_val_y, lookback
        )

        if len(X_seq_train) < 120 or len(X_seq_val) < 30:
            print("LSTM: secuencias insuficientes. Se omite.")
            return None, None, None, None, None

        validation_seed = Reproducibility.stable_seed(
            self.config.random_seed, "lstm_validation", len(X_train), X_train.shape[1]
        )
        validation_model = self._build_model(
            (lookback, X_train.shape[1]), validation_seed
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=6,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        print("Entrenando LSTM para validación interna...")
        history = validation_model.fit(
            X_seq_train,
            y_seq_train,
            validation_data=(X_seq_val, y_seq_val),
            epochs=self.config.lstm_epochs,
            batch_size=self.config.lstm_batch_size,
            verbose=0,
            callbacks=callbacks,
            class_weight=self._class_weights(y_seq_train),
            shuffle=False,
        )

        val_prob = validation_model.predict(X_seq_val, verbose=0).ravel()
        val_pred = (val_prob >= 0.5).astype(int)
        val_metrics = Metrics.classification(
            y_seq_val.astype(int), val_pred, val_prob
        )

        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        best_epoch = max(best_epoch, 1)

        print(
            f"LSTM validación interna: epochs={best_epoch}, "
            f"Acc={val_metrics['accuracy']:.3f}, "
            f"BalAcc={val_metrics['balanced_accuracy']:.3f}, "
            f"F1={val_metrics['f1']:.3f}, "
            f"AUC={val_metrics['roc_auc']:.3f}"
        )

        # Modelo final entrenado SOLO con train completo.
        scaler_final = StandardScaler()
        train_scaled = scaler_final.fit_transform(X_train)
        test_scaled = scaler_final.transform(X_test)

        X_seq_full, y_seq_full = self._make_sequences(
            train_scaled, y_train.values, lookback
        )

        final_seed = Reproducibility.stable_seed(
            self.config.random_seed, "lstm_final", len(X_train), X_train.shape[1], best_epoch
        )
        final_model = self._build_model(
            (lookback, X_train.shape[1]), final_seed
        )
        print(f"Entrenando LSTM final durante {best_epoch} epochs...")
        final_model.fit(
            X_seq_full,
            y_seq_full,
            epochs=best_epoch,
            batch_size=self.config.lstm_batch_size,
            verbose=0,
            class_weight=self._class_weights(y_seq_full),
            shuffle=False,
        )

        # Las primeras secuencias de test usan solo la cola histórica de train.
        combined_test_X = np.vstack([train_scaled[-lookback:], test_scaled])
        combined_test_y = np.concatenate([
            y_train.values[-lookback:], y_test.values
        ])
        X_seq_test, y_seq_test = self._make_sequences(
            combined_test_X, combined_test_y, lookback
        )

        test_prob = final_model.predict(X_seq_test, verbose=0).ravel()
        test_pred = (test_prob >= 0.5).astype(int)
        test_metrics = Metrics.classification(
            y_seq_test.astype(int), test_pred, test_prob
        )

        print(
            f"LSTM holdout: Acc={test_metrics['accuracy']:.3f}, "
            f"BalAcc={test_metrics['balanced_accuracy']:.3f}, "
            f"F1={test_metrics['f1']:.3f}, "
            f"AUC={test_metrics['roc_auc']:.3f}"
        )

        return final_model, scaler_final, val_metrics, test_metrics, test_prob

    def predict_latest(self, model, scaler, X_all: pd.DataFrame) -> Optional[float]:
        if model is None or scaler is None or len(X_all) < self.config.lstm_lookback:
            return None
        x = scaler.transform(X_all.iloc[-self.config.lstm_lookback:])
        x = np.expand_dims(x.astype(np.float32), axis=0)
        return float(model.predict(x, verbose=0).ravel()[0])


# =============================================================================
# 10. ENSEMBLE
# =============================================================================

class EnsemblePredictor:
    @staticmethod
    def combine_scalar(probabilities: Dict[str, float], weights: Dict[str, float]):
        valid = {k: v for k, v in probabilities.items() if v is not None and np.isfinite(v)}
        if not valid:
            return 0.5, {}

        local = {k: max(weights.get(k, 0.0), 0.0) for k in valid}
        total = sum(local.values())
        if total <= 0:
            local = {k: 1.0 / len(valid) for k in valid}
        else:
            local = {k: v / total for k, v in local.items()}

        prob = sum(valid[k] * local[k] for k in valid)
        return float(prob), local

    @staticmethod
    def combine_arrays(probabilities: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        valid = {k: v for k, v in probabilities.items() if v is not None}
        if not valid:
            raise ValueError("No hay probabilidades para construir el ensemble.")

        length = len(next(iter(valid.values())))
        valid = {k: v for k, v in valid.items() if len(v) == length}
        local = {k: max(weights.get(k, 0.0), 0.0) for k in valid}
        total = sum(local.values())
        if total <= 0:
            local = {k: 1.0 / len(valid) for k in valid}
        else:
            local = {k: v / total for k, v in local.items()}

        out = np.zeros(length, dtype=float)
        for k, arr in valid.items():
            out += local[k] * arr
        return out


# =============================================================================
# 11. BACKTESTING
# =============================================================================

class OverlappingTargetValidator:
    """
    Auditoría específica para horizontes con etiquetas solapadas.

    Para un horizonte H, target_t y target_{t+1} comparten H-1 sesiones
    futuras. Esto puede inflar la sensación de tamaño muestral efectivo.

    Se generan dos diagnósticos:
      1. Phase validation:
         para cada offset 0..H-1 se evalúan observaciones separadas H
         sesiones entre sí. Dentro de cada fase, los targets no se solapan.

      2. Moving-block bootstrap:
         bootstrap por bloques de longitud H sobre el holdout para estimar
         un intervalo de confianza del ROC-AUC compatible con dependencia
         temporal local.

    Estos diagnósticos NO sustituyen el holdout original; lo complementan.
    """

    def __init__(self, config: Config):
        self.config = config

    def evaluate(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        horizon: int,
        output_dir: str,
        label: str = "ENSEMBLE",
    ) -> Dict[str, Any]:

        if not self.config.validate_overlapping_targets:
            return {}

        y = np.asarray(y_true).astype(int)
        p = np.asarray(y_prob, dtype=float)

        phase_rows = []

        for offset in range(max(int(horizon), 1)):
            idx = np.arange(offset, len(y), max(int(horizon), 1))

            if len(idx) < 8:
                continue

            yy = y[idx]
            pp = p[idx]
            pred = (pp >= 0.5).astype(int)

            metrics = Metrics.classification(
                yy,
                pred,
                pp,
            )

            metrics.update({
                "phase_offset": offset,
                "n_observations": int(len(idx)),
                "n_positive": int(yy.sum()),
                "n_negative": int((yy == 0).sum()),
            })
            phase_rows.append(metrics)

        phase_df = pd.DataFrame(phase_rows)

        if not phase_df.empty:
            phase_df.to_csv(
                os.path.join(
                    output_dir,
                    f"{label.lower()}_nonoverlap_phases.csv",
                ),
                index=False,
            )

        summary = {
            "horizon": int(horizon),
            "holdout_n": int(len(y)),
            "approx_effective_n": int(
                max(1, np.ceil(len(y) / max(int(horizon), 1)))
            ),
            "n_valid_phases": int(len(phase_df)),
        }

        for metric in [
            "roc_auc",
            "balanced_accuracy",
            "accuracy",
            "f1",
            "brier",
        ]:
            if (
                not phase_df.empty
                and metric in phase_df.columns
            ):
                values = pd.to_numeric(
                    phase_df[metric],
                    errors="coerce",
                ).dropna()

                if not values.empty:
                    summary[
                        f"nonoverlap_{metric}_mean"
                    ] = float(values.mean())
                    summary[
                        f"nonoverlap_{metric}_std"
                    ] = float(values.std(ddof=0))

        bootstrap = self._block_bootstrap_auc(
            y=y,
            p=p,
            horizon=max(int(horizon), 1),
        )

        summary.update(bootstrap)

        pd.DataFrame([summary]).to_csv(
            os.path.join(
                output_dir,
                f"{label.lower()}_overlap_validation_summary.csv",
            ),
            index=False,
        )

        print("\nVALIDACIÓN DE TARGETS SOLAPADOS")
        print(
            f"Horizonte: {horizon} | Holdout n={len(y)} | "
            f"n efectivo aprox.={summary['approx_effective_n']}"
        )

        if "nonoverlap_roc_auc_mean" in summary:
            print(
                "AUC fases no solapadas: "
                f"{summary['nonoverlap_roc_auc_mean']:.4f} "
                f"± {summary['nonoverlap_roc_auc_std']:.4f}"
            )

        if "bootstrap_auc_low" in summary:
            print(
                "AUC bootstrap por bloques 95% CI: "
                f"[{summary['bootstrap_auc_low']:.4f}, "
                f"{summary['bootstrap_auc_high']:.4f}]"
            )

        return summary

    def _block_bootstrap_auc(
        self,
        y: np.ndarray,
        p: np.ndarray,
        horizon: int,
    ) -> Dict[str, float]:

        if len(y) < 20:
            return {}

        rng = np.random.default_rng(
            int(self.config.overlap_random_seed)
        )

        block_length = max(int(horizon), 1)
        n = len(y)
        values = []

        iterations = int(
            self.config.overlap_bootstrap_iterations
        )

        max_start = max(n - block_length, 0)

        for _ in range(iterations):
            sampled_idx = []

            while len(sampled_idx) < n:
                start = int(
                    rng.integers(0, max_start + 1)
                )
                block = list(
                    range(
                        start,
                        min(start + block_length, n),
                    )
                )
                sampled_idx.extend(block)

            sampled_idx = np.asarray(
                sampled_idx[:n],
                dtype=int,
            )

            yy = y[sampled_idx]
            pp = p[sampled_idx]

            if len(np.unique(yy)) < 2:
                continue

            try:
                values.append(
                    roc_auc_score(yy, pp)
                )
            except Exception:
                pass

        if not values:
            return {}

        values = np.asarray(values, dtype=float)

        return {
            "bootstrap_auc_mean": float(
                np.mean(values)
            ),
            "bootstrap_auc_low": float(
                np.quantile(values, 0.025)
            ),
            "bootstrap_auc_high": float(
                np.quantile(values, 0.975)
            ),
            "bootstrap_iterations_valid": int(
                len(values)
            ),
            "bootstrap_block_length": int(
                block_length
            ),
        }


class Backtester:
    def __init__(self, config: Config):
        self.config = config

    def run(self, test_df: pd.DataFrame, ensemble_prob: np.ndarray) -> pd.DataFrame:
        df = test_df.copy().reset_index(drop=True)
        if len(df) != len(ensemble_prob):
            raise ValueError("Longitud de probabilidades incompatible con el holdout.")

        df["ensemble_probability"] = ensemble_prob

        if self.config.allow_short:
            raw_position = np.where(
                df["ensemble_probability"] >= self.config.buy_threshold,
                1.0,
                np.where(
                    df["ensemble_probability"] <= self.config.sell_threshold,
                    -1.0,
                    0.0,
                ),
            )
        else:
            raw_position = np.where(
                df["ensemble_probability"] >= self.config.buy_threshold,
                1.0,
                0.0,
            )

        # Señal al cierre t; posición efectiva desde t+1.
        df["position"] = pd.Series(raw_position, index=df.index).shift(1).fillna(0.0)
        df["market_return"] = df["Close"].pct_change().fillna(0.0)
        df["position_change"] = df["position"].diff().abs().fillna(0.0)
        df["strategy_return"] = (
            df["position"] * df["market_return"]
            - df["position_change"] * self.config.transaction_cost
        )

        df["strategy_equity"] = self.config.initial_capital * (1 + df["strategy_return"]).cumprod()
        df["buy_hold_equity"] = self.config.initial_capital * (1 + df["market_return"]).cumprod()
        return df

    @staticmethod
    def metrics(df: pd.DataFrame) -> Dict[str, float]:
        strategy = df["strategy_return"].dropna()
        market = df["market_return"].dropna()

        def ann_return(r):
            if len(r) == 0:
                return 0.0
            total = float((1 + r).prod())
            years = len(r) / 252.0
            if years <= 0 or total <= 0:
                return -1.0
            return total ** (1 / years) - 1

        def ann_vol(r):
            return float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.0

        def sharpe(r):
            sd = r.std()
            if len(r) < 2 or sd == 0 or pd.isna(sd):
                return 0.0
            return float(np.sqrt(252) * r.mean() / sd)

        def sortino(r):
            downside = r[r < 0]
            sd = downside.std()
            if len(downside) < 2 or sd == 0 or pd.isna(sd):
                return 0.0
            return float(np.sqrt(252) * r.mean() / sd)

        def max_drawdown(equity):
            peak = equity.cummax()
            return float((equity / peak - 1).min())

        active_returns = df.loc[df["position"] != 0, "strategy_return"]
        gains = strategy[strategy > 0].sum()
        losses = abs(strategy[strategy < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else np.nan

        return {
            "strategy_total_return": float(df["strategy_equity"].iloc[-1] / df["strategy_equity"].iloc[0] - 1),
            "buy_hold_total_return": float(df["buy_hold_equity"].iloc[-1] / df["buy_hold_equity"].iloc[0] - 1),
            "strategy_annual_return": ann_return(strategy),
            "buy_hold_annual_return": ann_return(market),
            "strategy_volatility": ann_vol(strategy),
            "buy_hold_volatility": ann_vol(market),
            "strategy_sharpe": sharpe(strategy),
            "buy_hold_sharpe": sharpe(market),
            "strategy_sortino": sortino(strategy),
            "strategy_max_drawdown": max_drawdown(df["strategy_equity"]),
            "buy_hold_max_drawdown": max_drawdown(df["buy_hold_equity"]),
            "trades": int((df["position_change"] > 0).sum()),
            "market_exposure": float((df["position"] != 0).mean()),
            "win_rate_active_days": float((active_returns > 0).mean()) if len(active_returns) else 0.0,
            "profit_factor": profit_factor,
        }


# =============================================================================
# 12. REGISTRO DE PREDICCIONES
# =============================================================================

class PredictionJournal:
    def __init__(self, config: Config):
        self.config = config
        self.path = config.prediction_log

    def update_realized_results(self, ticker: str, market_df: pd.DataFrame):
        if not os.path.exists(self.path):
            return

        log = pd.read_csv(self.path)
        if log.empty:
            return

        log["prediction_date"] = pd.to_datetime(log["prediction_date"], format="%Y-%m-%d", errors="coerce")
        mask_ticker = log["ticker"].astype(str).str.upper() == ticker.upper()

        for idx in log[mask_ticker].index:
            if pd.notna(log.loc[idx].get("realized_return", np.nan)):
                continue

            pred_date = log.loc[idx, "prediction_date"]
            if pd.isna(pred_date):
                continue
            horizon = int(log.loc[idx, "horizon"])

            current = market_df[market_df["date"] <= pred_date].tail(1)
            future = market_df[market_df["date"] > pred_date].head(horizon)
            if current.empty or len(future) < horizon:
                continue

            realized = float(future["Close"].iloc[-1] / current["Close"].iloc[-1] - 1)
            actual_up = realized > 0
            predicted_up = float(log.loc[idx, "probability_up"]) >= 0.5

            log.loc[idx, "realized_return"] = realized
            log.loc[idx, "actual_direction"] = int(actual_up)
            log.loc[idx, "correct"] = int(actual_up == predicted_up)

        log.to_csv(self.path, index=False)

    def append(
        self,
        ticker: str,
        horizon: int,
        prediction_date: pd.Timestamp,
        probability_up: float,
        signal: str,
        confidence: str,
        sentiment_context: float,
        model_probabilities: Dict[str, float],
    ):
        row = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "ticker": ticker.upper(),
            "prediction_date": pd.to_datetime(prediction_date).strftime("%Y-%m-%d"),
            "horizon": int(horizon),
            "probability_up": float(probability_up),
            "probability_down": float(1 - probability_up),
            "signal": signal,
            "confidence": confidence,
            "sentiment_context": float(sentiment_context),
            "model_probabilities": json.dumps(model_probabilities),
            "realized_return": np.nan,
            "actual_direction": np.nan,
            "correct": np.nan,
        }
        new = pd.DataFrame([row])

        if os.path.exists(self.path):
            old = pd.read_csv(self.path)
            mask = (
                (old["ticker"].astype(str).str.upper() == ticker.upper())
                & (old["prediction_date"].astype(str) == row["prediction_date"])
                & (old["horizon"].astype(int) == int(horizon))
            )
            old = old.loc[~mask]
            out = pd.concat([old, new], ignore_index=True)
        else:
            out = new

        out.to_csv(self.path, index=False)


# =============================================================================
# 13. VISUALIZACIÓN
# =============================================================================

class Visualizer:
    @staticmethod
    def plot_equity(backtest_df: pd.DataFrame, title: str, path: str):
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_df["date"], backtest_df["buy_hold_equity"], label="Buy & Hold")
        plt.plot(backtest_df["date"], backtest_df["strategy_equity"], label="Ensemble ML")
        plt.title(title)
        plt.xlabel("Fecha")
        plt.ylabel("Capital")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=250)
        plt.show()

    @staticmethod
    def plot_probability(backtest_df: pd.DataFrame, title: str, path: str, config: Config):
        plt.figure(figsize=(12, 5))
        plt.plot(backtest_df["date"], backtest_df["ensemble_probability"], label="P(SUBIDA)")
        plt.axhline(config.buy_threshold, linestyle="--", label="Umbral alcista")
        plt.axhline(config.sell_threshold, linestyle="--", label="Umbral bajista")
        plt.axhline(0.50, linestyle=":", label="Neutral")
        plt.ylim(0, 1)
        plt.title(title)
        plt.xlabel("Fecha")
        plt.ylabel("Probabilidad")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=250)
        plt.show()

    @staticmethod
    def plot_clean_price(market_df: pd.DataFrame, ticker: str, path: str):
        plt.figure(figsize=(12, 5))
        plt.plot(market_df["date"], market_df["Close"])
        plt.title(f"{ticker} - Precio de cierre después del preprocesamiento")
        plt.xlabel("Fecha")
        plt.ylabel("Precio")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=250)
        plt.show()


# =============================================================================
# 14. UTILIDADES DE SEÑAL
# =============================================================================

def probability_to_signal(prob: float, buy_threshold: float, sell_threshold: float) -> str:
    if prob >= buy_threshold:
        return "SUBIDA / ALCISTA"
    if prob <= sell_threshold:
        return "BAJADA / BAJISTA"
    return "NEUTRAL / BAJA CONFIANZA"


def confidence_label(prob: float) -> str:
    distance = abs(prob - 0.5)
    if distance >= 0.25:
        return "ALTA"
    if distance >= 0.12:
        return "MODERADA"
    return "BAJA"


# =============================================================================
# 15. ESTUDIO DE ABLACIÓN NLP: CON vs SIN SENTIMIENTO
# =============================================================================

class SentimentAblationStudy:
    """
    Compara el valor añadido del sentimiento sin cambiar las fechas del
    experimento.

    Para las dos variantes se utilizan:
      - las mismas observaciones;
      - el mismo train;
      - el mismo purge;
      - el mismo holdout final;
      - TimeSeriesSplit con el mismo gap.

    Si el histórico NLP todavía no es apto, NO se fuerza el estudio usando
    ceros artificiales: se deja un fichero de estado y se continúa.
    """

    def __init__(self, config, classical, lstm):
        self.config = config
        self.classical = classical
        self.lstm = lstm

    def run(
        self,
        ticker,
        horizon,
        train_df,
        test_df,
        target_col,
        full_feature_cols,
        horizon_dir,
    ):
        status_path = os.path.join(
            horizon_dir, "sentiment_ablation_status.txt"
        )

        nlp_cols = [
            c for c in FeatureEngineer.SENTIMENT_FEATURES
            if c in full_feature_cols
        ]

        if not self.config.compare_sentiment_models:
            Path(status_path).write_text(
                "Estudio desactivado por configuración.",
                encoding="utf-8",
            )
            return None

        if not nlp_cols:
            message = (
                "NO EJECUTADO: el sentimiento histórico todavía no entra "
                "en el modelo. Se esperará a disponer de cobertura NLP real "
                "suficiente; no se simula sentimiento histórico con ceros."
            )
            print("\\nESTUDIO CON vs SIN SENTIMIENTO:")
            print(message)
            Path(status_path).write_text(message, encoding="utf-8")
            return None

        without_cols = [
            c for c in full_feature_cols if c not in nlp_cols
        ]
        variants = {
            "SIN_SENTIMIENTO": without_cols,
            "CON_SENTIMIENTO": list(full_feature_cols),
        }

        rows = []

        print("\\n" + "=" * 118)
        print(
            f"ABLACIÓN NLP {ticker} - HORIZONTE {horizon}d "
            "(mismas fechas y holdout)"
        )
        print("=" * 118)

        for variant, cols in variants.items():
            print(f"\\nVARIANTE: {variant}")

            X_train = train_df[cols]
            y_train = train_df[target_col].astype(int)
            X_test = test_df[cols]
            y_test = test_df[target_col].astype(int)

            cv_results, weights = self.classical.cross_validate(
                X_train, y_train, horizon
            )
            models = self.classical.fit_final(X_train, y_train)

            probs = {}
            for model_name, model in models.items():
                prob = model.predict_proba(X_test)[:, 1]
                pred = (prob >= 0.5).astype(int)
                m = Metrics.classification(y_test, pred, prob)
                m.update({
                    "variant": variant,
                    "model": model_name,
                    "n_features": len(cols),
                })
                rows.append(m)
                probs[model_name] = prob

            if (
                self.config.compare_lstm_sentiment
                and self.lstm.enabled()
            ):
                (
                    _model,
                    _scaler,
                    lstm_val,
                    lstm_test,
                    lstm_prob,
                ) = self.lstm.train_validate_and_test(
                    X_train, y_train, X_test, y_test
                )

                if lstm_val is not None:
                    raw = dict(weights)
                    raw["LSTM"] = Metrics.model_weight_score(lstm_val)
                    weights = ClassicalModelTrainer.normalize_weights(raw)

                if lstm_test is not None and lstm_prob is not None:
                    m = dict(lstm_test)
                    m.update({
                        "variant": variant,
                        "model": "LSTM",
                        "n_features": len(cols),
                    })
                    rows.append(m)
                    probs["LSTM"] = lstm_prob

            ensemble_prob = EnsemblePredictor.combine_arrays(
                probs, weights
            )
            ensemble_pred = (ensemble_prob >= 0.5).astype(int)
            em = Metrics.classification(
                y_test, ensemble_pred, ensemble_prob
            )
            em.update({
                "variant": variant,
                "model": "ENSEMBLE",
                "n_features": len(cols),
            })
            rows.append(em)

            print(
                f"ENSEMBLE {variant}: "
                f"AUC={em['roc_auc']:.4f} | "
                f"BalAcc={em['balanced_accuracy']:.4f} | "
                f"F1={em['f1']:.4f} | "
                f"Brier={em['brier']:.4f}"
            )

        result = pd.DataFrame(rows)

        preferred = [
            "variant", "model", "n_features",
            "accuracy", "balanced_accuracy",
            "precision", "recall", "f1",
            "roc_auc", "log_loss", "brier",
            "positive_rate",
        ]
        result = result[
            [c for c in preferred if c in result.columns]
        ]

        result.to_csv(
            os.path.join(
                horizon_dir, "sentiment_ablation_models.csv"
            ),
            index=False,
        )

        # Tabla de deltas: CON - SIN.
        delta_rows = []
        for model_name in result["model"].unique():
            sub = result[
                result["model"] == model_name
            ].set_index("variant")

            if (
                "CON_SENTIMIENTO" not in sub.index
                or "SIN_SENTIMIENTO" not in sub.index
            ):
                continue

            row = {"model": model_name}
            for metric in [
                "accuracy", "balanced_accuracy", "f1",
                "roc_auc", "log_loss", "brier",
            ]:
                sin = float(sub.loc["SIN_SENTIMIENTO", metric])
                con = float(sub.loc["CON_SENTIMIENTO", metric])
                row[f"sin_{metric}"] = sin
                row[f"con_{metric}"] = con
                row[f"delta_{metric}"] = con - sin
            delta_rows.append(row)

        delta = pd.DataFrame(delta_rows)
        delta.to_csv(
            os.path.join(
                horizon_dir, "sentiment_ablation_delta.csv"
            ),
            index=False,
        )

        # Figura AUC con/sin NLP.
        try:
            pivot = result.pivot(
                index="model",
                columns="variant",
                values="roc_auc",
            )
            ax = pivot.plot(kind="bar", figsize=(11, 6))
            ax.axhline(
                0.5, linestyle="--", linewidth=1,
                label="Azar (AUC=0.5)"
            )
            ax.set_title(
                f"{ticker} - Con vs sin sentimiento - "
                f"Horizonte {horizon}d"
            )
            ax.set_ylabel("ROC-AUC holdout")
            ax.set_xlabel("Modelo")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    horizon_dir,
                    "sentiment_ablation_auc.png",
                ),
                dpi=250,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as exc:
            print(f"Gráfica ablación NLP no creada: {exc}")

        Path(status_path).write_text(
            (
                "EJECUTADO. Comparación con las mismas fechas, "
                "train, purge y holdout.\\n"
                f"Features NLP: {nlp_cols}\\n"
            ),
            encoding="utf-8",
        )

        if not delta.empty:
            print("\\nDELTA CON - SIN SENTIMIENTO:")
            show = [
                c for c in [
                    "model",
                    "delta_roc_auc",
                    "delta_balanced_accuracy",
                    "delta_f1",
                    "delta_brier",
                ] if c in delta.columns
            ]
            print(delta[show].to_string(index=False))

        return result


# =============================================================================
# 16. IMPORTANCIA DE VARIABLES / SHAP
# =============================================================================

class ExplainabilityAnalyzer:
    """
    Explica los modelos ya entrenados.

    - XGBoost: SHAP global (mean |SHAP|, bar y beeswarm).
    - Random Forest: feature_importances_.
    - Logistic Regression: coeficientes del modelo tras StandardScaler.
    - XGBoost: importancia interna adicional.

    El holdout se utiliza solo para explicar predicciones ya generadas;
    SHAP NO se usa para elegir variables ni reentrenar el modelo.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        ticker,
        horizon,
        models,
        X_train,
        X_test,
        horizon_dir,
    ):
        out_dir = os.path.join(horizon_dir, "explainability")
        os.makedirs(out_dir, exist_ok=True)

        self._random_forest(models, X_train.columns, out_dir)
        self._logistic(models, X_train.columns, out_dir)
        self._xgb_builtin(models, X_train.columns, out_dir)

        result = {
            "shap_available": SHAP_AVAILABLE,
            "shap_executed": False,
        }

        if not self.config.use_shap:
            return result

        if not SHAP_AVAILABLE:
            msg = "SHAP no instalado. Ejecuta: pip install shap"
            Path(
                os.path.join(out_dir, "shap_status.txt")
            ).write_text(msg, encoding="utf-8")
            if self.config.require_shap:
                raise RuntimeError(msg)
            return result

        if "XGBoost" not in models:
            Path(
                os.path.join(out_dir, "shap_status.txt")
            ).write_text(
                "No hay XGBoost para explicar.",
                encoding="utf-8",
            )
            return result

        X_shap = X_test.copy()
        if len(X_shap) > self.config.shap_max_samples:
            X_shap = X_shap.sample(
                self.config.shap_max_samples,
                random_state=42,
            ).sort_index()

        print(
            f"\\nSHAP {ticker} {horizon}d: "
            f"{len(X_shap)} muestras, "
            f"{X_shap.shape[1]} variables."
        )

        try:
            model = models["XGBoost"]

            try:
                explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(X_shap)
            except Exception:
                explainer = shap.Explainer(model)
                explanation = explainer(X_shap)
                values = explanation.values

            if isinstance(values, list):
                values = values[-1]

            values = np.asarray(values)
            if values.ndim == 3:
                values = values[:, :, -1]

            if values.shape[1] != X_shap.shape[1]:
                raise ValueError(
                    "Dimensiones SHAP incompatibles con las features."
                )

            importance = pd.DataFrame({
                "feature": X_shap.columns,
                "mean_abs_shap": np.mean(
                    np.abs(values), axis=0
                ),
            }).sort_values(
                "mean_abs_shap", ascending=False
            )

            total = importance["mean_abs_shap"].sum()
            importance["importance_pct"] = (
                importance["mean_abs_shap"]
                / total * 100
                if total > 0 else 0.0
            )
            importance["rank"] = np.arange(
                1, len(importance) + 1
            )

            importance.to_csv(
                os.path.join(
                    out_dir, "shap_feature_importance.csv"
                ),
                index=False,
            )

            top_n = min(
                self.config.shap_top_features,
                X_shap.shape[1],
            )

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                values,
                X_shap,
                plot_type="bar",
                max_display=top_n,
                show=False,
            )
            plt.title(
                f"{ticker} - SHAP XGBoost - {horizon}d"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    out_dir, "shap_summary_bar.png"
                ),
                dpi=250,
                bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(11, 9))
            shap.summary_plot(
                values,
                X_shap,
                max_display=top_n,
                show=False,
            )
            plt.title(
                f"{ticker} - SHAP beeswarm XGBoost - "
                f"{horizon}d"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    out_dir, "shap_summary_beeswarm.png"
                ),
                dpi=250,
                bbox_inches="tight",
            )
            plt.close()

            print("\\nTOP VARIABLES SHAP:")
            print(
                importance.head(top_n).to_string(index=False)
            )

            Path(
                os.path.join(out_dir, "shap_status.txt")
            ).write_text(
                (
                    "SHAP ejecutado correctamente sobre XGBoost.\\n"
                    f"Muestras: {len(X_shap)}\\n"
                    f"Variables: {X_shap.shape[1]}\\n"
                ),
                encoding="utf-8",
            )

            result["shap_executed"] = True
            return result

        except Exception as exc:
            msg = (
                f"SHAP no pudo ejecutarse: "
                f"{type(exc).__name__}: {exc}"
            )
            print(msg)
            Path(
                os.path.join(out_dir, "shap_status.txt")
            ).write_text(msg, encoding="utf-8")
            return result

    @staticmethod
    def _random_forest(models, feature_cols, out_dir):
        model = models.get("RandomForest")
        if model is None or not hasattr(
            model, "feature_importances_"
        ):
            return

        df = pd.DataFrame({
            "feature": list(feature_cols),
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        df["rank"] = np.arange(1, len(df) + 1)
        df.to_csv(
            os.path.join(
                out_dir,
                "random_forest_feature_importance.csv",
            ),
            index=False,
        )

    @staticmethod
    def _logistic(models, feature_cols, out_dir):
        pipeline = models.get("LogisticRegression")
        if pipeline is None:
            return
        try:
            coef = (
                pipeline.named_steps["model"]
                .coef_.ravel()
            )
            df = pd.DataFrame({
                "feature": list(feature_cols),
                "coefficient": coef,
                "abs_coefficient": np.abs(coef),
                "direction": np.where(
                    coef >= 0,
                    "AUMENTA_P_SUBIDA",
                    "REDUCE_P_SUBIDA",
                ),
            }).sort_values(
                "abs_coefficient", ascending=False
            )
            df["rank"] = np.arange(1, len(df) + 1)
            df.to_csv(
                os.path.join(
                    out_dir, "logistic_coefficients.csv"
                ),
                index=False,
            )
        except Exception:
            pass

    @staticmethod
    def _xgb_builtin(models, feature_cols, out_dir):
        model = models.get("XGBoost")
        if model is None or not hasattr(
            model, "feature_importances_"
        ):
            return

        df = pd.DataFrame({
            "feature": list(feature_cols),
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        df["rank"] = np.arange(1, len(df) + 1)
        df.to_csv(
            os.path.join(
                out_dir,
                "xgboost_builtin_feature_importance.csv",
            ),
            index=False,
        )


# =============================================================================
# 17. PIPELINE PRINCIPAL
# =============================================================================


# =============================================================================
# FASE FINAL DE EXPERIMENTACIÓN
# =============================================================================

class FinalExperimentSuite:
    """
    Experimentos de cierre del TFM.

    1) Ablación incremental de grupos de variables:
       TECHNICAL -> TECHNICAL+MACRO -> TECHNICAL+MACRO+NLP

    2) Robustez frente a niveles absolutos de tipos:
       FULL vs FULL_WITHOUT_RATE_LEVELS

    3) Baseline técnico tradicional MA50/MA200.

    Para evitar multiplicar innecesariamente el coste computacional, estas
    ablaciones usan el ensemble CLÁSICO (Logistic/RF/XGBoost). La comparación
    LSTM ya se realiza en el experimento principal y en la ablación NLP.
    """

    RATE_LEVEL_FEATURES = [
        "ext_fedfunds_level",
        "ext_us2y_level",
        "ext_us10y_level",
    ]

    def __init__(
        self,
        config: Config,
        classical: ClassicalModelTrainer,
        backtester: Backtester,
    ):
        self.config = config
        self.classical = classical
        self.backtester = backtester

    def _evaluate_feature_set(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        horizon: int,
        label: str,
    ) -> Dict[str, Any]:
        cols = [c for c in feature_cols if c in train_df.columns]
        if not cols:
            raise ValueError(f"{label}: no hay features disponibles.")

        X_train = train_df[cols]
        y_train = train_df[target_col].astype(int)
        X_test = test_df[cols]
        y_test = test_df[target_col].astype(int)

        cv_results, weights = self.classical.cross_validate(X_train, y_train, horizon)
        models = self.classical.fit_final(X_train, y_train)

        probs = {}
        model_rows = []
        for name, model in models.items():
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = Metrics.classification(y_test, pred, prob)
            m.update({"variant": label, "model": name, "n_features": len(cols)})
            model_rows.append(m)
            probs[name] = prob

        ensemble_prob = EnsemblePredictor.combine_arrays(probs, weights)
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        metrics = Metrics.classification(y_test, ensemble_pred, ensemble_prob)
        metrics.update({
            "variant": label,
            "model": "CLASSICAL_ENSEMBLE",
            "n_features": len(cols),
        })

        bt = self.backtester.run(test_df, ensemble_prob)
        fin = self.backtester.metrics(bt)
        metrics.update({
            "strategy_total_return": fin["strategy_total_return"],
            "strategy_sharpe": fin["strategy_sharpe"],
            "strategy_max_drawdown": fin["strategy_max_drawdown"],
            "market_exposure": fin["market_exposure"],
        })
        model_rows.append(dict(metrics))

        return {
            "ensemble": metrics,
            "models": pd.DataFrame(model_rows),
            "cv": cv_results,
            "probabilities": ensemble_prob,
            "features": cols,
        }

    def run(
        self,
        ticker: str,
        horizon: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        full_feature_cols: List[str],
        horizon_dir: str,
    ) -> Dict[str, Any]:
        if (
            not self.config.run_final_experiments
            or horizon not in self.config.final_experiment_horizons
        ):
            return {}

        final_dir = os.path.join(horizon_dir, "final_experiments")
        os.makedirs(final_dir, exist_ok=True)
        outputs: Dict[str, Any] = {}

        technical = [
            c for c in FeatureEngineer.TECHNICAL_FEATURES
            if c in full_feature_cols
        ]
        macro = [
            c for c in FeatureEngineer.EXTERNAL_FEATURES
            if c in full_feature_cols
        ]
        nlp = [
            c for c in FeatureEngineer.SENTIMENT_FEATURES
            if c in full_feature_cols
        ]

        # --------------------------------------------------------------
        # 1) TECHNICAL vs TECH+MACRO vs FULL
        # --------------------------------------------------------------
        if self.config.run_feature_group_ablation:
            variants = {
                "TECHNICAL": technical,
                "TECHNICAL_MACRO": technical + macro,
            }
            if nlp:
                variants["TECHNICAL_MACRO_NLP"] = technical + macro + nlp

            rows = []
            detail_frames = []
            for label, cols in variants.items():
                print("\n" + "=" * 100)
                print(f"ABLACIÓN FINAL {ticker} {horizon}d: {label}")
                print("=" * 100)
                result = self._evaluate_feature_set(
                    train_df, test_df, target_col, cols, horizon, label
                )
                rows.append(result["ensemble"])
                detail_frames.append(result["models"])

            ablation = pd.DataFrame(rows)
            ablation.to_csv(
                os.path.join(final_dir, "feature_group_ablation.csv"),
                index=False,
            )
            pd.concat(detail_frames, ignore_index=True).to_csv(
                os.path.join(final_dir, "feature_group_ablation_models.csv"),
                index=False,
            )
            outputs["feature_group_ablation"] = ablation

            try:
                plot = ablation.set_index("variant")[["roc_auc", "balanced_accuracy", "f1"]]
                ax = plot.plot(kind="bar", figsize=(11, 6))
                ax.set_title(f"{ticker} - Ablación de features - {horizon}d")
                ax.set_ylabel("Métrica")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(final_dir, "feature_group_ablation.png"),
                    dpi=250,
                    bbox_inches="tight",
                )
                plt.close()
            except Exception as exc:
                print(f"No se pudo graficar ablación final: {exc}")

        # --------------------------------------------------------------
        # 2) Robustez sin niveles absolutos de tipos
        # --------------------------------------------------------------
        if self.config.run_rate_level_robustness:
            with_levels = list(full_feature_cols)
            without_levels = [
                c for c in full_feature_cols
                if c not in self.RATE_LEVEL_FEATURES
            ]

            rate_rows = []
            for label, cols in [
                ("FULL_WITH_RATE_LEVELS", with_levels),
                ("FULL_WITHOUT_RATE_LEVELS", without_levels),
            ]:
                result = self._evaluate_feature_set(
                    train_df, test_df, target_col, cols, horizon, label
                )
                rate_rows.append(result["ensemble"])

            rate_df = pd.DataFrame(rate_rows)
            if len(rate_df) == 2:
                a = rate_df.iloc[0]
                b = rate_df.iloc[1]
                delta = {
                    "delta_auc_without_minus_with": b["roc_auc"] - a["roc_auc"],
                    "delta_balacc_without_minus_with": b["balanced_accuracy"] - a["balanced_accuracy"],
                    "delta_f1_without_minus_with": b["f1"] - a["f1"],
                    "delta_sharpe_without_minus_with": b["strategy_sharpe"] - a["strategy_sharpe"],
                }
            else:
                delta = {}

            rate_df.to_csv(
                os.path.join(final_dir, "rate_level_robustness.csv"),
                index=False,
            )
            pd.DataFrame([delta]).to_csv(
                os.path.join(final_dir, "rate_level_robustness_delta.csv"),
                index=False,
            )
            outputs["rate_robustness"] = rate_df

        # --------------------------------------------------------------
        # 3) Baseline MA50/MA200
        # --------------------------------------------------------------
        if self.config.run_ma50_ma200_baseline:
            if "ma_50" in test_df.columns and "ma_200" in test_df.columns:
                probability = np.where(
                    test_df["ma_50"].values > test_df["ma_200"].values,
                    0.70,
                    0.30,
                ).astype(float)
                pred = (probability >= 0.5).astype(int)
                y_test = test_df[target_col].astype(int).values
                classification = Metrics.classification(y_test, pred, probability)
                bt = self.backtester.run(test_df, probability)
                financial = self.backtester.metrics(bt)

                row = {
                    "baseline": "MA50_MA200",
                    **classification,
                    **financial,
                }
                pd.DataFrame([row]).to_csv(
                    os.path.join(final_dir, "ma50_ma200_baseline.csv"),
                    index=False,
                )
                bt.to_csv(
                    os.path.join(final_dir, "ma50_ma200_backtest.csv"),
                    index=False,
                )
                outputs["ma_baseline"] = row

        return outputs


class AdvancedETFPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.sentiment_history = HistoricalSentimentLoader(config)
        self.current_news = NewsSentimentAnalyzer(config)
        self.alpha_history = AlphaVantageHistoricalNewsBackfiller(
            config,
            self.current_news,
        )
        self.guardian_history = GuardianHistoricalNewsBackfiller(
            config,
            self.current_news,
        )
        self.multisource_history = MultiSourceHistoricalSentimentBuilder(config)
        self.external_market = ExternalMarketDataLoader(config)
        self.classical = ClassicalModelTrainer(config)
        self.price_regression = PriceRegressionTrainer(config)
        self.lstm = LSTMTrainer(config)
        self.backtester = Backtester(config)
        self.final_experiments = FinalExperimentSuite(
            config, self.classical, self.backtester
        )
        self.overlap_validator = OverlappingTargetValidator(config)
        self.journal = PredictionJournal(config)
        self.sentiment_ablation = SentimentAblationStudy(
            config, self.classical, self.lstm
        )
        self.explainability = ExplainabilityAnalyzer(config)

    def run_ticker(self, ticker: str) -> Dict[int, Dict[str, Any]]:
        ticker = ticker.upper()
        print("\n" + "=" * 118)
        print(f"ANÁLISIS AVANZADO: {ticker}")
        print("=" * 118)

        # 1) Preprocesamiento y auditoría.
        market_df, preprocessing_report = self.preprocessor.load_and_clean(ticker)
        self.journal.update_realized_results(ticker, market_df)

        ticker_dir = os.path.join(self.config.output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        Visualizer.plot_clean_price(
            market_df,
            ticker,
            os.path.join(ticker_dir, "clean_price_curve.png"),
        )

        # 2) Histórico largo: Alpha Vantage + FinBERT.
        # Es reanudable y guarda los bloques ya completados.
        self.alpha_history.update(
            ticker,
            market_dates=market_df["date"],
        )

        # Segunda fuente independiente. Si Alpha alcanza cuota, Guardian puede
        # seguir ampliando el histórico y viceversa.
        self.guardian_history.update(
            ticker,
            market_dates=market_df["date"],
        )

        # Unificamos los dos proveedores a nivel de artículo ANTES de crear las
        # features diarias.
        self.multisource_history.build(ticker)

        # 3) Google News + FinBERT para noticias recientes. Estas noticias
        # tienen prioridad en las fechas recientes al fusionarse.
        current_sentiment, news_df = self.current_news.fetch_score_and_update_history(ticker)
        if not news_df.empty:
            news_df.to_csv(
                os.path.join(ticker_dir, "noticias_recientes_finbert.csv"),
                index=False,
            )

        # 4) Cargamos el histórico NLP ya fusionado GDELT + FinBERT.
        historical_sentiment = self.sentiment_history.load(
            ticker,
            market_dates=market_df["date"],
        )

        # 5) Variables externas de mercado y macroeconomía.
        external_features = self.external_market.build(market_df["date"])

        # 6) Feature engineering conjunto:
        # precio ETF + mercado/macro + NLP.
        features, feature_cols = FeatureEngineer.build(
            market_df,
            historical_sentiment=historical_sentiment,
            external_features=external_features,
        )

        # Congelación reproducible de la matriz experimental.
        frozen_path = os.path.join(
            self.config.frozen_data_dir, f"{ticker.lower()}_features_frozen.csv"
        )
        frozen_meta_path = os.path.join(
            self.config.frozen_data_dir, f"{ticker.lower()}_features_frozen.json"
        )

        if self.config.reuse_frozen_dataset and os.path.exists(frozen_path):
            print(f"Reutilizando dataset experimental congelado: {frozen_path}")
            features = pd.read_csv(frozen_path)
            features["date"] = pd.to_datetime(features["date"], errors="coerce")
            if os.path.exists(frozen_meta_path):
                try:
                    meta = json.loads(Path(frozen_meta_path).read_text(encoding="utf-8"))
                    feature_cols = [c for c in meta.get("feature_cols", feature_cols) if c in features.columns]
                except Exception:
                    pass
        elif self.config.freeze_experimental_dataset:
            features.to_csv(frozen_path, index=False)
            digest = hashlib.sha256(Path(frozen_path).read_bytes()).hexdigest()
            Path(frozen_meta_path).write_text(
                json.dumps({
                    "ticker": ticker,
                    "created_at": datetime.now().isoformat(),
                    "sha256": digest,
                    "rows": len(features),
                    "feature_cols": feature_cols,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Dataset experimental congelado: {frozen_path}")
            print(f"SHA256 dataset: {digest}")

        sentiment_in_model = historical_sentiment is not None and not historical_sentiment.empty
        print(f"Sentimiento en modelo predictivo: {'SÍ' if sentiment_in_model else 'NO'}")
        print(
            f"Sentimiento actual de contexto: {current_sentiment.get('sentiment_mean', 0.0):+.3f} "
            f"({int(current_sentiment.get('news_count', 0))} noticias)"
        )

        outputs = {}
        for horizon in self.config.horizons:
            outputs[horizon] = self._run_horizon(
                ticker=ticker,
                features=features,
                feature_cols=feature_cols,
                horizon=horizon,
                current_sentiment=current_sentiment,
                sentiment_in_model=sentiment_in_model,
                ticker_dir=ticker_dir,
            )
        return outputs

    def _run_horizon(
        self,
        ticker: str,
        features: pd.DataFrame,
        feature_cols: List[str],
        horizon: int,
        current_sentiment: Dict[str, float],
        sentiment_in_model: bool,
        ticker_dir: str,
    ) -> Dict[str, Any]:
        print("\n" + "-" * 118)
        print(f"{ticker} - HORIZONTE {horizon} SESIÓN/SESIONES")
        print("-" * 118)

        experiment_seed = Reproducibility.stable_seed(
            self.config.random_seed, ticker, horizon
        )
        Reproducibility.apply(
            experiment_seed, self.config.deterministic_tensorflow
        )
        print(f"Semilla reproducible experimento: {experiment_seed}")

        df = FeatureEngineer.add_target(features, horizon)
        target_col = f"target_{horizon}d"

        model_df = df.dropna(subset=feature_cols + [target_col]).copy().reset_index(drop=True)

        if len(model_df) < 400:
            raise ValueError(f"{ticker} horizonte {horizon}: dataset supervisado demasiado pequeño ({len(model_df)}).")

        # Holdout cronológico. PURGE: las últimas horizon filas del train se eliminan
        # para que su etiqueta no dependa de precios pertenecientes al holdout.
        split_idx = int(len(model_df) * (1 - self.config.test_size_ratio))
        raw_train_df = model_df.iloc[:split_idx].copy()
        test_df = model_df.iloc[split_idx:].copy()

        if len(raw_train_df) <= horizon:
            raise ValueError("Train insuficiente para aplicar purge temporal.")

        train_df = raw_train_df.iloc[:-horizon].copy()

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df[target_col].astype(int)

        print(f"Dataset supervisado: {len(model_df)} filas")
        print(f"Train purgado:       {len(train_df)} filas")
        print(f"Gap/purge:           {horizon} filas")
        print(f"Holdout final:       {len(test_df)} filas")
        print(f"Tasa de subidas train: {y_train.mean():.3f}")
        print(f"Tasa de subidas test:  {y_test.mean():.3f}")

        # 1) CV temporal dentro del train, con gap=horizon.
        cv_results, weights = self.classical.cross_validate(X_train, y_train, horizon)
        print("\nResumen validación cruzada:")
        print(cv_results.to_string(index=False))
        print("\nPesos por validación (sin usar holdout):")
        for k, v in weights.items():
            print(f"  {k}: {v:.3f}")

        # 2) Modelos clásicos finales.
        models = self.classical.fit_final(X_train, y_train)
        holdout_rows = []
        holdout_probabilities: Dict[str, np.ndarray] = {}

        for name, model in models.items():
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = Metrics.classification(y_test, pred, prob)
            m["model"] = name
            holdout_rows.append(m)
            holdout_probabilities[name] = prob

        # 3) LSTM: peso derivado de validación interna del train, no del holdout.
        lstm_model = None
        lstm_scaler = None
        lstm_val_metrics = None
        lstm_test_metrics = None
        lstm_prob = None

        if self.lstm.enabled():
            (
                lstm_model,
                lstm_scaler,
                lstm_val_metrics,
                lstm_test_metrics,
                lstm_prob,
            ) = self.lstm.train_validate_and_test(X_train, y_train, X_test, y_test)

            if lstm_val_metrics is not None:
                raw_weight_values = {k: v for k, v in weights.items()}
                # Convertimos pesos normalizados clásicos a scores relativos y añadimos LSTM.
                raw_weight_values["LSTM"] = Metrics.model_weight_score(lstm_val_metrics)
                weights = ClassicalModelTrainer.normalize_weights(raw_weight_values)

            if lstm_test_metrics is not None:
                row = dict(lstm_test_metrics)
                row["model"] = "LSTM"
                holdout_rows.append(row)
                holdout_probabilities["LSTM"] = lstm_prob
        else:
            if self.config.use_lstm:
                print("LSTM solicitada pero TensorFlow no está disponible. Se continúa sin LSTM.")

        holdout_results = pd.DataFrame(holdout_rows)
        columns = [
            "model", "accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "roc_auc", "log_loss", "brier", "positive_rate",
        ]
        holdout_results = holdout_results[columns].sort_values(["roc_auc", "balanced_accuracy"], ascending=False)

        print("\nEvaluación OUT-OF-SAMPLE (holdout temporal NO usado para pesos):")
        print(holdout_results.to_string(index=False))
        print("\nPesos finales del ensemble:")
        for k, v in weights.items():
            print(f"  {k}: {v:.3f}")

        # 4) Ensemble holdout totalmente independiente.
        ensemble_holdout = EnsemblePredictor.combine_arrays(holdout_probabilities, weights)
        ensemble_pred = (ensemble_holdout >= 0.5).astype(int)
        ensemble_metrics = Metrics.classification(y_test, ensemble_pred, ensemble_holdout)

        print("\nENSEMBLE OUT-OF-SAMPLE:")
        for k, v in ensemble_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Baseline muy importante para interpretar F1/AUC.
        majority_class = int(y_train.mean() >= 0.5)
        baseline_pred = np.full(len(y_test), majority_class)
        baseline_prob = np.full(len(y_test), y_train.mean())
        baseline_metrics = Metrics.classification(y_test, baseline_pred, baseline_prob)
        print("\nBASELINE (clase mayoritaria / probabilidad histórica):")
        for k, v in baseline_metrics.items():
            print(f"  {k}: {v:.4f}")

        # 5) Backtest.
        backtest_df = self.backtester.run(test_df, ensemble_holdout)
        financial_metrics = self.backtester.metrics(backtest_df)
        print("\nMÉTRICAS FINANCIERAS:")
        for k, v in financial_metrics.items():
            if isinstance(v, (int, np.integer)):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}")

        # 6) Predicción de última observación disponible.
        latest_df = df.dropna(subset=feature_cols).iloc[[-1]].copy()

        # Solo si el modelo fue entrenado con sentimiento histórico se actualizan
        # esas features con el sentimiento actual. Si no, solo se muestra como contexto.
        if sentiment_in_model:
            for col in FeatureEngineer.SENTIMENT_FEATURES:
                latest_df.loc[:, col] = current_sentiment.get(col, 0.0)

        X_latest = latest_df[feature_cols]
        latest_model_probabilities: Dict[str, float] = {}
        for name, model in models.items():
            latest_model_probabilities[name] = float(model.predict_proba(X_latest)[:, 1][0])

        if lstm_model is not None:
            X_all_for_lstm = df.dropna(subset=feature_cols)[feature_cols].copy()
            if sentiment_in_model:
                for col in FeatureEngineer.SENTIMENT_FEATURES:
                    X_all_for_lstm.iloc[-1, X_all_for_lstm.columns.get_loc(col)] = current_sentiment.get(col, 0.0)
            lstm_latest = self.lstm.predict_latest(lstm_model, lstm_scaler, X_all_for_lstm)
            if lstm_latest is not None:
                latest_model_probabilities["LSTM"] = lstm_latest

        ensemble_latest, used_weights = EnsemblePredictor.combine_scalar(latest_model_probabilities, weights)
        signal = probability_to_signal(ensemble_latest, self.config.buy_threshold, self.config.sell_threshold)
        confidence = confidence_label(ensemble_latest)
        latest_date = pd.to_datetime(latest_df["date"].iloc[0])

        print("\n" + "*" * 118)
        print(f"PREDICCIÓN FINAL {ticker} | Horizonte: {horizon} sesión/sesiones")
        print("*" * 118)
        print(f"Fecha del último dato:      {latest_date.date()}")
        print(f"Precio de cierre:           {latest_df['Close'].iloc[0]:.4f}")
        for name, prob in latest_model_probabilities.items():
            print(f"{name:27s} P(SUBE) = {prob * 100:6.2f}%")
        print(f"ENSEMBLE P(SUBE):           {ensemble_latest * 100:6.2f}%")
        print(f"ENSEMBLE P(BAJA):           {(1 - ensemble_latest) * 100:6.2f}%")
        print(f"SEÑAL:                      {signal}")
        print(f"CONFIANZA:                  {confidence}")
        print(f"SENTIMIENTO ACTUAL CONTEXTO:{current_sentiment.get('sentiment_mean', 0.0):+7.3f}")
        print(f"Nº NOTICIAS ACTUALES:       {int(current_sentiment.get('news_count', 0))}")
        print(f"SENTIMIENTO DENTRO MODELO:  {'SÍ' if sentiment_in_model else 'NO'}")
        print("*" * 118)

        # 7) Persistencia.
        horizon_dir = os.path.join(ticker_dir, f"horizon_{horizon}d")
        os.makedirs(horizon_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # SEGUNDO PROBLEMA SUPERVISADO: REGRESIÓN DEL PRECIO FUTURO
        # ------------------------------------------------------------------
        regression_result = self.price_regression.run(
            ticker=ticker,
            horizon=horizon,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            latest_df=latest_df,
            horizon_dir=horizon_dir,
        )

        combined_decision = {}
        if regression_result:
            reg_latest = regression_result["latest"]

            classification_bullish = "ALCISTA" in signal
            classification_bearish = "BAJISTA" in signal
            regression_bullish = reg_latest["price_bias"] == "ALCISTA"
            regression_bearish = reg_latest["price_bias"] == "BAJISTA"

            if classification_bullish and regression_bullish:
                experimental_action = "ALCISTA_CONFIRMADA"
            elif classification_bearish and regression_bearish:
                experimental_action = "BAJISTA_CONFIRMADA"
            elif classification_bullish or regression_bullish:
                experimental_action = "ALCISTA_NO_CONFIRMADA"
            elif classification_bearish or regression_bearish:
                experimental_action = "BAJISTA_NO_CONFIRMADA"
            else:
                experimental_action = "NEUTRAL"

            combined_decision = {
                "ticker": ticker,
                "horizon": horizon,
                "classification_signal": signal,
                "classification_probability_up": ensemble_latest,
                "regression_price_bias": reg_latest["price_bias"],
                "experimental_action": experimental_action,
                "current_close_entry_reference": reg_latest["current_close"],
                "max_entry_price_for_edge": reg_latest["max_entry_price_for_edge"],
                "future_target_price_reference": reg_latest["predicted_future_price"],
                "risk_reference_80_low": reg_latest["risk_reference_80_low"],
                "interval80_low": reg_latest["interval80_low"],
                "interval80_high": reg_latest["interval80_high"],
                "expected_return": reg_latest["expected_return"],
                "selected_regression_model": reg_latest["selected_regression_model"],
                "rmse": reg_latest["holdout_rmse"],
                "persistence_rmse": reg_latest["persistence_holdout_rmse"],
                "rmse_skill_vs_persistence": reg_latest["regression_rmse_skill_vs_persistence"],
            }

            decision_path = os.path.join(
                horizon_dir, "price_regression", "classification_regression_decision.csv"
            )
            pd.DataFrame([combined_decision]).to_csv(decision_path, index=False)

            print("\nCOMBINACIÓN CLASIFICACIÓN + REGRESIÓN")
            print(f"Señal clasificación:         {signal}")
            print(f"P(subida):                   {ensemble_latest * 100:.2f}%")
            print(f"Modelo regresión elegido:     {reg_latest['selected_regression_model']}")
            print(f"Sesgo regresión:             {reg_latest['price_bias']}")
            print(f"Acción experimental:         {experimental_action}")
            print(f"Precio actual / entrada ref.:{reg_latest['current_close']:.4f}")
            print(f"Entrada máxima ref. edge:    {reg_latest['max_entry_price_for_edge']:.4f}")
            print(f"Precio objetivo estimado:    {reg_latest['predicted_future_price']:.4f}")
            print(f"Retorno precio estimado:     {reg_latest['expected_return'] * 100:+.2f}%")
            print(f"RMSE regresión OOS:          {reg_latest['holdout_rmse']:.4f}")
            print(f"RMSE persistencia:           {reg_latest['persistence_holdout_rmse']:.4f}")
            print(f"Skill RMSE vs persistencia:  {reg_latest['regression_rmse_skill_vs_persistence']:+.4f}")
            print(f"Rango empírico 80%:          [{reg_latest['interval80_low']:.4f}, {reg_latest['interval80_high']:.4f}]")
            print("Estas referencias son resultados experimentales, no órdenes de compra/venta.")

        # ------------------------------------------------------------------
        # FASE FINAL: ablaciones y baselines adicionales
        # ------------------------------------------------------------------
        final_experiment_result = self.final_experiments.run(
            ticker=ticker,
            horizon=horizon,
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            full_feature_cols=feature_cols,
            horizon_dir=horizon_dir,
        )

        # Auditoría específica de targets solapados.
        overlap_validation = self.overlap_validator.evaluate(
            y_true=y_test,
            y_prob=ensemble_holdout,
            horizon=horizon,
            output_dir=horizon_dir,
            label="ENSEMBLE",
        )

        # Comparación con/sin sentimiento.
        sentiment_comparison = self.sentiment_ablation.run(
            ticker=ticker,
            horizon=horizon,
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            full_feature_cols=feature_cols,
            horizon_dir=horizon_dir,
        )

        # Explicabilidad de los modelos finales.
        explainability_result = self.explainability.run(
            ticker=ticker,
            horizon=horizon,
            models=models,
            X_train=X_train,
            X_test=X_test,
            horizon_dir=horizon_dir,
        )

        cv_results.to_csv(os.path.join(horizon_dir, "cross_validation.csv"), index=False)
        holdout_results.to_csv(os.path.join(horizon_dir, "holdout_metrics.csv"), index=False)
        backtest_df.to_csv(os.path.join(horizon_dir, "backtest.csv"), index=False)
        pd.DataFrame([ensemble_metrics]).to_csv(os.path.join(horizon_dir, "ensemble_metrics.csv"), index=False)
        pd.DataFrame([baseline_metrics]).to_csv(os.path.join(horizon_dir, "baseline_metrics.csv"), index=False)
        pd.DataFrame([financial_metrics]).to_csv(os.path.join(horizon_dir, "financial_metrics.csv"), index=False)

        latest_summary = pd.DataFrame([{
            "ticker": ticker,
            "date": latest_date.strftime("%Y-%m-%d"),
            "horizon": horizon,
            "probability_up": ensemble_latest,
            "probability_down": 1 - ensemble_latest,
            "signal": signal,
            "confidence": confidence,
            "sentiment_context": current_sentiment.get("sentiment_mean", 0.0),
            "sentiment_in_model": sentiment_in_model,
            "predicted_future_price": (
                regression_result.get("latest", {}).get("predicted_future_price", np.nan)
                if regression_result else np.nan
            ),
            "regression_model_selected": (
                regression_result.get("latest", {}).get("selected_regression_model", "")
                if regression_result else ""
            ),
            "regression_rmse": (
                regression_result.get("latest", {}).get("holdout_rmse", np.nan)
                if regression_result else np.nan
            ),
            "regression_persistence_rmse": (
                regression_result.get("latest", {}).get("persistence_holdout_rmse", np.nan)
                if regression_result else np.nan
            ),
            "regression_rmse_skill": (
                regression_result.get("latest", {}).get("regression_rmse_skill_vs_persistence", np.nan)
                if regression_result else np.nan
            ),
            "regression_expected_return": (
                regression_result.get("latest", {}).get("expected_return", np.nan)
                if regression_result else np.nan
            ),
            **{f"prob_{k}": v for k, v in latest_model_probabilities.items()},
            **{f"weight_{k}": v for k, v in used_weights.items()},
        }])
        latest_summary.to_csv(os.path.join(horizon_dir, "latest_prediction.csv"), index=False)

        for name, model in models.items():
            joblib.dump(model, os.path.join(horizon_dir, f"model_{name}.joblib"))

        if lstm_model is not None:
            try:
                lstm_model.save(os.path.join(horizon_dir, "model_LSTM.keras"))
                joblib.dump(lstm_scaler, os.path.join(horizon_dir, "lstm_scaler.joblib"))
            except Exception as exc:
                print(f"No se pudo guardar LSTM: {exc}")

        Visualizer.plot_equity(
            backtest_df,
            f"{ticker} - Ensemble vs Buy & Hold - Horizonte {horizon}d",
            os.path.join(horizon_dir, "equity_curve.png"),
        )
        Visualizer.plot_probability(
            backtest_df,
            f"{ticker} - Probabilidad de subida - Horizonte {horizon}d",
            os.path.join(horizon_dir, "probability_curve.png"),
            self.config,
        )

        self.journal.append(
            ticker=ticker,
            horizon=horizon,
            prediction_date=latest_date,
            probability_up=ensemble_latest,
            signal=signal,
            confidence=confidence,
            sentiment_context=current_sentiment.get("sentiment_mean", 0.0),
            model_probabilities=latest_model_probabilities,
        )

        return {
            "cv_results": cv_results,
            "holdout_results": holdout_results,
            "ensemble_metrics": ensemble_metrics,
            "baseline_metrics": baseline_metrics,
            "financial_metrics": financial_metrics,
            "latest_probability": ensemble_latest,
            "latest_signal": signal,
            "latest_confidence": confidence,
            "model_probabilities": latest_model_probabilities,
            "weights": used_weights,
            "sentiment_in_model": sentiment_in_model,
            "sentiment_ablation_executed": sentiment_comparison is not None,
            "shap_executed": bool(
                explainability_result.get("shap_executed", False)
            ),
            "overlap_validation": overlap_validation,
            "price_regression": regression_result,
            "classification_regression_decision": combined_decision,
            "final_experiments": final_experiment_result,
        }

    def run_all(self) -> Dict[str, Dict[int, Dict[str, Any]]]:
        outputs = {}
        for ticker in self.config.tickers:
            outputs[ticker.upper()] = self.run_ticker(ticker)
        self._print_final_summary(outputs)
        return outputs

    def _print_final_summary(self, outputs):
        print("\n" + "#" * 118)
        print("RESUMEN FINAL DE PREDICCIONES")
        print("#" * 118)
        rows = []

        for ticker, horizon_results in outputs.items():
            for horizon, result in horizon_results.items():
                rows.append({
                    "ETF": ticker,
                    "Horizonte": horizon,
                    "P_Subida": result["latest_probability"],
                    "P_Bajada": 1 - result["latest_probability"],
                    "Señal": result["latest_signal"],
                    "Confianza": result["latest_confidence"],
                    "AUC_ensemble": result["ensemble_metrics"]["roc_auc"],
                    "BalAcc_ensemble": result["ensemble_metrics"]["balanced_accuracy"],
                    "F1_ensemble": result["ensemble_metrics"]["f1"],
                    "AUC_baseline": result["baseline_metrics"]["roc_auc"],
                    "Sharpe": result["financial_metrics"]["strategy_sharpe"],
                    "Return_strategy": result["financial_metrics"]["strategy_total_return"],
                    "Return_BH": result["financial_metrics"]["buy_hold_total_return"],
                    "Exposure": result["financial_metrics"]["market_exposure"],
                    "NLP_model": result["sentiment_in_model"],
                    "AUC_nonoverlap_mean": result.get(
                        "overlap_validation", {}
                    ).get("nonoverlap_roc_auc_mean", np.nan),
                    "AUC_nonoverlap_std": result.get(
                        "overlap_validation", {}
                    ).get("nonoverlap_roc_auc_std", np.nan),
                    "AUC_bootstrap_low": result.get(
                        "overlap_validation", {}
                    ).get("bootstrap_auc_low", np.nan),
                    "AUC_bootstrap_high": result.get(
                        "overlap_validation", {}
                    ).get("bootstrap_auc_high", np.nan),
                    "Effective_N": result.get(
                        "overlap_validation", {}
                    ).get("approx_effective_n", np.nan),
                    "Predicted_future_price": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("predicted_future_price", np.nan),
                    "Regression_model_selected": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("selected_regression_model", ""),
                    "Regression_RMSE": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("holdout_rmse", np.nan),
                    "Regression_persistence_RMSE": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("persistence_holdout_rmse", np.nan),
                    "Regression_RMSE_skill": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("regression_rmse_skill_vs_persistence", np.nan),
                    "Regression_MAE": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("holdout_mae", np.nan),
                    "Regression_MAPE_pct": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("holdout_mape_pct", np.nan),
                    "Regression_R2": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("holdout_r2", np.nan),
                    "Regression_expected_return": result.get(
                        "price_regression", {}
                    ).get("latest", {}).get("expected_return", np.nan),
                })

        summary = pd.DataFrame(rows)
        if not summary.empty:
            display = summary.copy()
            display["P_Subida"] = display["P_Subida"].map(lambda x: f"{x * 100:.2f}%")
            display["P_Bajada"] = display["P_Bajada"].map(lambda x: f"{x * 100:.2f}%")
            print(display.to_string(index=False))
            path = os.path.join(self.config.output_dir, "resumen_predicciones.csv")
            summary.to_csv(path, index=False)
            print(f"\nResumen guardado en: {path}")

            # Ranking multivariable: no selecciona simplemente el mayor AUC.
            ranking = summary.copy()
            for col in ["AUC_ensemble", "BalAcc_ensemble", "Sharpe"]:
                ranking[col] = pd.to_numeric(ranking[col], errors="coerce")

            # Componente de regresión: skill frente a persistencia.
            #   skill = 1 - RMSE_model / RMSE_persistence
            #   > 0  => el regresor mejora la persistencia
            #   = 0  => empata con persistencia
            #   < 0  => es peor que persistencia
            ranking["Regression_RMSE_skill"] = pd.to_numeric(
                ranking["Regression_RMSE_skill"], errors="coerce"
            )
            ranking["regression_skill_component"] = np.clip(
                ranking["Regression_RMSE_skill"].fillna(0.0), -1.0, 1.0
            )

            ranking["final_score"] = (
                0.35 * ranking["AUC_ensemble"].fillna(0.5)
                + 0.25 * ranking["BalAcc_ensemble"].fillna(0.5)
                + 0.20 * np.clip(ranking["Sharpe"].fillna(0.0), -1, 2) / 2.0
                + 0.20 * ranking["regression_skill_component"]
            )
            ranking = ranking.sort_values("final_score", ascending=False)
            ranking_path = os.path.join(self.config.output_dir, "ranking_final_experimentos.csv")
            ranking.to_csv(ranking_path, index=False)
            print(f"Ranking final guardado en: {ranking_path}")

        print(f"Histórico de predicciones: {self.config.prediction_log}")
        print("#" * 118)


# =============================================================================
# 16. EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    config = Config(
        tickers=["SXR8", "SXRV"],

        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        historical_sentiment_dir="data/news_sentiment",
        output_dir="resultados_tfm",

        start_date="2015-01-01",
        end_date=None,
        horizons=[1, 5, 10, 15, 20, 30, 45, 60],

        # Déjalo vacío si los CSV son ISO YYYY-MM-DD o el formato es detectable.
        # Si tu CSV fuese DD/MM/YYYY, puedes usar:
        # date_format_overrides={"SXR8": "%d/%m/%Y", "SXRV": "%d/%m/%Y"},
        date_format_overrides={},

        cv_splits=5,
        test_size_ratio=0.20,
        random_seed=42,
        deterministic_tensorflow=True,
        single_thread_models=True,

        initial_capital=10_000.0,
        transaction_cost=0.001,
        buy_threshold=0.60,
        sell_threshold=0.40,
        allow_short=False,

        suspicious_daily_return=0.20,
        hard_daily_return_limit=2.00,
        max_calendar_gap_days=14,

        use_lstm=True,
        require_xgboost=True,
        require_tensorflow=True,
        lstm_lookback=60,
        lstm_epochs=35,
        lstm_batch_size=32,

        use_news=True,
        require_feedparser=True,
        require_transformers=True,
        require_torch=True,
        max_news_items=100,
        news_when_days=30,
        news_cache_dir="data/news_cache",
        min_sentiment_days_for_model=500,
        min_sentiment_coverage_for_model=0.35,

        sentiment_windows=[1, 5, 10, 15, 20, 30, 45, 60],
        sentiment_lag_days=1,

        # Histórico largo de noticias.
        use_alpha_vantage_history=True,
        require_alpha_vantage_history=True,
        alpha_vantage_history_dir="data/news_history_alpha",
        alpha_vantage_history_start="2018-01-01",
        alpha_vantage_initial_chunk_days=180,
        alpha_vantage_min_chunk_days=14,
        alpha_vantage_limit=1000,
        alpha_vantage_truncation_ratio=0.95,

        # Reparto conservador de cuota entre los dos ETFs.
        alpha_vantage_max_total_calls_per_run=20,
        alpha_vantage_max_calls_per_ticker_per_run=10,
        alpha_vantage_pause_seconds=1.5,
        alpha_vantage_min_relevance=0.22,

        # Segunda fuente histórica independiente.
        use_guardian_history=True,
        require_guardian_history=True,
        guardian_history_dir="data/news_history_guardian",
        guardian_history_start="2015-01-01",
        guardian_initial_chunk_days=180,
        guardian_min_chunk_days=14,
        guardian_page_size=50,
        guardian_max_pages_per_window=10,
        guardian_max_total_calls_per_run=20,
        guardian_max_calls_per_ticker_per_run=10,
        guardian_pause_seconds=0.75,
        guardian_min_relevance=0.20,

        # Auditoría de targets solapados.
        validate_overlapping_targets=True,
        overlap_bootstrap_iterations=300,
        overlap_random_seed=42,

        # Ablación NLP.
        compare_sentiment_models=True,
        compare_lstm_sentiment=True,

        # SHAP.
        use_shap=True,
        require_shap=True,
        shap_max_samples=500,
        shap_top_features=25,

        # Fase final de experimentación.
        run_final_experiments=True,
        final_experiment_horizons=[1, 5, 10, 15, 20, 30, 45, 60],
        run_feature_group_ablation=True,
        run_rate_level_robustness=True,
        run_ma50_ma200_baseline=True,

        # Regresión de precio / RMSE.
        use_price_regression=True,
        regression_target="log_return",
        regression_models=["Ridge", "RandomForestRegressor", "XGBRegressor"],
        regression_interval_low=0.10,
        regression_interval_high=0.90,
        regression_min_train_rows=400,
        regression_min_expected_edge=0.003,

        # Congelación del dataset experimental.
        freeze_experimental_dataset=True,
        reuse_frozen_dataset=True,
        frozen_data_dir="data/frozen_experiment",
    )

    with ConsoleLogger(config.output_dir) as console_log:
        Reproducibility.configure_runtime(
            config.random_seed,
            config.deterministic_tensorflow,
        )
        print("\nCONFIGURACIÓN DEL EXPERIMENTO")
        print("-" * 118)
        print(f"ETFs:                         {config.tickers}")
        print(f"Datos RAW:                    {config.raw_data_dir}")
        print(f"Datos procesados:             {config.processed_data_dir}")
        print(f"Sentimiento histórico:        {config.historical_sentiment_dir}")
        print(f"Resultados:                   {config.output_dir}")
        print(f"Fecha inicial:                {config.start_date}")
        print(f"Fecha final:                  {config.end_date}")
        print(f"Horizontes:                   {config.horizons}")
        print(f"TimeSeriesSplit folds:         {config.cv_splits}")
        print(f"Holdout temporal:             {config.test_size_ratio:.0%}")
        print(f"Capital inicial:              {config.initial_capital:,.2f}")
        print(f"Coste por transacción:        {config.transaction_cost:.4%}")
        print(f"Umbral alcista:               {config.buy_threshold:.2f}")
        print(f"Umbral bajista:               {config.sell_threshold:.2f}")
        print(f"Permitir posiciones cortas:   {config.allow_short}")
        print(f"LSTM solicitada:              {config.use_lstm}")
        print(f"XGBoost obligatorio:          {config.require_xgboost}")
        print(f"TensorFlow obligatorio:       {config.require_tensorflow}")
        print(f"Noticias/FinBERT solicitado:  {config.use_news}")
        print(f"Histórico GDELT solicitado:   {config.use_gdelt_history}")
        print(f"Variables externas FRED:      {config.use_external_features}")
        print(f"Lag variables externas:       {config.external_feature_lag_sessions}")
        print(f"feedparser obligatorio:        {config.require_feedparser}")
        print(f"transformers obligatorio:      {config.require_transformers}")
        print(f"PyTorch obligatorio NLP:       {config.require_torch}")
        print(f"Comparar con/sin sentimiento: {config.compare_sentiment_models}")
        print(f"Alpha Vantage histórico:      {config.use_alpha_vantage_history}")
        print(f"Inicio histórico noticias:    {config.alpha_vantage_history_start}")
        print(f"Ventanas NLP:                  {config.sentiment_windows}")
        print(f"Validación targets solapados: {config.validate_overlapping_targets}")
        print(f"Cuota Alpha total/run:        {config.alpha_vantage_max_total_calls_per_run}")
        print(f"Cuota Alpha por ETF/run:      {config.alpha_vantage_max_calls_per_ticker_per_run}")
        print(f"Guardian histórico:           {config.use_guardian_history}")
        print(f"Guardian API key presente:    {bool(config.guardian_api_key)}")
        print(f"Seed reproducible:            {config.random_seed}")
        print(f"Horizontes predicción:         {config.horizons}")
        print(f"LSTM en ablación NLP:          {config.compare_lstm_sentiment}")
        print(f"SHAP solicitado:               {config.use_shap}")
        print(f"Regresión precio / RMSE:       {config.use_price_regression}")
        print(f"Modelos regresión:             {config.regression_models}")
        print(f"Fase final experimental:       {config.run_final_experiments}")
        print(f"Ablación grupos features:      {config.run_feature_group_ablation}")
        print(f"Robustez niveles tipos:        {config.run_rate_level_robustness}")
        print(f"Baseline MA50/MA200:           {config.run_ma50_ma200_baseline}")
        print(f"Congelar dataset:              {config.freeze_experimental_dataset}")
        print(f"Reusar dataset congelado:      {config.reuse_frozen_dataset}")
        print("-" * 118)

        ComponentStatus.print_status(config)
        ComponentStatus.ensure_required(config)

        pipeline = AdvancedETFPipeline(config)
        results = pipeline.run_all()

        print("\n" + "#" * 118)
        print("FIN DEL PROGRAMA")
        print("#" * 118)
        print("Los CSV limpios se encuentran en data/processed/.")
        print("Los informes de preprocesamiento y anomalías se encuentran en resultados_tfm/preprocessing/.")
        print("Las métricas, modelos y gráficas se encuentran dentro de resultados_tfm/<ETF>/horizon_Xd/.")
        print("La regresión RMSE está en horizon_Xd/price_regression/.")
        print("Las ablaciones finales están en horizon_Xd/final_experiments/.")
        print("El ranking consolidado usa skill RMSE vs persistencia y está en resultados_tfm/ranking_final_experimentos.csv.")
        print(f"Log actual: {os.path.abspath(console_log.log_path)}")