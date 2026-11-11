import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import datetime
import hashlib
import hmac
import html
import math
import json
import time
import io
import wave
import requests
from fpdf import FPDF
from gtts import gTTS
from io import BytesIO
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% -20%, #151d36 0%, #080c19 50%, #03050a 100%) !important;
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    footer { visibility: hidden; }

    .main .block-container {
        padding: 1.8rem 1.2rem 3rem 1.2rem !important;
        max-width: 1300px !important;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: normal !important;
    }

    h1 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #00F0FF 0%, #7000FF 50%, #FF007F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        letter-spacing: 2px;
        text-align: center;
        margin-bottom: 1.5rem !important;
        filter: drop-shadow(0 0 15px rgba(0, 240, 255, 0.35));
    }

    h2, h3 {
        color: #00F0FF !important;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px;
        border-bottom: 1px dashed rgba(0, 240, 255, 0.25);
        padding-bottom: 8px;
        margin-top: 1.2rem !important;
    }

    div[data-testid="stMetric"], .stMetric {
        background: rgba(13, 21, 39, 0.75) !important;
        border: 1px solid rgba(0, 240, 255, 0.25) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(0, 240, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        transition: all 0.35s ease !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px) !important;
        border-color: #00F0FF !important;
        box-shadow: 0 10px 25px rgba(0, 240, 255, 0.3) !important;
    }

    div[data-testid="stMetricValue"] {
        color: #00F0FF !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #04060c 100%) !important;
        border-right: 1px solid rgba(0, 240, 255, 0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00F0FF 0%, #7000FF 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.4rem !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(0, 240, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 240, 255, 0.6) !important;
        background: linear-gradient(135deg, #7000FF 0%, #00F0FF 100%) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(0, 240, 255, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border-radius: 10px 10px 0 0 !important;
        color: #94A3B8 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 8px 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(0, 240, 255, 0.15) !important;
        color: #00F0FF !important;
        border: 1px solid #00F0FF !important;
        border-bottom: none !important;
    }

    .stDataFrame, div[data-testid="stTable"] {
        background: rgba(13, 21, 39, 0.7) !important;
        border-radius: 12px !important;
        overflow-x: auto !important;
        border: 1px solid rgba(0, 240, 255, 0.2) !important;
    }

    .stAlert {
        background: rgba(15, 23, 42, 0.8) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #05070e; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 240, 255, 0.4); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #00F0FF; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# ADVANCED ML & SECURITY IMPORTS (WITH SAFE FALLBACKS)
# =========================================================
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# This single-file build intentionally supports SQLite only.
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")):
    raise RuntimeError("PostgreSQL DATABASE_URL is not supported by this SQLite build. Unset DATABASE_URL or configure a PostgreSQL adapter.")
DB_NAME = os.getenv("SPACEVITAL_DB_PATH", "spacevital.db")


# =========================================================
# 1. DATABASE LAYER
# =========================================================
def get_db_connection():
    """Open the local SQLite database with foreign keys and a busy timeout."""
    conn = sqlite3.connect(DB_NAME, timeout=15)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


def auto_init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS astronauts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            full_name TEXT,
            national_id TEXT UNIQUE,
            role TEXT,
            blood_type TEXT,
            age INTEGER,
            physical_score INT,
            mental_score INT,
            lab_summary TEXT,
            medical_history TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sos_blackbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            astronaut_name TEXT,
            coordinates TEXT,
            hull_integrity TEXT,
            distress_message TEXT,
            audio_note TEXT,
            cv_diagnosis TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nasa_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_name TEXT,
            anomaly_category TEXT,
            symptoms TEXT,
            resolution_protocol TEXT,
            nasa_reference TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vitals_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            astronaut_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            heart_rate REAL,
            spo2 REAL,
            core_temp REAL,
            radiation REAL,
            source TEXT DEFAULT 'simulation'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            astronaut_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            severity TEXT,
            risk_score REAL,
            alert_message TEXT,
            contributing_factors TEXT
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vitals_crew_time ON vitals_history(astronaut_name, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_crew_time ON alert_log(astronaut_name, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sos_crew_time ON sos_blackbox(astronaut_name, timestamp)")

    cursor.execute("SELECT COUNT(*) FROM astronauts")
    if cursor.fetchone()[0] == 0:
        default_pass = hash_password("pass123")
        cursor.execute("""
            INSERT INTO astronauts
            (username, password, full_name, national_id, role, blood_type, age,
             physical_score, mental_score, lab_summary, medical_history)
            VALUES
            ('alex_vance', ?, 'Cmdr. Alex Vance', 'AST-001', 'Mission Commander', 'O+', 38, 95, 92,
             'Hemoglobin: 15.2 g/dL, Bone Density T-Score: +0.2, ECG Normal', 'Nominal baseline. Microgravity adaptation complete.'),
            ('elena_rostova', ?, 'Dr. Elena Rostova', 'AST-002', 'Chief Medical Officer', 'A-', 34, 88, 90,
             'Hemoglobin: 13.8 g/dL, Bone Density T-Score: -0.1, ECG Normal', 'History of mild space motion sickness.'),
            ('taka_nakamoto', ?, 'Eng. Taka Nakamoto', 'AST-003', 'Flight Systems Engineer', 'B+', 41, 91, 86,
             'Hemoglobin: 14.8 g/dL, Bone Density T-Score: +0.1, Mild Arrythmia', 'Elevated fatigue score during EVA assembly.')
        """, (default_pass, default_pass, default_pass))

    cursor.execute("SELECT COUNT(*) FROM nasa_archive")
    if cursor.fetchone()[0] == 0:
        historical_records = [
            ("Apollo 13 (1970)", "CO2 Scrubber Incompatibility", "High cabin CO2, Loss of main oxygen tanks", "Adapt Lunar Module square LiOH canisters to Command Module round slots using duct tape, plastic bags, and flight manual covers.", "INTERNAL_TRAINING_SUMMARY_SOURCE_NOT_VERIFIED"),
            ("Apollo 11 (1969)", "1202 & 1201 Computer Alarm", "Radar memory overflow during lunar descent", "Bypass non-essential real-time radar tasks and maintain thruster trajectory.", "INTERNAL_TRAINING_SUMMARY_SOURCE_NOT_VERIFIED"),
            ("ISS Expedition 36 (2013)", "EVA Suit Water Leak in Helmet", "Water accumulation behind astronaut visor, risk of asphyxiation", "Terminate EVA immediately and follow EVA emergency water-ingress procedure.", "INTERNAL_TRAINING_SUMMARY_SOURCE_NOT_VERIFIED"),
            ("Skylab 1 (1973)", "Meteoroid Shield & Solar Array Jam", "Severe overheating, loss of primary power", "Deploy thermal parasol and execute manual EVA repair procedures.", "INTERNAL_TRAINING_SUMMARY_SOURCE_NOT_VERIFIED"),
            ("Apollo 12 (1969)", "Lightning Strike / Power Loss", "Main bus power drops, telemetry garbled on launch", "Switch SCE to AUX to restore cabin instruments and telemetry.", "INTERNAL_TRAINING_SUMMARY_SOURCE_NOT_VERIFIED")
        ]
        cursor.executemany("""
            INSERT INTO nasa_archive
            (mission_name, anomaly_category, symptoms, resolution_protocol, nasa_reference)
            VALUES (?, ?, ?, ?, ?)
        """, historical_records)

    conn.commit()
    conn.close()


# =========================================================
# 2. ENHANCED SECURITY (BCRYPT + SALT)
# =========================================================
def hash_password(password):
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return "pbkdf2_sha256$310000$%s$%s" % (salt.hex(), digest.hex())


def verify_password(password, stored_value):
    if not stored_value:
        return False, False
    if HAS_BCRYPT and stored_value.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bool(bcrypt.checkpw(password.encode("utf-8"), stored_value.encode("utf-8"))), False
        except Exception:
            return False, False
    if stored_value.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt_hex, digest_hex = stored_value.split("$")
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)).hex()
            return hmac.compare_digest(actual, digest_hex), False
        except Exception:
            return False, False
    legacy_values = {
        hashlib.sha256((password + "SpaceVitalSalt2026").encode("utf-8")).hexdigest(),
        hashlib.sha256(password.encode("utf-8")).hexdigest(),
        password,
    }
    return stored_value in legacy_values, stored_value in legacy_values


# =========================================================
# 3. VITALS DATA LAYER
# =========================================================
def validate_vitals(hr, spo2, temp, radiation):
    values = {"heart_rate": hr, "spo2": spo2, "core_temp": temp, "radiation": radiation}
    if not all(math.isfinite(float(v)) for v in values.values()):
        raise ValueError("All telemetry values must be finite numbers.")
    bounds = {
        "heart_rate": (30, 220), "spo2": (50, 100),
        "core_temp": (30, 45), "radiation": (0, 100)
    }
    for key, value in values.items():
        low, high = bounds[key]
        if not low <= float(value) <= high:
            raise ValueError(f"{key} must be between {low} and {high} for this prototype input validator.")
    return {k: float(v) for k, v in values.items()}


def save_vitals(astronaut_name, hr, spo2, temp, radiation, source="simulation"):
    values = validate_vitals(hr, spo2, temp, radiation)
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO vitals_history
        (astronaut_name, heart_rate, spo2, core_temp, radiation, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (astronaut_name, values["heart_rate"], values["spo2"], values["core_temp"], values["radiation"], source))
    conn.commit()
    conn.close()


def get_vitals_history(astronaut_name, limit=100):
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT timestamp, heart_rate, spo2, core_temp, radiation, source
        FROM vitals_history
        WHERE astronaut_name = ?
        ORDER BY id DESC
        LIMIT ?
    """, conn, params=(astronaut_name, limit))
    conn.close()
    if not df.empty:
        df = df.sort_values("timestamp")
    return df


def import_telemetry_csv(astronaut_name, file_bytes, source_name="uploaded_csv"):
    try:
        frame = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}")
    required = ["heart_rate", "spo2", "core_temp", "radiation"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("CSV contains no telemetry rows.")
    if len(frame) > 5000:
        raise ValueError("Import is limited to 5,000 rows per upload.")
    prepared = []
    errors = []
    for idx, row in frame.iterrows():
        try:
            v = validate_vitals(row["heart_rate"], row["spo2"], row["core_temp"], row["radiation"])
            timestamp = None
            if "timestamp" in frame.columns and pd.notna(row["timestamp"]):
                parsed = pd.to_datetime(row["timestamp"], errors="raise", utc=True)
                timestamp = parsed.isoformat()
            prepared.append((astronaut_name, timestamp, v["heart_rate"], v["spo2"], v["core_temp"], v["radiation"],
                             "csv_import_unverified:" + os.path.basename(str(source_name))[:80]))
        except Exception as exc:
            errors.append(f"Row {idx + 2}: {exc}")
    if not prepared:
        raise ValueError("No valid rows were found. First errors: " + " | ".join(errors[:5]))
    conn = get_db_connection()
    conn.executemany("""
        INSERT INTO vitals_history (astronaut_name, timestamp, heart_rate, spo2, core_temp, radiation, source)
        VALUES (?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?, ?, ?)
    """, prepared)
    conn.commit()
    conn.close()
    return {"imported": len(prepared), "rejected": len(errors), "errors": errors[:10]}


def get_personal_baseline(astronaut_name, minimum_samples=5):
    df = get_vitals_history(astronaut_name, limit=50)
    if len(df) < minimum_samples:
        return None

    baseline = {}
    for col in ["heart_rate", "spo2", "core_temp", "radiation"]:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        baseline[col] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0
        }
    return baseline


def seed_demo_baseline_if_needed(astronaut_name):
    df = get_vitals_history(astronaut_name, limit=100)
    if len(df) >= 5:
        return

    profiles = {
        "Cmdr. Alex Vance": (72, 98.2, 36.8, 0.85),
        "Dr. Elena Rostova": (76, 97.9, 36.7, 0.80),
        "Eng. Taka Nakamoto": (79, 97.6, 36.9, 0.90)
    }
    if astronaut_name not in profiles:
        return
    base = profiles[astronaut_name]
    rng = np.random.default_rng(abs(hash(astronaut_name)) % (2**32))

    conn = get_db_connection()
    for i in range(12):
        ts = datetime.datetime.now() - datetime.timedelta(hours=(12 - i) * 6)
        hr = base[0] + rng.normal(0, 2.0)
        spo2 = base[1] + rng.normal(0, 0.25)
        temp = base[2] + rng.normal(0, 0.08)
        rad = max(0.1, base[3] + rng.normal(0, 0.08))

        conn.execute("""
            INSERT INTO vitals_history
            (astronaut_name, timestamp, heart_rate, spo2, core_temp, radiation, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (astronaut_name, ts.isoformat(), hr, spo2, temp, rad, "demo_seed"))
    conn.commit()
    conn.close()


# =========================================================
# 4. EXPERIMENTAL MACHINE LEARNING ANOMALY ENGINE
# =========================================================
def detect_ml_anomalies(astronaut_name, current_vitals):
    feature_cols = ["heart_rate", "spo2", "core_temp", "radiation"]
    history_df = get_vitals_history(astronaut_name, limit=250)
    if not HAS_SKLEARN:
        return {"ml_active": False, "is_anomaly": False, "ml_risk_score": 0.0,
                "status_text": "scikit-learn is not installed.", "training_rows": 0, "data_mode": "unavailable"}

    clean = history_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 20:
        return {"ml_active": False, "is_anomaly": False, "ml_risk_score": 0.0,
                "status_text": f"Need at least 20 complete historical readings; currently {len(clean)}.",
                "training_rows": int(len(clean)), "data_mode": "insufficient_data"}
    current = validate_vitals(current_vitals["heart_rate"], current_vitals["spo2"],
                              current_vitals["core_temp"], current_vitals["radiation"])
    sample = np.array([[current[c] for c in feature_cols]], dtype=float)
    same_row = np.isclose(clean[feature_cols].values.astype(float), sample, rtol=0, atol=1e-9).all(axis=1)
    training = clean.loc[~same_row]
    if len(training) < 20:
        return {"ml_active": False, "is_anomaly": False, "ml_risk_score": 0.0,
                "status_text": "At least 20 prior non-identical readings are required.",
                "training_rows": int(len(training)), "data_mode": "insufficient_prior_data"}
    X_train = training.values.astype(float)
    model = IsolationForest(n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train)
    sample = np.array([[current[c] for c in feature_cols]], dtype=float)
    prediction = int(model.predict(sample)[0])
    decision_score = float(model.decision_function(sample)[0])
    score = round(float(np.clip((0.15 - decision_score) * 250, 0, 100)), 1)
    sources = set(history_df.get("source", pd.Series(dtype=str)).dropna().astype(str).str.lower())
    synthetic = bool(sources) and sources.issubset({"demo_seed", "simulation"})
    data_mode = "synthetic_demo" if synthetic else ("mixed_or_unverified" if any(x in sources for x in ("simulation", "demo_seed")) else "entered_or_imported_unverified")
    return {"ml_active": True, "is_anomaly": prediction == -1, "ml_risk_score": score,
            "decision_score": round(decision_score, 5), "training_rows": int(len(X_train)),
            "data_mode": data_mode,
            "status_text": "Isolation Forest fitted to stored telemetry; anomaly output is not a diagnosis."}


@st.cache_data(show_spinner=False)
def evaluate_ml_synthetic_benchmark(random_state=42):
    if not HAS_SKLEARN:
        return None
    rng = np.random.default_rng(random_state)
    centers = np.array([75.0, 98.0, 36.8, 0.8])
    scales = np.array([3.0, 0.6, 0.15, 0.12])
    normal = rng.normal(centers, scales, size=(400, 4))
    anomalous = rng.normal(centers, scales * np.array([5, 5, 5, 6]), size=(80, 4))
    anomalous += np.array([18.0, -5.0, 1.4, 1.5])
    X_test = np.vstack([normal[300:], anomalous])
    y_test = np.concatenate([np.zeros(100, dtype=int), np.ones(80, dtype=int)])
    model = IsolationForest(n_estimators=300, contamination=0.05, random_state=random_state)
    model.fit(normal[:300])
    pred = (model.predict(X_test) == -1).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "note": "Synthetic injected-anomaly benchmark only; not clinical validation."}


def calculate_anomaly(current, baseline):
    result = {}
    for key, value in current.items():
        b = baseline[key]
        mean = b["mean"]
        std = b["std"]
        direction = abs(value - mean)
        scale_floor = max(abs(mean) * 0.005, 1e-6)
        effective_std = max(float(std) if math.isfinite(float(std)) else 0.0, scale_floor)
        z = direction / effective_std
        percent_change = (((value - mean) / mean) * 100) if mean else 0.0

        result[key] = {
            "value": value,
            "baseline": mean,
            "std": std,
            "effective_scale": effective_std,
            "z": z,
            "percent_change": percent_change,
            "anomaly": z >= 2.0
        }
    return result


def calculate_risk(anomalies, ml_result=None):
    weights = {"heart_rate": 0.35, "spo2": 0.40, "core_temp": 0.25}
    raw = 0.0
    contributions = {}

    for key, weight in weights.items():
        z = anomalies[key]["z"]
        component = min(z / 3.0, 1.0) * 100 * weight
        contributions[key] = component
        raw += component

    base_score = min(raw, 100)

    if ml_result and ml_result.get("ml_active"):
        score = round((base_score * 0.6) + (ml_result["ml_risk_score"] * 0.4), 1)
    else:
        score = round(base_score, 1)

    if score < 25:
        level = "LOW"
    elif score < 50:
        level = "MODERATE"
    elif score < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return score, level, contributions


def get_risk_message(level):
    messages = {
        "LOW": "Recorded pattern is close to the available personal baseline.",
        "MODERATE": "A deviation from baseline was detected. Verify the sensor and collect a repeat reading.",
        "HIGH": "A substantial telemetry deviation was detected. Review the source and request qualified human review.",
        "CRITICAL": "A high deviation index was detected. Follow the mission's approved emergency procedure and seek qualified human review."
    }
    return messages[level]


# =========================================================
# 5. REAL LIBROSA VOICE STRESS ANALYSIS
# =========================================================
def analyze_voice_stress(audio_bytes):
    if not HAS_LIBROSA:
        try:
            wav = wave.open(io.BytesIO(audio_bytes), "rb")
            frames = wav.readframes(wav.getnframes())
            sample_rate = wav.getframerate()
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            duration = len(samples) / sample_rate
            rms = float(np.sqrt(np.mean(np.square(samples)))) / 32768.0
            zcr = float(np.mean(np.sign(samples[1:]) != np.sign(samples[:-1])))
            wav.close()

            stress_score = round(min(100.0, max(0.0, 35.0 + (zcr - 0.08) * 250 + (rms - 0.08) * 80)), 1)
            return {
                "librosa_active": False,
                "duration": round(duration, 2),
                "rms": round(rms, 4),
                "zcr": round(zcr, 4),
                "pitch": 0.0,
                "spectral_centroid": 0.0,
                "stress_score": stress_score
            }
        except Exception:
            return None

    try:
        y, sr = librosa.load(BytesIO(audio_bytes), sr=None)
        if len(y) == 0:
            return None

        duration = float(librosa.get_duration(y=y, sr=sr))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        except Exception:
            mean_pitch = 0.0

        stress_score = min(100.0, max(0.0, 25.0 + (zcr * 140) + (rms * 120) + (spec_centroid / 1000.0 * 8)))

        return {
            "librosa_active": True,
            "duration": round(duration, 2),
            "rms": round(rms, 4),
            "zcr": round(zcr, 4),
            "pitch": round(mean_pitch, 1),
            "spectral_centroid": round(spec_centroid, 1),
            "stress_score": round(stress_score, 1)
        }
    except Exception:
        return None


# =========================================================
# 6. ALERT & UTILITIES
# =========================================================
def log_alert(astronaut_name, severity, risk_score, message, factors):
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO alert_log (astronaut_name, severity, risk_score, alert_message, contributing_factors)
        VALUES (?, ?, ?, ?, ?)
    """, (astronaut_name, severity, risk_score, message, ", ".join(factors)))
    conn.commit()
    conn.close()


def get_recent_alerts(astronaut_name, limit=10):
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT timestamp, severity, risk_score, alert_message, contributing_factors
        FROM alert_log WHERE astronaut_name = ? ORDER BY id DESC LIMIT ?
    """, conn, params=(astronaut_name, limit))
    conn.close()
    return df


def create_sos_record(astronaut_name, message, coordinates="Not provided", hull_integrity="Not assessed"):
    message = str(message).strip()
    if not message:
        raise ValueError("Emergency message cannot be empty.")
    if len(message) > 1000:
        raise ValueError("Emergency message must be 1,000 characters or fewer.")
    packet = generate_ccsds_packet(astronaut_name, "SOS", message)
    checksum = hashlib.sha256(packet.encode("utf-8")).hexdigest()
    conn = get_db_connection()
    cur = conn.execute("""
        INSERT INTO sos_blackbox (astronaut_name, coordinates, hull_integrity, distress_message, status)
        VALUES (?, ?, ?, ?, ?)
    """, (astronaut_name, str(coordinates)[:120], str(hull_integrity)[:120], message,
          "LOCAL_ONLY_NOT_TRANSMITTED"))
    record_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"record_id": record_id, "packet": packet, "sha256": checksum,
            "status": "LOCAL_ONLY_NOT_TRANSMITTED"}


def render_hospital_ecg(heart_rate=75):
    html_code = f"""
    <!DOCTYPE html><html><head><style>
    body {{ margin: 0; background-color: #040814; overflow: hidden; font-family: 'Courier New', monospace; }}
    .ecg-container {{ position: relative; width: 100%; height: 220px; border: 1px solid rgba(0, 242, 254, 0.4); border-radius: 12px; background: radial-gradient(circle, rgba(6,12,25,0.95) 0%, rgba(3,7,18,1) 100%); box-shadow: inset 0 0 25px rgba(0, 242, 254, 0.15); }}
    .hud-overlay {{ position: absolute; top: 12px; left: 15px; color: #00f2fe; font-size: 13px; font-weight: bold; letter-spacing: 1px; z-index: 10; }}
    .bpm-overlay {{ position: absolute; top: 12px; right: 20px; color: #10b981; font-size: 20px; font-weight: bold; z-index: 10; }}
    .bpm-value {{ font-size: 32px; color: #00f2fe; animation: pulse 1s infinite alternate; }}
    @keyframes pulse {{ from {{ opacity: 0.8; transform: scale(1); }} to {{ opacity: 1; transform: scale(1.05); }} }}
    canvas {{ display: block; }}
    </style></head><body>
    <div class="ecg-container">
        <div class="hud-overlay">💚 LEAD II ECG MONITOR | TELEMETRY VIEW<br><span style="font-size:10px;color:#94a3b8;">FILTER: 0.5Hz - 40Hz | SWEEP SPEED: 25mm/s</span></div>
        <div class="bpm-overlay">HR <span class="bpm-value">{heart_rate}</span><span style="font-size:13px;"> BPM</span></div>
        <canvas id="ecgCanvas"></canvas>
    </div>
    <script>
        const canvas = document.getElementById('ecgCanvas'); const ctx = canvas.getContext('2d');
        function resize() {{ canvas.width = canvas.parentElement.clientWidth; canvas.height = canvas.parentElement.clientHeight; }}
        resize(); window.addEventListener('resize', resize);
        let x = 0, points = [], speed = 2.5, targetBpm = {heart_rate}, step = 0;
        function drawGrid() {{
            ctx.strokeStyle = 'rgba(0, 242, 254, 0.08)'; ctx.lineWidth = 1;
            for (let i = 0; i < canvas.width; i += 20) {{ ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke(); }}
            for (let j = 0; j < canvas.height; j += 20) {{ ctx.beginPath(); ctx.moveTo(0, j); ctx.lineTo(canvas.width, j); ctx.stroke(); }}
        }}
        function getEcgY(t) {{
            const cycle = (t % 80), mid = canvas.height / 2;
            if (cycle > 25 && cycle < 30) return mid - 8;
            if (cycle >= 30 && cycle < 35) return mid;
            if (cycle === 36) return mid + 12;
            if (cycle === 38) return mid - 65;
            if (cycle === 40) return mid + 22;
            if (cycle > 48 && cycle < 58) return mid - 14;
            return mid + (Math.random() - 0.5) * 2;
        }}
        function animate() {{
            step += (targetBpm / 60) * 1.2; x = (x + speed) % canvas.width;
            const y = getEcgY(step); points[Math.floor(x)] = y;
            ctx.fillStyle = '#040814'; ctx.fillRect(0, 0, canvas.width, canvas.height);
            drawGrid(); ctx.beginPath(); ctx.strokeStyle = '#00f2fe'; ctx.lineWidth = 2.5; ctx.shadowBlur = 10; ctx.shadowColor = '#00f2fe';
            let drawing = false;
            for (let i = 0; i < canvas.width; i++) {{
                if (Math.abs(i - x) < 15) continue;
                if (points[i] !== undefined) {{
                    if (!drawing) {{ ctx.beginPath(); ctx.moveTo(i, points[i]); drawing = true; }}
                    else {{ ctx.lineTo(i, points[i]); }}
                }} else if (drawing) {{ ctx.stroke(); drawing = false; }}
            }}
            if (drawing) ctx.stroke();
            ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fillStyle = '#ffffff'; ctx.shadowColor = '#ffffff'; ctx.shadowBlur = 15; ctx.fill();
            requestAnimationFrame(animate);
        }}
        animate();
    </script></body></html>
    """
    components.html(html_code, height=235)


def get_live_iss_data():
    try:
        response = requests.get("https://api.wheretheiss.at/v1/satellites/25544", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return f"{round(data['velocity']):,} km/h", f"{round(data['altitude'], 1)} km"
    except Exception:
        pass
    return "Illustrative fallback", "Illustrative fallback"


def generate_ccsds_packet(astronaut_name, status_code, message):
    header_sync = "1ACFFC1D"
    apid = "0710"
    timestamp_hex = hex(int(datetime.datetime.now().timestamp()))[2:].zfill(8)
    name_hex = astronaut_name.encode("utf-8").hex()[:16]
    msg_hex = message.encode("utf-8").hex()[:32]
    return f"{header_sync}-{apid}-{timestamp_hex}-{name_hex}-{msg_hex}".upper()


auto_init_db()


# =========================================================
# PAGE CONFIGURATION & STYLES
# =========================================================
st.set_page_config(page_title="SpaceVital | Astronaut Health Monitoring", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Public+Sans:wght@300;400;600;700&family=Rajdhani:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
.stApp { background: radial-gradient(circle at 80% 10%, rgba(11, 61, 145, 0.25) 0%, rgba(4, 8, 20, 0.98) 70%), #040814 !important; font-family: 'Public Sans', sans-serif !important; color: #e2e8f0 !important; }
.glass-card { background: rgba(13, 27, 45, 0.75) !important; border: 1px solid rgba(0, 242, 254, 0.22) !important; border-radius: 16px !important; padding: 24px !important; margin-bottom: 24px !important; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important; backdrop-filter: blur(16px) !important; }
.glass-card-alert { background: rgba(35, 12, 18, 0.8) !important; border: 1px solid rgba(252, 61, 33, 0.5) !important; border-top: 4px solid #FC3D21 !important; border-radius: 16px !important; padding: 24px !important; margin-bottom: 24px !important; box-shadow: 0 10px 30px rgba(252, 61, 33, 0.2) !important; }
.hud-title { font-family: 'Orbitron', sans-serif !important; font-size: 1.15rem !important; font-weight: 700 !important; color: #00f2fe !important; letter-spacing: 1.2px; margin-bottom: 14px; }
.stButton > button { background: linear-gradient(135deg, #0B3D91 0%, #00f2fe 100%) !important; color: #ffffff !important; font-family: 'Orbitron', sans-serif !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; padding: 10px 24px !important; letter-spacing: 1px !important; box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important; }
.block-container { max-width: 1450px !important; padding-top: 1.5rem !important; padding-bottom: 4rem !important; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { background: transparent !important; }
.spacevital-brand { display: flex; align-items: center; gap: 16px; margin: 4px 0 14px 0; }
.spacevital-logo { width: 50px; height: 50px; min-width: 50px; border-radius: 50%; border: 1px solid rgba(0, 220, 255, 0.55); display: flex; align-items: center; justify-content: center; background: radial-gradient(circle at 35% 35%, rgba(0,220,255,.35), rgba(0,50,90,.25) 55%, rgba(0,0,0,.4)); box-shadow: 0 0 22px rgba(0,210,255,.16); }
.spacevital-logo span { font-family: 'Orbitron', sans-serif; color: #65e7ff; font-weight: 700; font-size: 13px; }
.spacevital-title { font-family: 'Orbitron', sans-serif; font-size: 30px; font-weight: 700; letter-spacing: .08em; color: #f2fbff; line-height: 1.1; }
.spacevital-title span { color: #45dfff; }
.spacevital-subtitle { color: #7895a7; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; margin-top: 4px; }
.mission-telemetry { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 14px 18px; margin: 0 0 20px 0; background: rgba(5,16,29,.86); border: 1px solid rgba(67,207,255,.18); border-radius: 14px; box-shadow: inset 0 0 25px rgba(0,170,255,.025), 0 10px 40px rgba(0,0,0,.25); backdrop-filter: blur(16px); }
.mission-telemetry-item { padding: 5px 14px; border-right: 1px solid rgba(100,170,200,.12); }
.mission-telemetry-item:last-child { border-right: none; }
.mission-telemetry-label { display: block; font-size: 9px; color: #66879a; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 4px; }
.mission-telemetry-value { color: #dffaff; font-family: 'Orbitron', sans-serif; font-size: 13px; }
.mission-telemetry-ok { color: #55e6b0; }
.spacevital-footer { margin-top: 50px; padding-top: 18px; border-top: 1px solid rgba(70,180,220,.1); text-align: center; color: #526d7d; font-size: 10px; letter-spacing: .1em; }
.spacevital-footer span { color: #3ed9fa; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MISSION CONTROL HEADER
# =========================================================
st.markdown("""
<div class="spacevital-brand">
    <div class="spacevital-logo"><span>SV</span></div>
    <div>
        <div class="spacevital-title"><span>SPACEVITAL</span></div>
        <div class="spacevital-subtitle">Deep-Space Astronaut Health Research Prototype</div>
    </div>
</div>
""", unsafe_allow_html=True)

iss_velocity, iss_altitude = get_live_iss_data()

st.markdown(f"""
<div class="mission-telemetry">
    <div class="mission-telemetry-item"><span class="mission-telemetry-label">Mission Velocity</span><span class="mission-telemetry-value">{iss_velocity}</span></div>
    <div class="mission-telemetry-item"><span class="mission-telemetry-label">Orbital Altitude</span><span class="mission-telemetry-value">{iss_altitude}</span></div>
    <div class="mission-telemetry-item"><span class="mission-telemetry-label">Hull Integrity</span><span class="mission-telemetry-value mission-telemetry-ok">100% NOMINAL</span></div>
    <div class="mission-telemetry-item"><span class="mission-telemetry-label">Cabin Environment</span><span class="mission-telemetry-value mission-telemetry-ok">21.5°C NOMINAL</span></div>
</div>
""", unsafe_allow_html=True)
st.caption("ISS position/velocity values are fetched from WhereTheISS.at when reachable; fallback values are illustrative only. This feed is not astronaut biomedical telemetry.")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("<h3 class='hud-title'>📡 GATEWAY LINK</h3>", unsafe_allow_html=True)
edge_mode = st.sidebar.toggle("Deep Space Offline (Edge AI)", value=True)
if edge_mode:
    st.sidebar.caption("Local prototype mode. ML activates only when enough stored readings are available.")

if "authenticated_user" not in st.session_state:
    st.session_state["authenticated_user"] = None

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("SELECT full_name FROM astronauts ORDER BY id")
all_crew_names = [row[0] for row in cursor.fetchall()]
conn.close()

if not all_crew_names:
    all_crew_names = ["Cmdr. Alex Vance", "Dr. Elena Rostova", "Eng. Taka Nakamoto"]


# =========================================================
# AUTHENTICATION SCREEN
# =========================================================
if not st.session_state["authenticated_user"]:
    st.markdown("""<div style="display:flex;align-items:center;gap:15px;margin-bottom:15px;">
<div style="font-family:'Orbitron';font-size:2.2rem;font-weight:900;color:#00f2fe;">SPACEVITAL</div>
<span style="background:#FC3D21;color:white;padding:4px 10px;border-radius:4px;font-size:0.75rem;font-family:'Orbitron';font-weight:bold;">RESEARCH PROTOTYPE</span>
</div>
<p style="color:#94a3b8;font-size:1.05rem;">Astronaut Health Monitoring, Experimental ML &amp; Local Emergency Black Box</p>""", unsafe_allow_html=True)

    tab_login, tab_signup, tab_emergency = st.tabs(["🔐 Sign in", "📝 Crew Profile Registration", "🚨 Emergency Demo"])

    with tab_login:
        col_c, col_b = st.columns([1.1, 1], gap="large")
        with col_c:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="hud-title">Demo Credentials</div>', unsafe_allow_html=True)
            login_user = st.text_input("Username", key="l_user")
            login_pass = st.text_input("Security Key", type="password", key="l_pass")
            st.warning("Demo account is for local demonstration only. Change/remove demo credentials before any public deployment.")
            st.caption("Demo login: alex_vance / pass123")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="hud-title">Optional Reference Media</div>', unsafe_allow_html=True)
            st.camera_input("Optional reference photo (not used for authentication)", key="face_cam")
            st.file_uploader("Optional hand reference image (not biometric verification)", type=["png", "jpg", "jpeg"], key="hand_scan")
            st.caption("Images are not processed for identity verification and do not affect login.")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("AUTHENTICATE & GRANT ACCESS", use_container_width=True):
            if login_user and login_pass:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT full_name, password FROM astronauts WHERE username = ?", (login_user,))
                result = cursor.fetchone()

                valid_login, needs_upgrade = verify_password(login_pass, result[1]) if result else (False, False)

                if valid_login and needs_upgrade:
                    cursor.execute("UPDATE astronauts SET password = ? WHERE username = ?", (hash_password(login_pass), login_user))
                    conn.commit()

                conn.close()

                if result and valid_login:
                    st.session_state["authenticated_user"] = result[0]
                    st.session_state["authenticated_username"] = login_user
                    st.success(f"Authenticated Session: {result[0]}")
                    st.rerun()
                else:
                    st.error("Invalid Username or Security Key.")
            else:
                st.error("Please fill in authentication credentials.")

    with tab_signup:
        with st.form("signup_complex_form"):
            st.markdown('<div class="hud-title">1. Personal & Mission Parameters</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            new_fullname = c1.text_input("Full Name")
            new_nat_id = c2.text_input("Crew ID Number")

            c3, c4, c5 = st.columns(3)
            new_username = c3.text_input("Username")
            new_password = c4.text_input("Password", type="password")
            new_role = c5.selectbox("Mission Role", ["Mission Commander", "Chief Medical Officer", "Flight Engineer", "Payload Specialist"])

            c6, c7 = st.columns(2)
            new_blood = c6.selectbox("Blood Type", ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"])
            new_age = c7.number_input("Age", 20, 65, 32)

            st.markdown("---")
            st.markdown('<div class="hud-title">2. Baseline Physiological Index</div>', unsafe_allow_html=True)

            cp, cm = st.columns(2)
            phys_score = cp.slider("Physical Health Score (0-100)", 0, 100, 90)
            ment_score = cm.slider("Mental Resilience Score (0-100)", 0, 100, 88)

            lab_data = st.text_area("Lab Results Summary", "Hemoglobin: 14.5 g/dL, Bone Density: Normal, BP: 120/80 mmHg")
            med_hist = st.text_area("Medical History Log", "No chronic conditions reported.")

            submit = st.form_submit_button("REGISTER ASTRONAUT PROFILE (SECURE PASSWORD HASH)")

            if submit:
                if new_username and new_password and new_fullname and new_nat_id:
                    if len(new_password) < 10:
                        st.error("Use a password with at least 10 characters.")
                        st.stop()
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO astronauts (username, password, full_name, national_id, role, blood_type, age, physical_score, mental_score, lab_summary, medical_history)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (new_username, hash_password(new_password), new_fullname, new_nat_id, new_role, new_blood, new_age, phys_score, ment_score, lab_data, med_hist))
                        conn.commit()
                        conn.close()
                        st.success("Registration successful with salted password hashing. Proceed to sign in.")
                    except sqlite3.IntegrityError:
                        st.error("Username or Crew ID already registered.")

    with tab_emergency:
        st.markdown('<div class="glass-card-alert">', unsafe_allow_html=True)
        st.markdown('<div class="hud-title" style="color:#FC3D21;">🚨 Emergency One-Touch Bypass Matrix</div>', unsafe_allow_html=True)

        c_p1, c_p2 = st.columns([1, 1.2], gap="large")
        with c_p1:
            st.markdown("##### Step 1: Manual Demo Trigger")
            if st.button("START EMERGENCY DEMO WORKFLOW", use_container_width=True):
                st.session_state["emergency_demo_started"] = True
                st.warning("Demo workflow opened. No biometric verification or real override occurred.")

        with c_p2:
            st.markdown("##### 🧾 Emergency Payload")
            emergency_text = st.text_area("Describe the emergency", "Crew member requires immediate mission medical review.", key="login_emergency_text")
            if st.button("Save emergency record locally", use_container_width=True):
                try:
                    result = create_sos_record("DEMO SUBJECT", emergency_text, "Not provided", "Not assessed")
                    st.warning("Saved to this device's local SQLite black box only. No signal was transmitted.")
                    st.code(f"""Record ID: {result['record_id']}
Status: {result['status']}
SHA-256: {result['sha256']}
Packet: {result['packet']}""")
                except ValueError as exc:
                    st.error(str(exc))

        st.caption("Demo only: no computer-vision diagnosis, biometric identity check, satellite link, or external SOS transmission is implemented.")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# AUTHENTICATED DASHBOARD
# =========================================================
else:
    authenticated_username = st.session_state.get("authenticated_username")
    conn = get_db_connection()
    row = conn.execute("SELECT full_name FROM astronauts WHERE username = ?", (authenticated_username,)).fetchone()
    conn.close()
    if not row:
        st.session_state["authenticated_user"] = None
        st.session_state.pop("authenticated_username", None)
        st.error("Your crew profile could not be found. Please sign in again.")
        st.stop()
    selected_crew = row[0]
    st.sidebar.markdown(f"**ACTIVE CREW:** {html.escape(str(selected_crew))}")
    try:
        seed_demo_baseline_if_needed(selected_crew)
    except Exception:
        pass

    if st.sidebar.button("🔒 LOGOUT SESSION"):
        st.session_state["authenticated_user"] = None
        st.session_state.pop("authenticated_username", None)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 class='hud-title'>🚀 COMMAND MODULES</h3>", unsafe_allow_html=True)

    page = st.sidebar.radio(
        "NAVIGATE TELEMETRY MATRIX",
        [
            "🛸 01. ASTRONAUT DIGITAL TWIN & PROFILE",
            "📡 02. VITALS MONITOR & EARLY WARNING",
            "🎙️ 03. VOICE FEATURE EXPLORER",
            "💬 04. MONITORING EXPLANATION PANEL",
            "📅 05. PERSONAL HEALTH BASELINE",
            "💡 06. HEALTH FACTORS & RISK ENGINE",
            "💊 07. MEDICATION REFERENCE",
            "📄 08. RESEARCH REPORT PDF",
            "👨‍🚀 09. EVA SUIT SIMULATOR",
            "🧠 10. CREW INTERACTION MATRIX",
            "🚨 11. EMERGENCY LOG & MISSION ARCHIVE"
        ]
    )

    # -----------------------------------------------------
    # MODULE 01
    # -----------------------------------------------------
    if page == "🛸 01. ASTRONAUT DIGITAL TWIN & PROFILE":
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT national_id, role, blood_type, age, physical_score, mental_score, lab_summary, medical_history FROM astronauts WHERE full_name = ?", (selected_crew,))
        u = cursor.fetchone()
        conn.close()

        nat_id = u[0] if u else "AST-001"
        role = u[1] if u else "Mission Commander"
        blood = u[2] if u else "O+"
        age = u[3] if u else 38
        p_score = u[4] if u else 95
        m_score = u[5] if u else 92
        history = u[7] if u else "Nominal baseline."

        vitals_df = get_vitals_history(selected_crew, 1)
        current_risk = 0

        if not vitals_df.empty:
            latest = vitals_df.iloc[-1]
            baseline = get_personal_baseline(selected_crew)
            if baseline:
                current = {"heart_rate": latest["heart_rate"], "spo2": latest["spo2"], "core_temp": latest["core_temp"], "radiation": latest["radiation"]}
                anomalies = calculate_anomaly(current, baseline)
                ml_res = detect_ml_anomalies(selected_crew, current)
                current_risk, risk_level, _ = calculate_risk(anomalies, ml_res)

        st.markdown(f"<h2 style='color:#00f2fe; font-family:Orbitron;'>{html.escape(str(selected_crew))} — Dynamic Digital Twin</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.2], gap="large")

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="hud-title">Astronaut Profile Data</div>', unsafe_allow_html=True)
            st.write(f"**Crew ID:** `{nat_id}` | **Role:** {role}")
            st.write(f"**Age:** {age} Yrs | **Blood Type:** `{blood}`")
            st.markdown("---")
            st.write(f"**Physical Index:** `{p_score}%` | **Mental Index:** `{m_score}%`")
            st.write(f"**Medical History:** {history}")
            if current_risk:
                st.metric("Current Physiological Deviation Index (HR / SpO2 / Temp)", f"{current_risk}/100")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="hud-title">Personalized Health State</div>', unsafe_allow_html=True)
            regions = ["Cerebral Stress", "Cardiovascular", "Musculoskeletal", "Bone Density"]
            stress = [100 - m_score, 12, 100 - p_score, 18]

            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.barh(regions, stress)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Prototype index")
            ax.set_title("Baseline Profile Indicators")
            st.pyplot(fig, use_container_width=True)
            st.caption("Digital Twin status is a monitoring representation.")
            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # MODULE 02: VITALS MONITOR & ISOLATION FOREST ML
    # -----------------------------------------------------
    elif page == "📡 02. VITALS MONITOR & EARLY WARNING":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>📡 Telemetry Input → Personal Baseline → Isolation Forest → Early Warning</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown('<div class="hud-title">1. Telemetry Input Layer</div>', unsafe_allow_html=True)
        source = st.selectbox("Telemetry Source", ["Simulation", "Manual entry (unverified)"])
        st.caption("CSV uploads are separately tagged as unverified imports. For every dataset, document its source, units, collection time, and transformation steps.")
        st.markdown("**Normalized CSV import** — required columns: `heart_rate, spo2, core_temp, radiation`; optional `timestamp` (ISO-8601). This importer does not automatically map arbitrary NASA OSDR files.")
        csv_template = """timestamp,heart_rate,spo2,core_temp,radiation
2026-01-01T12:00:00Z,75,98.0,36.8,0.8 
"""
        st.download_button("Download telemetry CSV template", csv_template, file_name="spacevital_telemetry_template.csv", mime="text/csv")
        telemetry_file = st.file_uploader("Upload normalized telemetry CSV", type=["csv"], key="telemetry_csv_import")
        if telemetry_file is not None and st.button("IMPORT CSV INTO CREW HISTORY"):
            try:
                import_result = import_telemetry_csv(selected_crew, telemetry_file.getvalue(), telemetry_file.name)
                st.success(f"Imported {import_result['imported']} rows. Rejected: {import_result['rejected']}.")
                if import_result["errors"]:
                    st.warning("Some rows were rejected: " + " | ".join(import_result["errors"]))
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        c1, c2, c3, c4 = st.columns(4)
        hr = c1.number_input("Heart Rate (BPM)", 40.0, 180.0, 75.0, step=1.0)
        spo2 = c2.number_input("SpO2 (%)", 80.0, 100.0, 98.2, step=0.1)
        temp = c3.number_input("Core Temp (°C)", 34.0, 42.0, 36.8, step=0.1)
        rad = c4.number_input("Radiation (mSv/day)", 0.0, 10.0, 0.85, step=0.01)

        if st.button("📥 RECORD TELEMETRY READING", use_container_width=True):
            try:
                save_vitals(selected_crew, hr, spo2, temp, rad, source.lower().replace(" ", "_"))
                st.success("Telemetry reading stored. Source provenance is retained.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        st.markdown("---")
        st.markdown('<div class="hud-title">2. Current Personal Baseline (provisional)</div>', unsafe_allow_html=True)
        baseline = get_personal_baseline(selected_crew)

        if baseline is None:
            st.warning("Not enough historical data yet. Record at least 5 readings for a provisional baseline; 20+ clean readings are needed before the experimental ML model can run.")
        else:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Baseline HR", f"{baseline['heart_rate']['mean']:.1f} BPM")
            b2.metric("Baseline SpO2", f"{baseline['spo2']['mean']:.1f}%")
            b3.metric("Baseline Temp", f"{baseline['core_temp']['mean']:.2f} °C")
            b4.metric("Baseline Radiation", f"{baseline['radiation']['mean']:.2f} mSv/day")

            current = {"heart_rate": hr, "spo2": spo2, "core_temp": temp, "radiation": rad}
            anomalies = calculate_anomaly(current, baseline)
            ml_res = detect_ml_anomalies(selected_crew, current)
            risk_score, risk_level, contributions = calculate_risk(anomalies, ml_res)

            st.markdown("---")
            st.markdown('<div class="hud-title">3. ML Anomaly Detection (Isolation Forest — Experimental)</div>', unsafe_allow_html=True)

            ml_col1, ml_col2 = st.columns([1, 2])
            with ml_col1:
                st.metric("Statistical Anomaly Flag?", "⚠️ YES" if ml_res["is_anomaly"] else "✓ NO")
                st.metric("Relative Anomaly Score", f"{ml_res['ml_risk_score']}/100")
            with ml_col2:
                st.info(f"🧠 **scikit-learn Status:** {ml_res['status_text']}")
                st.caption(f"Training rows: {ml_res.get('training_rows', 0)} | Data mode: {ml_res.get('data_mode', 'unknown')}")
                if ml_res["ml_active"]:
                    st.write(f"**Isolation Forest Decision Score:** `{ml_res['decision_score']}`")
                    st.caption("Decision score is model-relative; it is not a probability, medical threshold, or diagnosis.")
                benchmark = evaluate_ml_synthetic_benchmark()
                if benchmark:
                    with st.expander("Reproducible ML QA benchmark (synthetic data only)"):
                        q1, q2, q3 = st.columns(3)
                        q1.metric("Precision", f"{benchmark['precision']:.2f}")
                        q2.metric("Recall", f"{benchmark['recall']:.2f}")
                        q3.metric("F1-score", f"{benchmark['f1']:.2f}")
                        st.write({k: benchmark[k] for k in ("tn", "fp", "fn", "tp")})
                        st.caption(benchmark["note"])

            st.markdown("---")
            st.markdown('<div class="hud-title">4. Explainable Telemetry Deviation & Early Warning</div>', unsafe_allow_html=True)
            st.warning("This is a prototype deviation index, not a clinically validated risk score. It must not be used for diagnosis, treatment, or real mission decisions.")
            st.markdown("**Research/data provenance:** [NASA Open Science Data Repository (OSDR)](https://science.nasa.gov/biological-physical/data/osdr/) provides space-life-science datasets; [NASA Human Research Program evidence reports](https://humanresearchroadmap.nasa.gov/Evidence/) summarize known human-spaceflight health risks. This app does not automatically ingest those datasets yet.")
            r1, r2 = st.columns([1, 2])
            with r1:
                st.metric("Physiological Deviation Index (HR / SpO2 / Temp)", f"{risk_score}/100")
                st.metric("Deviation Band", risk_level)

            with r2:
                if risk_level == "LOW":
                    st.success(f"🟢 {get_risk_message(risk_level)}")
                elif risk_level == "MODERATE":
                    st.warning(f"🟡 {get_risk_message(risk_level)}")
                else:
                    st.error(f"🔴 {get_risk_message(risk_level)}")

            if risk_level in ["HIGH", "CRITICAL"]:
                if st.button("🚨 LOG EARLY WARNING EVENT"):
                    flagged = [name for name, detail in anomalies.items() if detail["anomaly"]]
                    log_alert(selected_crew, risk_level, risk_score, f"Telemetry deviation event ({risk_level})", flagged or ["Isolation Forest anomaly flag"])
                    st.success("Early warning event logged.")

            st.markdown("---")
            st.markdown('<div class="hud-title">5. Telemetry History & Trends</div>', unsafe_allow_html=True)
            history = get_vitals_history(selected_crew, 30)
            if not history.empty:
                st.line_chart(history.set_index("timestamp")[["heart_rate", "spo2", "core_temp", "radiation"]], use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="hud-title">💓 Synthetic ECG-Style Visualization</div>', unsafe_allow_html=True)
        st.caption("This is a generated illustrative waveform driven by the entered heart-rate number; it is not an ECG recording or clinical signal.")
        render_hospital_ecg(heart_rate=hr)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # MODULE 03: LIBROSA REAL VOICE STRESS ANALYSIS
    # -----------------------------------------------------
    elif page == "🎙️ 03. VOICE FEATURE EXPLORER":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>🎙️ Experimental Voice Feature Explorer (Librosa)</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        c_v1, c_v2 = st.columns(2)
        recorded_audio = None

        with c_v1:
            user_audio_rec = st.audio_input("Record Voice Sample", key="mic_module3")
            if user_audio_rec:
                recorded_audio = user_audio_rec.getvalue()
                st.audio(user_audio_rec)

        with c_v2:
            voice_file = st.file_uploader("Upload WAV Audio", type=["wav"], key="file_module3")
            if voice_file:
                recorded_audio = voice_file.getvalue()
                st.audio(voice_file)

        if recorded_audio:
            res = analyze_voice_stress(recorded_audio)
            if res:
                st.markdown('<div class="hud-title">Acoustic Feature Extraction Layer</div>', unsafe_allow_html=True)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Duration", f"{res['duration']} s")
                m2.metric("Fundamental F0 (Pitch)", f"{res['pitch']} Hz")
                m3.metric("Spectral Centroid", f"{res['spectral_centroid']} Hz")
                m4.metric("RMS Energy", f"{res['rms']}")
                m5.metric("Zero Crossing Rate", f"{res['zcr']}")

                st.markdown("---")
                if res.get("librosa_active"):
                    st.success("🔬 **Librosa Audio Engine:** Extracting real acoustic spectral features.")
                else:
                    st.info("ℹ️ **Standard Audio Engine:** Install `librosa` for full spectral pitch processing.")

                score = res["stress_score"]
                st.metric("Experimental acoustic index", f"{score}/100")
                st.caption("This hand-designed index is exploratory only. Voice features cannot reliably diagnose psychological stress or medical danger and are not used by the three core safety modules.")
            else:
                st.error("Could not parse audio file structure.")

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # MODULE 04 TO 11
    # -----------------------------------------------------
    elif page == "💬 04. MONITORING EXPLANATION PANEL":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>💬 Health Monitoring Assistant</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        baseline = get_personal_baseline(selected_crew)
        latest_df = get_vitals_history(selected_crew, 1)
        risk_level = "INITIALIZING"

        if baseline and not latest_df.empty:
            latest = latest_df.iloc[-1]
            current = {"heart_rate": latest["heart_rate"], "spo2": latest["spo2"], "core_temp": latest["core_temp"], "radiation": latest["radiation"]}
            anomalies = calculate_anomaly(current, baseline)
            ml_res = detect_ml_anomalies(selected_crew, current)
            risk_score, risk_level, _ = calculate_risk(anomalies, ml_res)
            st.info(f"Current Status for {selected_crew}: {risk_level} (Deviation Index: {risk_score}/100). ML Status: {'Anomaly' if ml_res['is_anomaly'] else 'Nominal'}.")

        user_text = st.text_input("Ask about current monitoring state:")
        if user_text:
            resp = f"System analysis for {selected_crew}: Vitals compared with personal baseline & Isolation Forest ML. Current state is {risk_level if baseline else 'Initializing'}."
            st.write(f"**Assistant:** {resp}")
            try:
                tts = gTTS(text=resp, lang="en")
                fp = BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                st.audio(fp.read(), format="audio/mp3")
            except Exception:
                pass
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "📅 05. PERSONAL HEALTH BASELINE":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>📅 Personal Longitudinal Baseline</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        baseline = get_personal_baseline(selected_crew)
        history = get_vitals_history(selected_crew, 100)

        if baseline:
            st.info("Baseline is computed from this crew profile's stored readings. Demo accounts may contain clearly tagged synthetic seed readings; newly registered profiles are not auto-filled with invented medical history.")
            st.dataframe(pd.DataFrame({
                "Indicator": ["Heart Rate", "SpO2", "Core Temp", "Radiation"],
                "Personal Mean": [baseline["heart_rate"]["mean"], baseline["spo2"]["mean"], baseline["core_temp"]["mean"], baseline["radiation"]["mean"]],
                "Variation (Std)": [baseline["heart_rate"]["std"], baseline["spo2"]["std"], baseline["core_temp"]["std"], baseline["radiation"]["std"]]
            }), use_container_width=True, hide_index=True)

        if not history.empty:
            st.line_chart(history.set_index("timestamp")[["heart_rate", "spo2", "core_temp", "radiation"]], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "💡 06. HEALTH FACTORS & RISK ENGINE":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>💡 Health Factors & Explainable Risk Engine</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        baseline = get_personal_baseline(selected_crew)
        latest_df = get_vitals_history(selected_crew, 1)

        if baseline and not latest_df.empty:
            latest = latest_df.iloc[-1]
            current = {"heart_rate": latest["heart_rate"], "spo2": latest["spo2"], "core_temp": latest["core_temp"], "radiation": latest["radiation"]}
            anomalies = calculate_anomaly(current, baseline)
            ml_res = detect_ml_anomalies(selected_crew, current)
            risk_score, risk_level, contributions = calculate_risk(anomalies, ml_res)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.barh(list(contributions.keys()), list(contributions.values()))
            ax.set_title("Explainable Feature Attribution")
            st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "💊 07. MEDICATION REFERENCE":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>💊 Medication & Inventory Reference</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        med_df = pd.DataFrame({
            "Category": ["Analgesic", "Anti-nausea", "Emergency", "Monitoring"],
            "Status": ["Reference only", "Reference only", "Medical Kit", "Telemetry-linked"],
            "Authorization": ["CMO Required", "CMO Required", "Mission Protocol", "Monitoring Only"]
        })
        st.dataframe(med_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "📄 08. RESEARCH REPORT PDF":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>📄 Monitoring Report Generator</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if st.button("GENERATE PDF REPORT"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(190, 10, txt="SPACEVITAL HEALTH MONITORING REPORT", ln=True, align="C")
            pdf.set_font("Arial", size=11)
            pdf.ln(5)
            pdf.cell(190, 8, txt=f"Astronaut: {selected_crew}", ln=True)
            pdf.cell(190, 8, txt=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            output = pdf.output(dest="S")
            pdf_bytes = output.encode("latin-1") if isinstance(output, str) else bytes(output)
            st.download_button("⬇️ DOWNLOAD PDF", data=pdf_bytes, file_name=f"{selected_crew}_SpaceVital_Report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "👨‍🚀 09. EVA SUIT SIMULATOR":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>👨‍🚀 EVA Suit Life Support Monitor</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c_s1, c_s2, c_s3, c_s4 = st.columns(4)
        p = c_s1.number_input("Suit Pressure (psi)", 2.0, 6.0, 4.3)
        t = c_s2.number_input("Suit Temp (°C)", 0.0, 40.0, 18.5)
        co2 = c_s3.slider("CO2 Scrubber (%)", 0, 100, 84)
        bat = c_s4.slider("Suit Battery (%)", 0, 100, 92)

        if 3.5 <= p <= 5.0 and 10 <= t <= 30 and co2 >= 70 and bat >= 25:
            st.success("✅ EVA Suit Life Support Systems Nominal.")
        else:
            st.error("⚠️ EVA environmental deviation detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "🧠 10. CREW INTERACTION MATRIX":
        st.markdown("<h2 style='color:#00f2fe; font-family:Orbitron;'>🧠 Crew Mission Context Matrix</h2>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        crew = ["Cmdr. Alex", "Dr. Elena", "Eng. Taka"]
        matrix = np.array([[1.00, 0.94, 0.88], [0.94, 1.00, 0.85], [0.88, 0.85, 1.00]])
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(matrix, annot=True, xticklabels=crew, yticklabels=crew, cmap="mako", ax=ax, cbar=False)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "🚨 11. EMERGENCY LOG & MISSION ARCHIVE":
        st.markdown("<h2 style='color:#FC3D21; font-family:Orbitron;'>🚨 Emergency Black Box & NASA Archives</h2>", unsafe_allow_html=True)
        tab_sos, tab_archive = st.tabs(["🚨 Send SOS Signal", "📚 NASA Anomaly Archive"])

        with tab_sos:
            st.markdown('<div class="glass-card-alert">', unsafe_allow_html=True)
            sos_msg = st.text_area("Emergency Distress Payload:", "Crew member requires immediate mission medical review.", max_chars=1000)
            coordinates = st.text_input("Coordinates / location (optional)", "Not provided")
            hull_state = st.selectbox("Vehicle / suit status", ["Not assessed", "Nominal", "Degraded", "Critical — reported by operator"])
            st.caption("Dispatch in this prototype means local database logging only. It does not transmit over radio, satellite, or NASA systems.")
            if st.button("SAVE SOS TO LOCAL BLACK BOX", use_container_width=True):
                try:
                    result = create_sos_record(selected_crew, sos_msg, coordinates, hull_state)
                    st.error("SOS EVENT LOGGED LOCALLY — NOT TRANSMITTED")
                    st.code(f"""Record ID: {result['record_id']}
Status: {result['status']}
SHA-256: {result['sha256']}
CCSDS-inspired packet: {result['packet']}""", language="text")
                except ValueError as exc:
                    st.error(str(exc))
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_archive:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            conn = get_db_connection()
            df_archive = pd.read_sql_query("SELECT mission_name AS 'Mission', anomaly_category AS 'Anomaly', symptoms AS 'Symptoms', resolution_protocol AS 'Protocol' FROM nasa_archive", conn)
            conn.close()
            st.dataframe(df_archive, use_container_width=True, hide_index=True)
            st.caption("Historical scenarios are educational summaries. Internal reference labels are not verified NASA document identifiers; consult primary NASA sources before operational use.")
            st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CORE SYSTEM VALIDATION & DATA ETHICS
# =========================================================
with st.expander("System validation, limitations & research references"):
    st.markdown("""
    **Core workflow**
    - Health monitoring: validated numeric input → SQLite history → personal baseline → trend visualization.
    - ML: Isolation Forest anomaly flag trained on stored records; minimum 20 complete rows.
    - Emergency: local-only SOS record + packet-like payload + SHA-256 integrity checksum.

    **Data integrity**
    - `demo_seed` rows are synthetic and must never be described as astronaut measurements.
    - Manual/imported rows remain unverified until a source, units, collection time, and quality checks are documented.
    - No NASA biomedical dataset is automatically downloaded or ingested by this build.
    - SQLite records are not encrypted at rest; do not enter real patient/astronaut identifiers or confidential medical records.

    **Safety boundary**
    - Deviation bands are heuristic engineering indicators, not medical diagnoses, calibrated probabilities, or clinical alarm limits.
    - The SOS function stores a local record only; there is no radio, satellite, ground-station, or NASA dispatch.
    - The ISS feed is orbital context only, not a source of crew vital signs.

    **Primary research references**
    - [NASA Open Science Data Repository](https://science.nasa.gov/biological-physical/data/osdr/)
    - [NASA OSDR API and data access FAQ](https://science.nasa.gov/reference/osdr-faq/)
    - [NASA Human Research Program Evidence Reports](https://humanresearchroadmap.nasa.gov/Evidence/)
    - [NASA Human Spaceflight Risks](https://www.nasa.gov/hrp/risks/)

    **Deployment status:** local academic prototype only. Before any real deployment, add encrypted storage, managed secrets, TLS, role-based access controls, audit logs, backup/restore testing, and a formal clinical/safety validation process.
    """)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="spacevital-footer">
    SPACEVITAL // RESEARCH PROTOTYPE — LOCAL TELEMETRY &amp; EARLY WARNING<br>
    <span>NOT A MEDICAL DEVICE • NOT AN OFFICIAL NASA SYSTEM • NO LIVE SOS TRANSMISSION</span>
</div>
""", unsafe_allow_html=True)
