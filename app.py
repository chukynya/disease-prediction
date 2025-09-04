import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "Glucose": (70, 140),
    "Cholesterol": (125, 200),
    "Hemoglobin": (13.5, 17.5),
    "Platelets": (150000, 450000),
    "White Blood Cells": (4000, 11000),
    "Red Blood Cells": (4.2, 5.4),
    "Hematocrit": (38, 52),
    "Mean Corpuscular Volume": (80, 100),
    "Mean Corpuscular Hemoglobin": (27, 33),
    "Mean Corpuscular Hemoglobin Concentration": (32, 36),
    "Insulin": (5, 25),
    "BMI": (18.5, 24.9),
    "Systolic Blood Pressure": (90, 120),
    "Diastolic Blood Pressure": (60, 80),
    "Triglycerides": (50, 150),
    "HbA1c": (4, 6),
    "LDL Cholesterol": (70, 130),
    "HDL Cholesterol": (40, 60),
    "ALT": (10, 40),
    "AST": (10, 40),
    "Heart Rate": (60, 100),
    "Creatinine": (0.6, 1.2),
    "Troponin": (0.0, 0.04),
    "C-reactive Protein": (0.0, 3.0),
}

FEATURE_ORDER = [
    "Glucose",
    "Cholesterol",
    "Hemoglobin",
    "Platelets",
    "White Blood Cells",
    "Red Blood Cells",
    "Hematocrit",
    "Mean Corpuscular Volume",
    "Mean Corpuscular Hemoglobin",
    "Mean Corpuscular Hemoglobin Concentration",
    "Insulin",
    "BMI",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Triglycerides",
    "HbA1c",
    "LDL Cholesterol",
    "HDL Cholesterol",
    "ALT",
    "AST",
    "Heart Rate",
    "Creatinine",
    "Troponin",
    "C-reactive Protein",
]


def _scale_to_unit(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "secret-key")

    palette = {
        "Healthy": "#22c55e",
        "Diabetes": "#ef4444",
        "Anemia": "#f59e0b",
        "Thalasse": "#8b5cf6",
        "Thromboc": "#06b6d4",
    }

    # Saran singkat per kelas (informasi umum, bukan nasihat medis)
    SUGGESTIONS = {
        "Healthy": [
            "Pertahankan pola makan seimbang dan aktif 150 menit/minggu.",
            "Pantau berat badan, tekanan darah, dan cek rutin sesuai anjuran.",
        ],
        "Diabetes": [
            "Konsultasi ke dokter untuk evaluasi terapi dan target HbA1c.",
            "Batasi gula sederhana, perbanyak serat, atur porsi makan.",
            "Olahraga aerobik + latihan kekuatan secara teratur.",
            "Pantau gula darah; waspadai gejala hipoglikemia/hiperglikemia.",
        ],
        "Anemia": [
            "Diskusikan pemeriksaan feritin/B12/folat untuk cari penyebab.",
            "Konsumsi makanan kaya zat besi (daging merah, sayuran hijau) + vitamin C.",
            "Hindari suplemen besi tanpa anjuran bila penyebab belum jelas.",
        ],
        "Thalasse": [
            "Konsultasi hematologi; pertimbangkan konseling genetik keluarga.",
            "Hindari suplemen besi tanpa indikasi; evaluasi kebutuhan transfusi.",
        ],
        "Thromboc": [
            "Waspadai memar/pendarahan; hindari aspirin/NSAID tanpa anjuran medis.",
            "Cek ulang hitung trombosit dan faktor pemicu (obat/infeksi).",
        ],
    }
    DEFAULT_SUGGESTION = [
        "Hasil termasuk kategori lain. Konsultasi dengan tenaga kesehatan untuk evaluasi lebih lanjut.",
    ]

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("model/model.pkl tidak ditemukan. Jalankan train.py terlebih dahulu.")
    pipeline = joblib.load(model_path)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            feature_ranges=FEATURE_RANGES,
            prediction=None,
            probabilities=None,
            palette=palette,
            inputs={},
            suggestions=None,
        )

    @app.post("/predict")
    def predict():
        try:
            raw_inputs: Dict[str, float] = {}
            for feat in FEATURE_ORDER:
                val_str = request.form.get(feat, "").strip()
                if val_str == "":
                    raise ValueError(f"Input '{feat}' kosong")
                raw_inputs[feat] = float(val_str)

            scaled_values = []
            for feat in FEATURE_ORDER:
                vmin, vmax = FEATURE_RANGES[feat]
                scaled_values.append(_scale_to_unit(raw_inputs[feat], vmin, vmax))

            features_df = pd.DataFrame([scaled_values], columns=FEATURE_ORDER)
            pred = pipeline.predict(features_df)[0]
            probs = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    proba_arr = pipeline.predict_proba(features_df)[0]
                    clf = None
                    if hasattr(pipeline, "named_steps"):
                        clf = pipeline.named_steps.get("rf")
                    class_labels = list(getattr(clf or pipeline, "classes_", []))
                    probs = []
                    for cls, p in zip(class_labels, proba_arr):
                        probs.append({"label": str(cls), "prob": float(p)})
                    probs.sort(key=lambda x: x["prob"], reverse=True)
                except Exception:
                    probs = None
            # Ambil saran berdasarkan prediksi
            suggest_items = SUGGESTIONS.get(str(pred), DEFAULT_SUGGESTION)

            return render_template(
                "index.html",
                feature_ranges=FEATURE_RANGES,
                prediction=str(pred),
                probabilities=probs,
                palette=palette,
                inputs=raw_inputs,
                suggestions=suggest_items,
            )
        except Exception as e:
            flash(str(e), "error")
            return redirect(url_for("index"))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
