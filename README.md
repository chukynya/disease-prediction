# Lifestyle Disease Classification

Klasifikasi `Disease` dari 24 fitur klinis, dengan skrip pelatihan, pelaporan metrik, dan visualisasi korelasi. Fokus penggunaan: `train.py`, `report_metrics.py`, `correlation.py`. (Abaikan `train_and_predict.py`).

Fitur (urutan penting):
Glucose, Cholesterol, Hemoglobin, Platelets, White Blood Cells, Red Blood Cells, Hematocrit, Mean Corpuscular Volume, Mean Corpuscular Hemoglobin, Mean Corpuscular Hemoglobin Concentration, Insulin, BMI, Systolic Blood Pressure, Diastolic Blood Pressure, Triglycerides, HbA1c, LDL Cholesterol, HDL Cholesterol, ALT, AST, Heart Rate, Creatinine, Troponin, C-reactive Protein

## Setup

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Dataset: `data/dataset_10k.csv` dengan kolom 24 fitur + `Disease`.

## Training model (train.py)

```powershell
python .\train.py
```

- Melatih RandomForest dengan GridSearchCV (cv=5)
- Preprocessing: SimpleImputer(median)
- Split stratified 80/20 dan simpan model ke `model/model.pkl` serta indeks test ke `model/test_indices.npy`

## Evaluasi holdout (report_metrics.py)

```powershell
python .\report_metrics.py
```

- Memuat `model/model.pkl`
- Mengevaluasi hanya 20% test set (menggunakan `model/test_indices.npy`)
- Mencetak akurasi, classification report (4 desimal), confusion matrix counts dan versi ternormalisasi

## Korelasi fitur (correlation.py)

```powershell
python .\correlation.py
```

- Membuat heatmap korelasi Pearson antar fitur numerik dan menyimpannya di `data/correlation_heatmap.png`

## Web app (opsional)

```powershell
python .\app.py
```

- Form 24 input klinis, model akan memprediksi disease.
- App menskalakan ke [0,1] berdasarkan rentang klinis bawaan.

## Contoh input (dalam satuan klinis) untuk memicu kelas tertentu

Catatan: ini ilustratif; model Anda mungkin sedikit berbeda. Nilai di luar rentang normal akan di-clip di app.

- Healthy (direkomendasikan; sudah diuji cocok)
```json
{
  "Glucose": 100,
  "Cholesterol": 170,
  "Hemoglobin": 16.0,
  "Platelets": 350000,
  "White Blood Cells": 7500,
  "Red Blood Cells": 4.8,
  "Hematocrit": 47,
  "Mean Corpuscular Volume": 92,
  "Mean Corpuscular Hemoglobin": 31,
  "Mean Corpuscular Hemoglobin Concentration": 34,
  "Insulin": 12,
  "BMI": 22.0,
  "Systolic Blood Pressure": 115,
  "Diastolic Blood Pressure": 75,
  "Triglycerides": 100,
  "HbA1c": 5.2,
  "LDL Cholesterol": 100,
  "HDL Cholesterol": 55,
  "ALT": 20,
  "AST": 20,
  "Heart Rate": 72,
  "Creatinine": 0.9,
  "Troponin": 0.00,
  "C-reactive Protein": 0.5
}
```

- Diabetes (hiperglikemia, HbA1c tinggi, dislipidemia)
```json
{
  "Glucose": 240,
  "Cholesterol": 220,
  "Hemoglobin": 14.5,
  "Platelets": 260000,
  "White Blood Cells": 9000,
  "Red Blood Cells": 4.9,
  "Hematocrit": 46,
  "Mean Corpuscular Volume": 90,
  "Mean Corpuscular Hemoglobin": 30,
  "Mean Corpuscular Hemoglobin Concentration": 34,
  "Insulin": 28,
  "BMI": 31.0,
  "Systolic Blood Pressure": 138,
  "Diastolic Blood Pressure": 88,
  "Triglycerides": 250,
  "HbA1c": 8.5,
  "LDL Cholesterol": 160,
  "HDL Cholesterol": 38,
  "ALT": 45,
  "AST": 42,
  "Heart Rate": 88,
  "Creatinine": 1.1,
  "Troponin": 0.01,
  "C-reactive Protein": 3.5
}
```

- Anemia (Hb/Hct, MCV/MCH/MCHC rendah)
```json
{
  "Glucose": 95,
  "Cholesterol": 170,
  "Hemoglobin": 9.5,
  "Platelets": 300000,
  "White Blood Cells": 7500,
  "Red Blood Cells": 3.5,
  "Hematocrit": 30,
  "Mean Corpuscular Volume": 72,
  "Mean Corpuscular Hemoglobin": 23,
  "Mean Corpuscular Hemoglobin Concentration": 30,
  "Insulin": 10,
  "BMI": 21.0,
  "Systolic Blood Pressure": 110,
  "Diastolic Blood Pressure": 70,
  "Triglycerides": 100,
  "HbA1c": 5.1,
  "LDL Cholesterol": 100,
  "HDL Cholesterol": 55,
  "ALT": 18,
  "AST": 20,
  "Heart Rate": 90,
  "Creatinine": 0.8,
  "Troponin": 0.00,
  "C-reactive Protein": 1.5
}
```

- Thalasse (mikrositik; MCV/MCH/MCHC sangat rendah, RBC relatif tinggi)
```json
{
  "Glucose": 90,
  "Cholesterol": 165,
  "Hemoglobin": 10.5,
  "Platelets": 240000,
  "White Blood Cells": 6800,
  "Red Blood Cells": 5.8,
  "Hematocrit": 33,
  "Mean Corpuscular Volume": 65,
  "Mean Corpuscular Hemoglobin": 20,
  "Mean Corpuscular Hemoglobin Concentration": 28,
  "Insulin": 9,
  "BMI": 20.5,
  "Systolic Blood Pressure": 112,
  "Diastolic Blood Pressure": 72,
  "Triglycerides": 95,
  "HbA1c": 5.0,
  "LDL Cholesterol": 95,
  "HDL Cholesterol": 58,
  "ALT": 16,
  "AST": 18,
  "Heart Rate": 85,
  "Creatinine": 0.9,
  "Troponin": 0.00,
  "C-reactive Protein": 0.9
}
```

- Thromboc (platelet sangat rendah)
```json
{
  "Glucose": 92,
  "Cholesterol": 175,
  "Hemoglobin": 13.2,
  "Platelets": 45000,
  "White Blood Cells": 7000,
  "Red Blood Cells": 4.4,
  "Hematocrit": 37,
  "Mean Corpuscular Volume": 88,
  "Mean Corpuscular Hemoglobin": 29,
  "Mean Corpuscular Hemoglobin Concentration": 33,
  "Insulin": 11,
  "BMI": 22.0,
  "Systolic Blood Pressure": 114,
  "Diastolic Blood Pressure": 73,
  "Triglycerides": 105,
  "HbA1c": 5.2,
  "LDL Cholesterol": 105,
  "HDL Cholesterol": 54,
  "ALT": 20,
  "AST": 19,
  "Heart Rate": 78,
  "Creatinine": 0.9,
  "Troponin": 0.00,
  "C-reactive Protein": 1.0
}
```

Penyesuaian cepat bila prediksi tidak sesuai:
- Naikkan Glucose/HbA1c, Triglycerides, BMI untuk mengarah ke Diabetes; turunkan HDL.
- Turunkan Hemoglobin/Hematocrit/MCV/MCH/MCHC untuk Anemia.
- Turunkan MCV/MCH/MCHC lebih ekstrem dan naikkan RBC untuk Thalasse.
- Turunkan Platelets jauh di bawah 150k untuk Thromboc.
