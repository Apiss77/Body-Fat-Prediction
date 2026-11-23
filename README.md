# Prediksi Persentase Body Fat

Model prediksi persentase lemak tubuh menggunakan regresi linear dengan SelectKBest dan interpretasi SHAP pada data antropometri.

**Hafiz Akbar Habibie Wijaya · 23.53.028035 · S1 Ilmu Komputer · Universitas Muhammadiyah Palangkaraya · 2026**

---

## Alur Metodologi

1. **Data**: 252 sampel pria, 15 variabel, target `BodyFat`. Kolom `Density` dihapus (deterministik).
2. **Preprocessing**: tidak ada penghapusan outlier (tanpa IQR, tanpa penyaringan kuantil). Imputasi median dan StandardScaler dijalankan di dalam pipeline.
3. **Split data**: 80% latih (201 sampel), 20% uji (51 sampel), `random_state=42`.
4. **Seleksi fitur**: `SelectKBest(f_regression)`, kandidat k ∈ {3, 5, 7, 10}. **k dipilih otomatis dari rata-rata R² CV 5-lipatan** (hasilnya k = 3: Chest, Abdomen, Hip).
5. **Tiga model** dievaluasi pada fitur terpilih yang sama:
   - **Linear Regression** (model utama)
   - **Ridge Regression** (alpha dipilih via GridSearchCV, 25 kandidat log-scale)
   - **Random Forest Regressor** (grid: n\_estimators×max\_depth×min\_samples\_leaf = 18 kombinasi)
6. **Intersep dan CI 95%** koefisien Linear Regression dihitung secara OLS pada data latih (satuan asli, dof = 197, t kritis ≈ 1.9721).
7. **Interpretasi SHAP**: LinearExplainer untuk Linear Regression dan Ridge; TreeExplainer untuk Random Forest.
8. **Aplikasi Streamlit**: tiga input (lingkar dada, perut, pinggul dalam cm), tabel CI 95%, perbandingan model, dan beeswarm SHAP global.

---

## Hasil Utama

| Model             | MAE    | RMSE   | R² uji | R² CV ± Std       |
|-------------------|--------|--------|--------|-------------------|
| Linear Regression | 3.7514 | 4.7775 | 0.5093 | 0.7152 ± 0.0385   |
| Ridge Regression  | 3.7514 | 4.7774 | 0.5094 | 0.7152 ± 0.0385   |
| Random Forest     | 3.3455 | 4.1078 | 0.6373 | 0.6534 ± 0.0703   |

**Fitur terpilih (k = 3):** Chest, Abdomen, Hip  
**Baseline (13 fitur):** R² uji = 0.6135, R² CV = 0.7034 ± 0.0524

### Intersep dan CI 95% (Linear Regression, satuan asli)

| Parameter | β        | SE     | CI bawah | CI atas  | β terstandarisasi |
|-----------|----------|--------|----------|----------|-------------------|
| Intersep  | −19.4779 | 5.9315 | −31.1753 | −7.7805  | 19.4358           |
| Chest     | −0.2654  | 0.0980 | −0.4587  | −0.0722  | −2.2515           |
| Abdomen   |  1.0460  | 0.0871 |  0.8742  |  1.2177  | 11.1327           |
| Hip       | −0.3132  | 0.0976 | −0.5057  | −0.1207  | −2.1384           |

---

## Cara Pakai

### 1. Instal dependensi
```bash
pip install -r requirements.txt
```

### 2. Latih model (hasilkan artefak di `outputs/`)
```bash
python train_model.py
```

### 3. Jalankan aplikasi Streamlit
```bash
streamlit run app.py
```

### 4. Verifikasi angka skripsi
```bash
python verify_thesis_numbers.py
```

### 5. Analisis tambahan (opsional, butuh statsmodels)
```bash
python analisis_tambahan.py bodyfat.csv
```

---

## Struktur Folder

```
├── bodyfat.csv               # Dataset
├── train_model.py            # Skrip pelatihan model
├── app.py                    # Aplikasi Streamlit
├── requirements.txt          # Dependensi
├── verify_thesis_numbers.py  # Skrip verifikasi angka skripsi
├── analisis_tambahan.py      # Analisis tambahan (CI, VIF 13 fitur, CV berulang)
└── outputs/
    ├── model_linreg.pkl
    ├── model_ridge.pkl
    ├── model_rf.pkl
    ├── preprocessor.pkl
    ├── meta.pkl
    ├── report.json
    └── *.png                 # Grafik evaluasi dan SHAP
```

---

## Dependensi Utama

- Python 3.12.10
- scikit-learn 1.9.0
- pandas 3.0.5
- numpy 2.5.2
- shap 0.52.0
- streamlit 1.62.0
- matplotlib 3.11.1
- scipy 1.15.3
- joblib 1.6.0

---

## Lisensi

MIT License — lihat [LICENSE](LICENSE).
