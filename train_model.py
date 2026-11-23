"""
====================================================================
MODEL PREDIKSI PERSENTASE BODY FAT
Perbandingan 3 Model: Linear Regression, Ridge, Random Forest
+ SelectKBest + Interpretasi SHAP (per model)
Hafiz Akbar Habibie Wijaya - 23.53.028035 - Ilmu Komputer UMPR
====================================================================
Pipeline 9 tahap:
  1. Setup environment & load data
  2. Preprocessing (hapus Density, split, imputasi dan scaling di dalam pipeline)
  3. Feature Selection (SelectKBest, k dipilih otomatis dari CV — basis: Linear Regression)
  4. Pembangunan Model Utama (Linear Regression, baseline vs SelectKBest)
  5. Pembangunan Model Pembanding (Ridge Regression, Random Forest Regressor)
     -> dievaluasi pada fitur terpilih yang SAMA (adil, apple-to-apple)
  6. Evaluasi & Perbandingan (MAE/MSE/RMSE/R2 + CV pada fitur terpilih)
  7. Interpretasi SHAP (global + lokal, untuk SEMUA model)
  8. Intersep dan CI 95% koefisien Linear Regression (satuan asli)
  9. Simpan artefak untuk Streamlit
====================================================================
CATATAN METODOLOGI:
  - k terbaik untuk SelectKBest ditentukan dari rata-rata R² CV 5-lipatan
    dengan Linear Regression sebagai basis (model utama penelitian ini).
    Ridge dan Random Forest kemudian dievaluasi pada set fitur yang SAMA
    agar perbandingan antar model adil (bukan tiap model memilih fitur berbeda).
  - SHAP explainer disesuaikan per tipe model:
      * Linear Regression & Ridge -> shap.LinearExplainer (exact, closed-form)
      * Random Forest             -> shap.TreeExplainer (exact untuk tree-based)
  - Tidak ada penghapusan outlier (IQR maupun kuantil) pada data latih maupun uji.
"""

import warnings
warnings.filterwarnings("ignore")
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json
import uuid

from scipy import stats as scipy_stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, GridSearchCV
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

report = {}

# ====================================================================
# TAHAP 1 — LOAD DATA
# ====================================================================
print("\n[TAHAP 1] Memuat dataset...")
df = pd.read_csv(os.path.join(BASE_DIR, "bodyfat.csv"))
print(f"  Dataset awal: {df.shape[0]} baris, {df.shape[1]} kolom")
report["n_awal"] = int(df.shape[0])

# ====================================================================
# TAHAP 2 — PREPROCESSING
# ====================================================================
print("\n[TAHAP 2] Preprocessing...")

# (a) Hapus kolom Density (deterministik terhadap BodyFat via persamaan Siri)
df = df.drop(columns=["Density"])
print("  - Kolom 'Density' dihapus (hubungan deterministik dgn target)")

# (b) Cek missing value
n_missing = int(df.isnull().sum().sum())
print(f"  - Missing value terdeteksi: {n_missing} (imputasi median disiapkan di pipeline, "
      f"{'aktif' if n_missing > 0 else 'tidak aktif pada dataset ini'})")
report["imputer_aktif"] = n_missing > 0

feature_cols = [c for c in df.columns if c != "BodyFat"]

# Pisahkan data sebelum pemodelan. Tidak ada penghapusan outlier (IQR maupun kuantil).
X = df[feature_cols]
y = df["BodyFat"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

print("  - Tidak ada baris yang dihapus sebagai outlier (tidak ada IQR/kuantil filtering)")
report["metode_outlier"] = "Tidak dihapus (tanpa IQR, tanpa penyaringan kuantil)"
report["n_outlier_dihapus_train"] = 0
report["n_train"] = int(len(X_train))
report["n_test"] = int(len(X_test))
report["n_fitur"] = int(len(feature_cols))
report["n_setelah_clean"] = int(len(X_train) + len(X_test))
report["n_outlier_dihapus"] = 0

print(f"  - Split 80/20 -> train: {len(X_train)}, test: {len(X_test)}, fitur: {len(feature_cols)}")
print("  - StandardScaler dan seleksi fitur di-fit di dalam pipeline")

# Helper metrik
def evaluate(model, Xte, yte):
    pred = model.predict(Xte)
    mse = mean_squared_error(yte, pred)
    return {
        "MAE": round(mean_absolute_error(yte, pred), 4),
        "MSE": round(mse, 4),
        "RMSE": round(np.sqrt(mse), 4),
        "R2": round(r2_score(yte, pred), 4),
    }

def cv_r2(model, Xtr, ytr):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    s = cross_val_score(model, Xtr, ytr, cv=kf, scoring="r2")
    return round(s.mean(), 4), round(s.std(), 4)

# ====================================================================
# TAHAP 3 & 4 — FEATURE SELECTION + MODEL UTAMA (LINEAR REGRESSION)
# ====================================================================
print("\n[TAHAP 3+4] Feature selection (SelectKBest) + model utama (Linear Regression)...")

# Model BASELINE: semua fitur, scaling di pipeline
baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("reg", LinearRegression()),
])
baseline.fit(X_train, y_train)
base_metrics = evaluate(baseline, X_test, y_test)
base_cv_mean, base_cv_std = cv_r2(baseline, X_train, y_train)
print(f"  Baseline (semua {len(feature_cols)} fitur): R2_test={base_metrics['R2']}, "
      f"CV_R2={base_cv_mean}+/-{base_cv_std}")

report["baseline"] = {**base_metrics, "CV_R2_mean": base_cv_mean, "CV_R2_std": base_cv_std}

# Eksperimen k = {3,5,7,10} dengan SelectKBest DI DALAM Pipeline
# k terbaik dipilih berdasarkan rata-rata R² CV 5-lipatan (bukan di-hardcode)
k_values = [3, 5, 7, 10]

k_results = []
cv_means = {}
for k in k_values:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_regression, k=k)),
        ("reg", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    m = evaluate(pipe, X_test, y_test)
    cvm, cvs = cv_r2(pipe, X_train, y_train)
    cv_means[k] = cvm
    k_results.append({"k": k, **m, "CV_R2_mean": cvm, "CV_R2_std": cvs})
    print(f"  k={k:2d}: R2_test={m['R2']}, RMSE={m['RMSE']}, CV_R2={cvm}+/-{cvs}")

# Pilih k dengan rata-rata R² CV tertinggi (tidak di-hardcode)
best_k = max(cv_means, key=cv_means.get)
selected_k_result = next(r for r in k_results if r["k"] == best_k)
print(f"\n  >> K terpilih dari CV = {best_k} (CV_R2={selected_k_result['CV_R2_mean']})")

report["k_results"] = k_results
report["best_k"] = best_k
report["selection_method"] = "max_cv_r2"

# Model UTAMA final: Linear Regression dengan k terbaik
linreg_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_regression, k=best_k)),
    ("reg", LinearRegression()),
])
linreg_model.fit(X_train, y_train)
linreg_metrics = evaluate(linreg_model, X_test, y_test)

# Fitur terpilih + skor F (dipakai bersama oleh SEMUA model demi perbandingan adil)
selector = linreg_model.named_steps["select"]
f_scores_arr = selector.scores_
selected_mask = selector.get_support()
selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
print(f"  Fitur terpilih (k={best_k}): {selected_features}")
report["selected_features"] = selected_features
report["all_features"] = feature_cols

# F-score semua fitur (untuk bar chart)
f_scores = sorted(zip(feature_cols, f_scores_arr), key=lambda x: x[1], reverse=True)
report["f_scores"] = [{"fitur": f, "F_score": round(float(s), 2)} for f, s in f_scores]

# Siapkan data ter-preprocess (scaled + selected fitur) untuk dipakai bersama
# oleh Ridge, Random Forest, dan SHAP -- fitur SAMA persis dengan Linear Regression
pre = Pipeline(linreg_model.steps[:-1])  # imputer + scaler + select (tanpa regressor)
X_train_sel = pre.transform(X_train)
X_test_sel = pre.transform(X_test)

# ====================================================================
# TAHAP 5 — MODEL PEMBANDING: RIDGE REGRESSION & RANDOM FOREST
# ====================================================================
print("\n[TAHAP 5] Model pembanding: Ridge Regression & Random Forest...")

alphas = np.logspace(-3, 3, 25)
rf_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# --- (a) RIDGE REGRESSION -------------------------------------------------
ridge_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_regression, k=best_k)),
    ("reg", Ridge()),
])
ridge_search = GridSearchCV(
    ridge_pipe, {"reg__alpha": alphas},
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="r2", n_jobs=-1,
)
ridge_search.fit(X_train, y_train)
best_alpha = float(ridge_search.best_params_["reg__alpha"])
print(f"  Ridge: alpha terbaik = {best_alpha:.6f} (dari {len(alphas)} kandidat, grid log-scale)")

# Fit ridge pada data latih terpreproses (fitur terpilih yang sama)
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_sel, y_train)
ridge_metrics = evaluate(ridge_model, X_test_sel, y_test)

# CV R2 untuk Ridge: clone best estimator, eval di data latih mentah (pipeline lengkap)
best_ridge_pipeline = ridge_search.best_estimator_
ridge_cv_scores = cross_val_score(
    clone(best_ridge_pipeline), X_train, y_train,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring="r2"
)
ridge_cv_mean = round(ridge_cv_scores.mean(), 4)
ridge_cv_std = round(ridge_cv_scores.std(), 4)
print(f"  Ridge (alpha={best_alpha:.6f}): R2_test={ridge_metrics['R2']}, "
      f"CV_R2={ridge_cv_mean}+/-{ridge_cv_std}")
report["ridge_alpha_terbaik"] = best_alpha

# --- (b) RANDOM FOREST REGRESSOR ------------------------------------------
rf_search = GridSearchCV(
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_regression, k=best_k)),
        ("reg", RandomForestRegressor(random_state=RANDOM_STATE)),
    ]),
    param_grid={f"reg__{key}": value for key, value in rf_param_grid.items()},
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="r2",
    n_jobs=-1,
)
rf_search.fit(X_train, y_train)
rf_pipeline = rf_search.best_estimator_
rf_model = rf_pipeline.named_steps["reg"]
rf_metrics = evaluate(rf_pipeline, X_test, y_test)

# CV R2 untuk RF: clone best estimator
rf_cv_scores = cross_val_score(
    clone(rf_pipeline), X_train, y_train,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring="r2"
)
rf_cv_mean = round(rf_cv_scores.mean(), 4)
rf_cv_std = round(rf_cv_scores.std(), 4)
print(f"  Random Forest (best params={rf_search.best_params_}): "
      f"R2_test={rf_metrics['R2']}, CV_R2={rf_cv_mean}+/-{rf_cv_std}")
rf_best_params = {key.removeprefix("reg__"): value for key, value in rf_search.best_params_.items()}
report["rf_best_params"] = rf_best_params

# CV R2 untuk Linear Regression (clone pipeline linreg)
linreg_cv_scores = cross_val_score(
    clone(linreg_model), X_train, y_train,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring="r2"
)
linreg_cv_mean = round(linreg_cv_scores.mean(), 4)
linreg_cv_std = round(linreg_cv_scores.std(), 4)

# ====================================================================
# TAHAP 6 — RINGKASAN PERBANDINGAN 3 MODEL
# ====================================================================
print(f"\n[TAHAP 6] Ringkasan perbandingan 3 model (fitur terpilih sama, k={best_k})...")

model_comparison = [
    {"model": "Linear Regression", **linreg_metrics,
     "CV_R2_mean": linreg_cv_mean, "CV_R2_std": linreg_cv_std},
    {"model": "Ridge Regression", **ridge_metrics,
     "CV_R2_mean": ridge_cv_mean, "CV_R2_std": ridge_cv_std,
     "alpha": best_alpha},
    {"model": "Random Forest", **rf_metrics,
     "CV_R2_mean": rf_cv_mean, "CV_R2_std": rf_cv_std,
     "params": rf_best_params},
]
report["model_comparison"] = model_comparison
report["final_model"] = model_comparison[0]  # kompatibilitas: model utama tetap Linear Regression

for m in model_comparison:
    print(f"  {m['model']:<18s} R2_test={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  "
          f"CV_R2={m['CV_R2_mean']:.4f}+/-{m['CV_R2_std']:.4f}")

best_overall = max(model_comparison, key=lambda m: m["CV_R2_mean"])
print(f"\n  >> Model dengan CV_R2 tertinggi: {best_overall['model']} "
      f"(CV_R2={best_overall['CV_R2_mean']})")
report["model_terbaik_cv"] = best_overall["model"]
artifact_id = uuid.uuid4().hex

# ====================================================================
# TAHAP 6b — INTERSEP DAN CI 95% KOEFISIEN LINEAR REGRESSION (satuan asli)
# ====================================================================
print("\n[TAHAP 6b] Menghitung intersep dan CI 95% koefisien (OLS, satuan asli)...")

# Ambil data latih fitur terpilih dalam satuan asli (tanpa standardisasi)
X_train_raw_sel = X_train[selected_features].values
y_train_arr = y_train.values.astype(float)

# OLS manual via numpy: b = (X'X)^-1 X'y
n_obs = len(y_train_arr)
n_pred = len(selected_features)  # jumlah prediktor (tidak termasuk intersep)
Xd = np.column_stack([np.ones(n_obs), X_train_raw_sel])
beta = np.linalg.solve(Xd.T @ Xd, Xd.T @ y_train_arr)
resid = y_train_arr - Xd @ beta
dof_ci = n_obs - n_pred - 1
s2 = (resid @ resid) / dof_ci
XtX_inv = np.linalg.inv(Xd.T @ Xd)
se_ci = np.sqrt(s2 * np.diag(XtX_inv))
t_kritis = float(scipy_stats.t.ppf(0.975, dof_ci))
ci_bawah = beta - t_kritis * se_ci
ci_atas = beta + t_kritis * se_ci

# Verifikasi: bandingkan beta dengan koefisien sklearn (harus sama dalam toleransi 1e-8)
sk_coef = linreg_model.named_steps["reg"].coef_
sk_intercept = linreg_model.named_steps["reg"].intercept_
# CATATAN: sklearn memakai fitur yang di-scale, beta OLS di-fit pada satuan asli,
# keduanya harus menghasilkan prediksi yang sama tapi koefisiennya berbeda skala.
# Verifikasi lewat prediksi pada X_test_sel vs OLS prediction:
Xd_test = np.column_stack([np.ones(len(X_test)), X_test[selected_features].values])
pred_ols = Xd_test @ beta
pred_sk = linreg_model.predict(X_test)
max_pred_diff = float(np.max(np.abs(pred_ols - pred_sk)))
print(f"  Maks selisih prediksi OLS vs sklearn: {max_pred_diff:.2e} "
      f"({'OK (< 1e-6)' if max_pred_diff < 1e-6 else 'PERIKSA!'})")

# Koefisien terstandarisasi (dari sklearn yang difit pada data terstandarisasi)
# Intersep terstandarisasi = rata-rata y_train (konvensi skripsi)
mean_y_train = float(np.mean(y_train_arr))
std_coef_values = [mean_y_train] + list(sk_coef)

param_names = ["Intersep"] + selected_features
ci_rows = []
for i, name in enumerate(param_names):
    ci_rows.append({
        "parameter": name,
        "koef": round(float(beta[i]), 4),
        "se": round(float(se_ci[i]), 4),
        "ci_bawah": round(float(ci_bawah[i]), 4),
        "ci_atas": round(float(ci_atas[i]), 4),
        "koef_terstandarisasi": round(float(std_coef_values[i]), 4),
    })
    print(f"  {name:12s}: beta={beta[i]:.4f}  SE={se_ci[i]:.4f}  "
          f"CI=[{ci_bawah[i]:.4f}, {ci_atas[i]:.4f}]  "
          f"std_koef={std_coef_values[i]:.4f}")

report["ci_linreg"] = ci_rows
report["ci_dof"] = int(dof_ci)
report["ci_t_kritis"] = round(t_kritis, 4)

# ====================================================================
# TAHAP 6c — MULTIKOLINEARITAS FITUR TERPILIH
# ====================================================================
print("\n[TAHAP 6c] Cek multikolinearitas fitur terpilih...")

X_train_sel_df = X_train[selected_features]

# Korelasi pasangan fitur terpilih
corr_matrix = X_train_sel_df.corr()
korelasi_pairs = []
for i, f1 in enumerate(selected_features):
    for f2 in selected_features[i + 1:]:
        c = corr_matrix.loc[f1, f2]
        korelasi_pairs.append({"fitur_1": f1, "fitur_2": f2, "korelasi": round(float(c), 4)})
        print(f"  Korelasi {f1}–{f2}: {c:.4f}")

report["korelasi_fitur_terpilih"] = korelasi_pairs

# VIF dengan konstanta menggunakan formula numpy:
# VIF_i = 1/(1 - R2_i) di mana R2_i dari regresi X_i pada [1, X_j, j!=i]
# Ini setara dengan statsmodels variance_inflation_factor pada add_constant(X)
X_sel_raw = X_train_sel_df.fillna(X_train_sel_df.median()).values.astype(float)
n_sel = X_sel_raw.shape[0]
vif_values = []
for i, f in enumerate(selected_features):
    # Tambah konstanta (kolom 1) ke X_rest untuk VIF "dengan konstanta"
    X_others = np.delete(X_sel_raw, i, axis=1)
    X_others_c = np.column_stack([np.ones(n_sel), X_others])
    y_i = X_sel_raw[:, i]
    if X_others_c.shape[1] == 0:
        vif = 1.0
    else:
        # Fit OLS: y_i = intercept + beta * X_others
        beta_vif = np.linalg.lstsq(X_others_c, y_i, rcond=None)[0]
        y_pred_i = X_others_c @ beta_vif
        ss_res = np.sum((y_i - y_pred_i) ** 2)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2_i = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = float("inf") if r2_i >= 1.0 else 1.0 / (1.0 - r2_i)
    vif_values.append({"fitur": f, "VIF": round(vif, 2)})
    print(f"  VIF {f}: {vif:.2f}")

report["vif_fitur_terpilih"] = vif_values

ada_multikolinearitas = bool(korelasi_pairs or any(v["VIF"] > 10 for v in vif_values))
if ada_multikolinearitas:
    print("  !! Terdeteksi indikasi multikolinearitas pada fitur terpilih "
          "-> Ridge Regression relevan sebagai model pembanding")

# ====================================================================
# TAHAP 7 — VISUALISASI EVALUASI
# ====================================================================
print("\n[TAHAP 7] Membuat visualisasi...")

# Bar chart F-score
plt.figure(figsize=(9, 5))
fnames = [f for f, _ in f_scores]
fvals = [s for _, s in f_scores]
colors = ["#2563eb" if f in selected_features else "#cbd5e1" for f in fnames]
plt.barh(fnames[::-1], fvals[::-1], color=colors[::-1])
plt.xlabel("Skor F-statistic (f_regression)")
plt.title(f"Skor SelectKBest per Fitur (biru = terpilih, k={best_k})")
plt.tight_layout()
plt.savefig(f"{OUT}/fscore_barchart.png", dpi=130)
plt.close()

# Bar chart perbandingan R2 & RMSE antar 3 model
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
names = [m["model"] for m in model_comparison]
r2_vals = [m["R2"] for m in model_comparison]
rmse_vals = [m["RMSE"] for m in model_comparison]
bar_colors = ["#2563eb", "#16a34a", "#ea580c"]
axes[0].bar(names, r2_vals, color=bar_colors)
axes[0].set_title("R² (test) per Model")
axes[0].set_ylabel("R²")
for i, v in enumerate(r2_vals):
    axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom")
axes[1].bar(names, rmse_vals, color=bar_colors)
axes[1].set_title("RMSE (test) per Model")
axes[1].set_ylabel("RMSE")
for i, v in enumerate(rmse_vals):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{OUT}/model_comparison_bar.png", dpi=130)
plt.close()

# Scatter prediksi vs aktual — 3 subplot (satu per model)
preds = {
    "Linear Regression": linreg_model.predict(X_test),
    "Ridge Regression": ridge_model.predict(X_test_sel),
    "Random Forest": rf_pipeline.predict(X_test),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, pred) in zip(axes, preds.items()):
    ax.scatter(y_test, pred, alpha=0.6, color="#2563eb")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel("BodyFat Aktual (%)")
    ax.set_ylabel("BodyFat Prediksi (%)")
    r2_this = r2_score(y_test, pred)
    ax.set_title(f"{name}\n(R²={r2_this:.3f})")
plt.tight_layout()
plt.savefig(f"{OUT}/pred_vs_actual.png", dpi=130)
plt.close()

# ====================================================================
# TAHAP 8 — INTERPRETASI SHAP (untuk SEMUA model)
# ====================================================================
print("\n[TAHAP 8] Interpretasi SHAP untuk semua model...")
print("  - Linear Regression & Ridge -> shap.LinearExplainer (exact)")
print("  - Random Forest             -> shap.TreeExplainer (exact utk tree-based)")

shap_results = {}

def shap_for_linear(model, name, tag):
    reg = model.named_steps["reg"] if hasattr(model, "named_steps") else model
    explainer = shap.LinearExplainer(reg, X_train_sel)
    sv = explainer.shap_values(X_test_sel)

    plt.figure()
    shap.summary_plot(sv, X_test_sel, feature_names=selected_features,
                       plot_type="bar", show=False)
    plt.title(f"SHAP Global (Bar) - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_bar_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X_test_sel, feature_names=selected_features, show=False)
    plt.title(f"SHAP Global (Beeswarm) - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_beeswarm_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

    idx_high = int(np.argmax(y_test.values))
    expl_obj = shap.Explanation(
        values=sv[idx_high], base_values=explainer.expected_value,
        data=X_test_sel[idx_high], feature_names=selected_features,
    )
    plt.figure()
    shap.plots.waterfall(expl_obj, show=False)
    plt.title(f"SHAP Lokal (Waterfall) - {name} - sampel BodyFat tinggi "
              f"({y_test.values[idx_high]:.1f}%)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_waterfall_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(sv).mean(axis=0)
    return (
        sorted(
            [{"fitur": f, "mean_abs_shap": round(float(v), 4)}
             for f, v in zip(selected_features, mean_abs)],
            key=lambda x: x["mean_abs_shap"], reverse=True
        ),
        explainer.expected_value
    )

def shap_for_tree(model, name, tag):
    explainer = shap.TreeExplainer(model.named_steps["reg"], feature_perturbation="tree_path_dependent")
    rf_test_sel = model[:-1].transform(X_test)
    sv = explainer.shap_values(rf_test_sel)
    # shap>=0.4x: regressor TreeExplainer sudah 2D array langsung
    if isinstance(sv, list):
        sv = sv[0]

    plt.figure()
    shap.summary_plot(sv, rf_test_sel, feature_names=selected_features,
                       plot_type="bar", show=False)
    plt.title(f"SHAP Global (Bar) - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_bar_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, rf_test_sel, feature_names=selected_features, show=False)
    plt.title(f"SHAP Global (Beeswarm) - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_beeswarm_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

    idx_high = int(np.argmax(y_test.values))
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]
    expl_obj = shap.Explanation(
        values=sv[idx_high], base_values=base_val,
        data=rf_test_sel[idx_high], feature_names=selected_features,
    )
    plt.figure()
    shap.plots.waterfall(expl_obj, show=False)
    plt.title(f"SHAP Lokal (Waterfall) - {name} - sampel BodyFat tinggi "
              f"({y_test.values[idx_high]:.1f}%)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_waterfall_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(sv).mean(axis=0)
    return sorted(
        [{"fitur": f, "mean_abs_shap": round(float(v), 4)}
         for f, v in zip(selected_features, mean_abs)],
        key=lambda x: x["mean_abs_shap"], reverse=True
    )

shap_linreg, shap_base_linreg = shap_for_linear(linreg_model, "Linear Regression", "linreg")
shap_ridge, _ = shap_for_linear(ridge_model, "Ridge Regression", "ridge")
shap_rf = shap_for_tree(rf_pipeline, "Random Forest", "rf")
shap_results["Linear Regression"] = shap_linreg
shap_results["Ridge Regression"] = shap_ridge
shap_results["Random Forest"] = shap_rf
report["shap_importance"] = shap_results
report["shap_base_linreg"] = round(float(shap_base_linreg), 4)

# Koefisien Linear Regression & Ridge (terstandarisasi — dari sklearn)
report["koefisien_linreg"] = [
    {"fitur": f, "koef": round(float(c), 4)}
    for f, c in zip(selected_features, linreg_model.named_steps["reg"].coef_)
]
report["intercept_linreg"] = round(float(linreg_model.named_steps["reg"].intercept_), 4)
report["koefisien_ridge"] = [
    {"fitur": f, "koef": round(float(c), 4)}
    for f, c in zip(selected_features, ridge_model.coef_)
]
report["intercept_ridge"] = round(float(ridge_model.intercept_), 4)
report["feature_importance_rf"] = sorted(
    [{"fitur": f, "importance": round(float(v), 4)}
     for f, v in zip(selected_features, rf_model.feature_importances_)],
    key=lambda x: x["importance"], reverse=True
)
report["artifact_id"] = artifact_id

# ====================================================================
# TAHAP 9 — SIMPAN ARTEFAK
# ====================================================================
print("\n[TAHAP 9] Menyimpan artefak...")
joblib.dump(linreg_model, f"{OUT}/model_linreg.pkl")   # pipeline lengkap (imputer+scaler+select+reg)
joblib.dump(ridge_model, f"{OUT}/model_ridge.pkl")     # fit di ruang X_train_sel (sudah diproses via `pre`)
joblib.dump(rf_pipeline, f"{OUT}/model_rf.pkl")        # pipeline lengkap untuk input mentah
joblib.dump(pre, f"{OUT}/preprocessor.pkl")            # imputer+scaler+select, dipakai bersama Ridge & RF
joblib.dump({"artifact_version": 4, "artifact_id": artifact_id,
             "features": feature_cols, "selected": selected_features,
             "best_k": best_k, "ridge_alpha": best_alpha,
             "rf_params": rf_best_params, "shap_background": X_train_sel},
            f"{OUT}/meta.pkl")
with open(f"{OUT}/report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n===== SELESAI =====")
print(f"Baseline           R2_test = {base_metrics['R2']}  | CV = {base_cv_mean}")
for m in model_comparison:
    print(f"{m['model']:<18s} R2_test = {m['R2']}  | CV = {m['CV_R2_mean']}  (k={best_k})")
print(f"Model CV_R2 tertinggi: {best_overall['model']}")
print(f"Intersep OLS: {beta[0]:.4f}  |  CI: [{ci_bawah[0]:.4f}, {ci_atas[0]:.4f}]")
print(f"Artefak tersimpan di {OUT}/")
