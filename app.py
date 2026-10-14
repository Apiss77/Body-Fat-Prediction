"""Streamlit app for body fat prediction and model evaluation."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
POUND_PER_KG = 2.2046226218

st.set_page_config(page_title="Prediksi Body Fat", page_icon="💪", layout="wide")
st.markdown("""
<style>
.main .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
.stAppViewContainer, section.main { height: 100vh; overflow-y: auto !important; }
section.main .block-container { min-height: 100vh; height: auto; overflow: visible; }
h1, h2, h3, h4 { color: #1e293b; }
.metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 18px 20px; margin-bottom: 12px; }
.metric-title { color: #64748b; font-size: .85rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .05em; }
.metric-value { color: #0f172a; font-size: 2.1rem; font-weight: 700; line-height: 1.2; }
.status-badge { display: inline-block; padding: 4px 12px; margin-top: 8px; border-radius: 6px;
    font-size: .85rem; font-weight: 600; }
.badge-success { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.badge-info { background: #e0f2fe; color: #075985; border: 1px solid #bae6fd; }
.badge-warning { background: #fef9c3; color: #854d0e; border: 1px solid #fef08a; }
.badge-danger { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    required = ["model_linreg.pkl", "model_ridge.pkl", "model_rf.pkl", "preprocessor.pkl", "meta.pkl"]
    missing = [name for name in required if not (OUTPUT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Artefak model belum lengkap. Jalankan `python train_model.py` terlebih dahulu. "
            f"Berkas yang hilang: {', '.join(missing)}"
        )
    linreg = joblib.load(OUTPUT_DIR / "model_linreg.pkl")
    ridge = joblib.load(OUTPUT_DIR / "model_ridge.pkl")
    forest = joblib.load(OUTPUT_DIR / "model_rf.pkl")
    pre = joblib.load(OUTPUT_DIR / "preprocessor.pkl")
    meta = joblib.load(OUTPUT_DIR / "meta.pkl")
    if meta.get("artifact_version") != 3 or "shap_background" not in meta:
        raise RuntimeError("Artefak model lama/tidak cocok. Jalankan ulang `python train_model.py`.")
    if meta.get("best_k") != 7 or len(meta.get("selected", [])) != 7:
        raise RuntimeError("Aplikasi ini memakai K=7. Jalankan ulang `python train_model.py`.")
    return linreg, ridge, forest, pre, meta

linreg_model, ridge_model, rf_model, pre, meta = load_artifacts()
features, selected = meta["features"], meta["selected"]
MODELS = {
    "Linear Regression": {"model": linreg_model, "kind": "linear_pipeline"},
    "Ridge Regression": {"model": ridge_model, "kind": "linear_pre"},
    "Random Forest": {"model": rf_model, "kind": "tree_pipeline"},
}
DEFAULTS = {
    "Age": 44.9, "Weight": 178.9, "Height": 70.1, "Neck": 38.0, "Chest": 100.8,
    "Abdomen": 92.6, "Hip": 99.9, "Thigh": 59.4, "Knee": 38.6, "Ankle": 23.1,
    "Biceps": 32.3, "Forearm": 28.7, "Wrist": 18.2,
}
LABELS = {
    "Age": "Usia", "Weight": "Berat badan", "Height": "Tinggi badan", "Neck": "Lingkar leher",
    "Chest": "Lingkar dada", "Abdomen": "Lingkar perut", "Hip": "Lingkar pinggul",
    "Thigh": "Lingkar paha", "Knee": "Lingkar lutut", "Ankle": "Lingkar pergelangan kaki",
    "Biceps": "Lingkar bisep", "Forearm": "Lingkar lengan bawah", "Wrist": "Lingkar pergelangan tangan",
}
LOW = {"Age": 18, "Weight": 100, "Height": 25, "Neck": 25, "Chest": 70, "Abdomen": 60,
       "Hip": 75, "Thigh": 40, "Knee": 28, "Ankle": 15, "Biceps": 20, "Forearm": 18, "Wrist": 13}
HIGH = {"Age": 90, "Weight": 400, "Height": 85, "Neck": 55, "Chest": 145, "Abdomen": 155,
        "Hip": 155, "Thigh": 95, "Knee": 55, "Ankle": 38, "Biceps": 50, "Forearm": 38, "Wrist": 24}
UNITS = {"Age": "tahun", "Weight": "kg", "Height": "inci"}

@st.cache_data
def load_report():
    path = OUTPUT_DIR / "report.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

@st.cache_data
def load_typical_ranges():
    data = pd.read_csv(BASE_DIR / "bodyfat.csv").drop(columns=["Density"], errors="ignore")
    return {name: (float(data[name].quantile(.05)), float(data[name].quantile(.95))) for name in features}

report = load_report()
if report and report.get("artifact_id") != meta.get("artifact_id"):
    raise RuntimeError("Laporan evaluasi tidak cocok dengan model. Jalankan ulang `python train_model.py`.")
typical_ranges = load_typical_ranges()

with st.sidebar:
    st.markdown("### Informasi Penelitian")
    st.markdown("""
**Peneliti:** Hafiz Akbar Habibie Wijaya  
**NIM:** 23.53.028035  
**Program Studi:** S1 Ilmu Komputer  
**Institusi:** Universitas Muhammadiyah Palangkaraya  
**Tahun:** 2026
""")
    st.divider()
    st.markdown("### Seleksi Fitur")
    st.markdown(f"**Metode:** SelectKBest (`f_regression`)  \n**Jumlah fitur:** K=7")
    for index, name in enumerate(selected, 1):
        st.markdown(f"{index}. {LABELS.get(name, name)}")
    st.divider()
    st.markdown("### Model terbaik")
    st.info(f"**{report.get('model_terbaik_cv', 'Belum tersedia')}**\n\nBerdasarkan R² validasi silang.")

st.title(" Prediksi Persentase Body Fat")
st.caption("Prediksi dengan tiga model regresi, SelectKBest K=7, dan interpretasi SHAP.")
tab1, tab2, tab3 = st.tabs([
    "Prediksi dan SHAP",
    "Perbandingan Evaluasi Model",
    "Seleksi Fitur dan Multikolinearitas",
])

with tab1:
    col_input, col_result = st.columns([1, 1.25], gap="large")
    with col_input:
        with st.container(border=True):
            st.markdown("#### 1. Pengaturan Model")
            model_name = st.selectbox("Pilih model regresi", list(MODELS.keys()))
            info = MODELS[model_name]
            st.markdown("#### 2. Input Data Antropometri")
            weight_unit = st.radio("Satuan berat", ["Kilogram (kg)", "Pound (lbs)"], horizontal=True)
            st.caption(f"Model memakai K=7: {len(selected)} fitur terpilih.")
            vals, display_vals, display_numbers = {}, {}, {}
            for name in selected:
                label = LABELS.get(name, name)
                if name == "Weight" and weight_unit == "Kilogram (kg)":
                    value = st.number_input("Berat badan (kg)", value=DEFAULTS[name] / POUND_PER_KG,
                                            min_value=LOW[name] / POUND_PER_KG,
                                            max_value=HIGH[name] / POUND_PER_KG, step=0.1)
                    vals[name] = value * POUND_PER_KG
                    display_vals[name], display_numbers[name] = f"{value:.1f} kg", value
                else:
                    lo, hi = LOW[name], HIGH[name]
                    suffix = "lbs" if name == "Weight" else UNITS.get(name, "cm")
                    value = st.number_input(f"{label} ({suffix})", value=float(DEFAULTS[name]),
                                            min_value=float(lo), max_value=float(hi), step=0.1)
                    vals[name] = value
                    display_vals[name], display_numbers[name] = f"{value:.1f} {suffix}", value
            predict = st.button("Hitung Prediksi Body Fat", type="primary", use_container_width=True)

    with col_result:
        if predict:
            full_row = {name: vals.get(name, DEFAULTS[name]) for name in features}
            X_input = pd.DataFrame([full_row])[features]
            kind, model = info["kind"], info["model"]
            if kind == "linear_pipeline":
                raw_pred = float(model.predict(X_input)[0])
                x_sel = pre.transform(X_input)
            elif kind == "tree_pipeline":
                raw_pred = float(model.predict(X_input)[0])
                x_sel = model[:-1].transform(X_input)
            else:
                x_sel = pre.transform(X_input)
                raw_pred = float(model.predict(x_sel)[0])
            pred = float(np.clip(raw_pred, 0, 100))
            if pred < 6:
                category, badge_class = "Lemak esensial (< 6%)", "badge-warning"
            elif pred < 14:
                category, badge_class = "Atlet (6–13%)", "badge-success"
            elif pred < 18:
                category, badge_class = "Bugar / fitness (14–17%)", "badge-success"
            elif pred < 25:
                category, badge_class = "Rata-rata normal (18–24%)", "badge-info"
            else:
                category, badge_class = "Tinggi / obesitas (≥ 25%)", "badge-danger"

            with st.container(border=True):
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Hasil Prediksi Persentase Lemak Tubuh ({model_name})</div>
                    <div class="metric-value">{pred:.1f}%</div>
                    <div class="status-badge {badge_class}">Status: {category}</div>
                </div>
                """, unsafe_allow_html=True)
                if pred != raw_pred:
                    st.warning(f"Prediksi mentah {raw_pred:.1f}% dibatasi ke rentang 0–100%.")

            all_predictions = {
                "Linear Regression": float(np.clip(linreg_model.predict(X_input)[0], 0, 100)),
                "Ridge Regression": float(np.clip(ridge_model.predict(pre.transform(X_input))[0], 0, 100)),
                "Random Forest": float(np.clip(rf_model.predict(X_input)[0], 0, 100)),
            }
            metrics_by_model = {item["model"]: item for item in report.get("model_comparison", [])}
            comparison_rows = []
            for name, value in all_predictions.items():
                metrics = metrics_by_model.get(name, {})
                comparison_rows.append({
                    "Model": name,
                    "Prediksi": f"{value:.2f}%",
                    "R² uji": metrics.get("R2", "—"),
                    "R² CV": metrics.get("CV_R2_mean", "—"),
                    "Status": "Dipilih" if name == model_name else "",
                })
            with st.container(border=True):
                st.markdown("##### Perbandingan Prediksi Ketiga Model")
                st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")

            if kind == "tree_pipeline":
                explainer = shap.TreeExplainer(model.named_steps["reg"], feature_perturbation="tree_path_dependent")
                sv = explainer.shap_values(x_sel)
                if isinstance(sv, list): sv = sv[0]
                base_val = explainer.expected_value
            else:
                reg = model.named_steps["reg"] if hasattr(model, "named_steps") else model
                # Gunakan seluruh data train yang tersimpan sebagai background agar
                # baseline SHAP konsisten setiap aplikasi dirender.
                background = meta["shap_background"]
                masker = shap.maskers.Independent(background, max_samples=len(background))
                explainer = shap.LinearExplainer(reg, masker)
                sv = explainer.shap_values(x_sel)
                base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)): base_val = base_val[0]
            shap_labels = [f"{LABELS.get(name, name)} ({display_vals[name]})" for name in selected]
            shap_input_values = np.array([display_numbers[name] for name in selected])
            expl = shap.Explanation(values=sv[0], base_values=base_val, data=shap_input_values,
                                    feature_names=shap_labels)
            with st.container(border=True):
                st.markdown(f"#### Interpretasi SHAP ({model_name})")
                st.caption("Waterfall menunjukkan kontribusi tiap fitur terhadap prediksi model.")
                fig, ax = plt.subplots(figsize=(7, 4.2))
                shap.plots.waterfall(expl, show=False)
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True, use_container_width=False)
                plt.close(fig)
                st.caption(f"Baseline: {float(base_val):.2f}%  ·  Total kontribusi: {float(np.asarray(sv)[0].sum()):+.2f} poin")
                contributions = sorted(zip(selected, np.asarray(sv)[0]), key=lambda item: abs(item[1]), reverse=True)
                table = pd.DataFrame([
                    {"Fitur": LABELS.get(name, name), "Nilai input": display_vals[name],
                     "Dampak (poin %)": round(float(value), 2)} for name, value in contributions
                ])
                st.markdown("##### Rincian kontribusi fitur")
                st.dataframe(table, hide_index=True, width="stretch")
        else:
            with st.container(border=True):
                st.info("Pilih model, isi data antropometri, lalu klik **Hitung Prediksi Body Fat**.")

with tab2:
    st.subheader("Evaluasi dan Komparasi Model Regresi")
    st.caption("Kinerja diukur pada data uji 20% dan validasi silang bersarang pada data latih.")
    comparison = report.get("model_comparison", [])
    if comparison:
        rows = [{"Model": row["model"], "R² (uji)": row["R2"], "MAE": row["MAE"],
                 "RMSE": row["RMSE"], "R² CV": row["CV_R2_mean"],
                 "Std CV": row["CV_R2_std"]} for row in comparison]
        with st.container(border=True):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("Laporan evaluasi belum tersedia. Jalankan `python train_model.py`.")
    chart_left, chart_right = st.columns(2, gap="medium")
    for container, filename, title in [
        (chart_left, "model_comparison_bar.png", "Perbandingan Metrik R² dan RMSE"),
        (chart_right, "pred_vs_actual.png", "Scatter Plot: Aktual dan Prediksi"),
    ]:
        image_path = OUTPUT_DIR / filename
        with container:
            with st.container(border=True):
                st.markdown(f"##### {title}")
                if image_path.is_file():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.info("Grafik belum tersedia.")
    if comparison:
        best = report.get("model_terbaik_cv", "model dengan R² CV tertinggi")
        with st.container(border=True):
            st.markdown("##### Kesimpulan Komparasi")
            st.write(f"Model dengan R² validasi silang tertinggi adalah **{best}**. "
                     "Nilai evaluasi di atas dibaca langsung dari hasil training terakhir.")

with tab3:
    st.subheader("Analisis Seleksi Fitur dan Multikolinearitas")
    selection_col, vif_col = st.columns([1.2, 1], gap="medium")
    with selection_col:
        with st.container(border=True):
            st.markdown("##### Peringkat F-score SelectKBest")
            fscore_path = OUTPUT_DIR / "fscore_barchart.png"
            if fscore_path.is_file():
                st.image(str(fscore_path), use_container_width=True)
            else:
                st.info("Grafik F-score belum tersedia.")
    with vif_col:
        with st.container(border=True):
            st.markdown("##### Uji Multikolinearitas (VIF)")
            st.caption("VIF yang tinggi menunjukkan hubungan linear kuat antar fitur.")
            vif = report.get("vif_fitur_terpilih", [])
            if vif:
                vif_table = pd.DataFrame(vif).rename(columns={"fitur": "Fitur", "VIF": "Skor VIF"})
                st.dataframe(vif_table, hide_index=True, width="stretch")
                high_vif = vif_table.loc[vif_table["Skor VIF"] > 10, "Fitur"].tolist()
                if high_vif:
                    st.info("VIF di atas 10: " + ", ".join(high_vif) + ". Ridge Regression membantu menstabilkan koefisien pada kondisi ini.")
            else:
                st.info("Data VIF belum tersedia.")
    with st.container(border=True):
        st.markdown("##### SHAP Global Beeswarm Plot")
        shap_model = st.radio("Pilih model", list(MODELS.keys()), horizontal=True, key="global_shap_model")
        tag = {"Linear Regression": "linreg", "Ridge Regression": "ridge", "Random Forest": "rf"}[shap_model]
        beeswarm_path = OUTPUT_DIR / f"shap_beeswarm_{tag}.png"
        if beeswarm_path.is_file():
            st.image(str(beeswarm_path), use_container_width=True)
        else:
            st.info("Grafik SHAP global belum tersedia.")

st.divider()
st.caption("Hafiz Akbar Habibie Wijaya · 23.53.028035 · Ilmu Komputer UMPR · 2026")
