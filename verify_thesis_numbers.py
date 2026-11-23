"""
Skrip verifikasi angka skripsi.

Membaca outputs/report.json dan membandingkan dengan tabel target dari dokumen skripsi (REVISI_4).
Toleransi:
  - 0.0005 untuk Linear Regression, Ridge, VIF, korelasi, F-score, CI, SHAP linear
  - 0.005  untuk Random Forest dan SHAP Random Forest

Keluar dengan kode 1 jika ada item yang di luar toleransi.

Pemakaian:
    python verify_thesis_numbers.py
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "outputs" / "report.json"

# ------------------------------------------------------------------
# Target dari dokumen skripsi REVISI_4
# ------------------------------------------------------------------
TARGETS = {
    # Data split
    "n_train": (201, 0, "exact"),
    "n_test": (51, 0, "exact"),
    "best_k": (3, 0, "exact"),

    # Baseline (13 fitur)
    "baseline_MAE": (3.3293, 0.0005, "linear"),
    "baseline_RMSE": (4.2403, 0.0005, "linear"),
    "baseline_R2": (0.6135, 0.0005, "linear"),
    "baseline_CV_R2_mean": (0.7034, 0.0005, "linear"),
    "baseline_CV_R2_std": (0.0524, 0.0005, "linear"),

    # k=3 (terpilih)
    "k3_MAE": (3.7514, 0.0005, "linear"),
    "k3_RMSE": (4.7775, 0.0005, "linear"),
    "k3_R2": (0.5093, 0.0005, "linear"),
    "k3_CV_R2_mean": (0.7152, 0.0005, "linear"),
    "k3_CV_R2_std": (0.0385, 0.0005, "linear"),

    # k=5
    "k5_MAE": (3.3989, 0.0005, "linear"),
    "k5_RMSE": (4.2540, 0.0005, "linear"),
    "k5_R2": (0.6110, 0.0005, "linear"),
    "k5_CV_R2_mean": (0.7067, 0.0005, "linear"),
    "k5_CV_R2_std": (0.0352, 0.0005, "linear"),

    # k=7
    "k7_MAE": (3.3661, 0.0005, "linear"),
    "k7_RMSE": (4.1047, 0.0005, "linear"),
    "k7_R2": (0.6378, 0.0005, "linear"),
    "k7_CV_R2_mean": (0.7091, 0.0005, "linear"),
    "k7_CV_R2_std": (0.0271, 0.0005, "linear"),

    # k=10
    "k10_MAE": (3.2488, 0.0005, "linear"),
    "k10_RMSE": (4.0295, 0.0005, "linear"),
    "k10_R2": (0.6510, 0.0005, "linear"),
    "k10_CV_R2_mean": (0.7093, 0.0005, "linear"),
    "k10_CV_R2_std": (0.0464, 0.0005, "linear"),

    # Skor F (urutan dari tinggi ke rendah)
    "fscore_Abdomen": (445.81, 0.0005, "linear"),
    "fscore_Chest": (204.79, 0.0005, "linear"),
    "fscore_Hip": (146.21, 0.0005, "linear"),
    "fscore_Weight": (136.63, 0.0005, "linear"),
    "fscore_Age": (18.37, 0.0005, "linear"),
    "fscore_Ankle": (20.65, 0.0005, "linear"),
    "fscore_Wrist": (31.38, 0.0005, "linear"),

    # 3 model pada k=3
    "linreg_MAE": (3.7514, 0.0005, "linear"),
    "linreg_MSE": (22.8242, 0.0005, "linear"),
    "linreg_RMSE": (4.7775, 0.0005, "linear"),
    "linreg_R2": (0.5093, 0.0005, "linear"),
    "linreg_CV_R2_mean": (0.7152, 0.0005, "linear"),
    "linreg_CV_R2_std": (0.0385, 0.0005, "linear"),

    "ridge_MAE": (3.7514, 0.0005, "linear"),
    "ridge_MSE": (22.8239, 0.0005, "linear"),
    "ridge_RMSE": (4.7774, 0.0005, "linear"),
    "ridge_R2": (0.5094, 0.0005, "linear"),
    "ridge_CV_R2_mean": (0.7152, 0.0005, "linear"),
    "ridge_CV_R2_std": (0.0385, 0.0005, "linear"),
    "ridge_alpha": (0.001, 0.0005, "linear"),

    "rf_MAE": (3.3455, 0.005, "rf"),
    "rf_MSE": (16.8736, 0.005, "rf"),
    "rf_RMSE": (4.1078, 0.005, "rf"),
    "rf_R2": (0.6373, 0.005, "rf"),
    "rf_CV_R2_mean": (0.6534, 0.005, "rf"),
    "rf_CV_R2_std": (0.0703, 0.005, "rf"),

    # CI 95% (OLS, satuan asli, dof=197, t=1.9721)
    "ci_dof": (197, 0, "exact"),
    "ci_t_kritis": (1.9721, 0.0005, "linear"),

    "ci_Intersep_koef": (-19.4779, 0.0005, "linear"),
    "ci_Intersep_se": (5.9315, 0.0005, "linear"),
    "ci_Intersep_bawah": (-31.1753, 0.0005, "linear"),
    "ci_Intersep_atas": (-7.7805, 0.0005, "linear"),
    "ci_Intersep_std": (19.4358, 0.0005, "linear"),

    "ci_Chest_koef": (-0.2654, 0.0005, "linear"),
    "ci_Chest_se": (0.0980, 0.0005, "linear"),
    "ci_Chest_bawah": (-0.4587, 0.0005, "linear"),
    "ci_Chest_atas": (-0.0722, 0.0005, "linear"),
    "ci_Chest_std": (-2.2515, 0.0005, "linear"),

    "ci_Abdomen_koef": (1.0460, 0.0005, "linear"),
    "ci_Abdomen_se": (0.0871, 0.0005, "linear"),
    "ci_Abdomen_bawah": (0.8742, 0.0005, "linear"),
    "ci_Abdomen_atas": (1.2177, 0.0005, "linear"),
    "ci_Abdomen_std": (11.1327, 0.0005, "linear"),

    "ci_Hip_koef": (-0.3132, 0.0005, "linear"),
    "ci_Hip_se": (0.0976, 0.0005, "linear"),
    "ci_Hip_bawah": (-0.5057, 0.0005, "linear"),
    "ci_Hip_atas": (-0.1207, 0.0005, "linear"),
    "ci_Hip_std": (-2.1384, 0.0005, "linear"),

    # VIF fitur terpilih (dengan konstanta)
    "vif_Abdomen": (8.09, 0.0005, "linear"),
    "vif_Chest": (6.50, 0.0005, "linear"),
    "vif_Hip": (4.18, 0.0005, "linear"),

    # Korelasi fitur terpilih
    "corr_Chest_Abdomen": (0.9170, 0.0005, "linear"),
    "corr_Abdomen_Hip": (0.8675, 0.0005, "linear"),
    "corr_Chest_Hip": (0.8319, 0.0005, "linear"),

    # SHAP rata-rata nilai absolut (data uji)
    "shap_linreg_Abdomen": (7.6392, 0.0005, "linear"),
    "shap_linreg_Hip": (1.5514, 0.0005, "linear"),
    "shap_linreg_Chest": (1.4773, 0.0005, "linear"),

    "shap_ridge_Abdomen": (7.6388, 0.0005, "linear"),
    "shap_ridge_Hip": (1.5513, 0.0005, "linear"),
    "shap_ridge_Chest": (1.4771, 0.0005, "linear"),

    "shap_rf_Abdomen": (5.3156, 0.005, "rf"),
    "shap_rf_Hip": (0.7593, 0.005, "rf"),
    "shap_rf_Chest": (0.5710, 0.005, "rf"),

    # Nilai dasar SHAP Linear Regression
    "shap_base_linreg": (19.316, 0.0005, "linear"),

    # Importance internal RF
    "rf_importance_Abdomen": (0.8843, 0.005, "rf"),
    "rf_importance_Hip": (0.0588, 0.005, "rf"),
    "rf_importance_Chest": (0.0570, 0.005, "rf"),
}


def extract_values(report):
    """Ekstrak nilai dari report.json untuk dibandingkan dengan target."""
    vals = {}

    # Data split
    vals["n_train"] = report.get("n_train")
    vals["n_test"] = report.get("n_test")
    vals["best_k"] = report.get("best_k")

    # Baseline
    baseline = report.get("baseline", {})
    vals["baseline_MAE"] = baseline.get("MAE")
    vals["baseline_RMSE"] = baseline.get("RMSE")
    vals["baseline_R2"] = baseline.get("R2")
    vals["baseline_CV_R2_mean"] = baseline.get("CV_R2_mean")
    vals["baseline_CV_R2_std"] = baseline.get("CV_R2_std")

    # k results
    k_map = {r["k"]: r for r in report.get("k_results", [])}
    for k, prefix in [(3, "k3"), (5, "k5"), (7, "k7"), (10, "k10")]:
        r = k_map.get(k, {})
        vals[f"{prefix}_MAE"] = r.get("MAE")
        vals[f"{prefix}_RMSE"] = r.get("RMSE")
        vals[f"{prefix}_R2"] = r.get("R2")
        vals[f"{prefix}_CV_R2_mean"] = r.get("CV_R2_mean")
        vals[f"{prefix}_CV_R2_std"] = r.get("CV_R2_std")

    # F-scores
    fscore_map = {r["fitur"]: r["F_score"] for r in report.get("f_scores", [])}
    for fitur in ["Abdomen", "Chest", "Hip", "Weight", "Age", "Ankle", "Wrist"]:
        vals[f"fscore_{fitur}"] = fscore_map.get(fitur)

    # Model comparison
    mc_map = {r["model"]: r for r in report.get("model_comparison", [])}

    lr = mc_map.get("Linear Regression", {})
    vals["linreg_MAE"] = lr.get("MAE")
    vals["linreg_MSE"] = lr.get("MSE")
    vals["linreg_RMSE"] = lr.get("RMSE")
    vals["linreg_R2"] = lr.get("R2")
    vals["linreg_CV_R2_mean"] = lr.get("CV_R2_mean")
    vals["linreg_CV_R2_std"] = lr.get("CV_R2_std")

    rd = mc_map.get("Ridge Regression", {})
    vals["ridge_MAE"] = rd.get("MAE")
    vals["ridge_MSE"] = rd.get("MSE")
    vals["ridge_RMSE"] = rd.get("RMSE")
    vals["ridge_R2"] = rd.get("R2")
    vals["ridge_CV_R2_mean"] = rd.get("CV_R2_mean")
    vals["ridge_CV_R2_std"] = rd.get("CV_R2_std")
    vals["ridge_alpha"] = report.get("ridge_alpha_terbaik")

    rf = mc_map.get("Random Forest", {})
    vals["rf_MAE"] = rf.get("MAE")
    vals["rf_MSE"] = rf.get("MSE")
    vals["rf_RMSE"] = rf.get("RMSE")
    vals["rf_R2"] = rf.get("R2")
    vals["rf_CV_R2_mean"] = rf.get("CV_R2_mean")
    vals["rf_CV_R2_std"] = rf.get("CV_R2_std")

    # CI
    vals["ci_dof"] = report.get("ci_dof")
    vals["ci_t_kritis"] = report.get("ci_t_kritis")

    ci_map = {r["parameter"]: r for r in report.get("ci_linreg", [])}
    for param in ["Intersep", "Chest", "Abdomen", "Hip"]:
        r = ci_map.get(param, {})
        vals[f"ci_{param}_koef"] = r.get("koef")
        vals[f"ci_{param}_se"] = r.get("se")
        vals[f"ci_{param}_bawah"] = r.get("ci_bawah")
        vals[f"ci_{param}_atas"] = r.get("ci_atas")
        vals[f"ci_{param}_std"] = r.get("koef_terstandarisasi")

    # VIF
    vif_map = {r["fitur"]: r["VIF"] for r in report.get("vif_fitur_terpilih", [])}
    for fitur in ["Abdomen", "Chest", "Hip"]:
        vals[f"vif_{fitur}"] = vif_map.get(fitur)

    # Korelasi
    corr_map = {}
    for r in report.get("korelasi_fitur_terpilih", []):
        key = tuple(sorted([r["fitur_1"], r["fitur_2"]]))
        corr_map[key] = r["korelasi"]
    vals["corr_Chest_Abdomen"] = corr_map.get(("Abdomen", "Chest"))
    vals["corr_Abdomen_Hip"] = corr_map.get(("Abdomen", "Hip"))
    vals["corr_Chest_Hip"] = corr_map.get(("Chest", "Hip"))

    # SHAP
    shap_data = report.get("shap_importance", {})
    for model_key, prefix in [
        ("Linear Regression", "shap_linreg"),
        ("Ridge Regression", "shap_ridge"),
        ("Random Forest", "shap_rf"),
    ]:
        shap_map = {r["fitur"]: r["mean_abs_shap"] for r in shap_data.get(model_key, [])}
        for fitur in ["Abdomen", "Hip", "Chest"]:
            vals[f"{prefix}_{fitur}"] = shap_map.get(fitur)

    vals["shap_base_linreg"] = report.get("shap_base_linreg")

    # RF importance
    rf_imp_map = {r["fitur"]: r["importance"] for r in report.get("feature_importance_rf", [])}
    for fitur in ["Abdomen", "Hip", "Chest"]:
        vals[f"rf_importance_{fitur}"] = rf_imp_map.get(fitur)

    return vals


def main():
    if not REPORT_PATH.exists():
        print(f"ERROR: {REPORT_PATH} tidak ditemukan. Jalankan `python train_model.py` terlebih dahulu.")
        sys.exit(1)

    with REPORT_PATH.open(encoding="utf-8") as f:
        report = json.load(f)

    actual = extract_values(report)

    # Cetak tabel
    header = f"{'Item':<35} {'Target':>12} {'Hasil':>12} {'Selisih':>12} {'Toleransi':>10} {'Status':>7}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    n_fail = 0
    n_ok = 0
    failures = []

    for item, (target, tol, kind) in TARGETS.items():
        result = actual.get(item)
        if result is None:
            status = "MISSING"
            diff = "N/A"
            n_fail += 1
            failures.append(item)
        elif kind == "exact":
            match = (result == target)
            diff = abs(result - target) if isinstance(result, (int, float)) else "N/A"
            status = "OK" if match else "FAIL"
            if not match:
                n_fail += 1
                failures.append(item)
            else:
                n_ok += 1
        else:
            diff = abs(float(result) - float(target))
            status = "OK" if diff <= tol else "FAIL"
            if status == "FAIL":
                n_fail += 1
                failures.append(item)
            else:
                n_ok += 1

        result_str = f"{result}" if result is not None else "N/A"
        diff_str = f"{diff:.6f}" if isinstance(diff, float) else str(diff)
        tol_str = f"{tol}" if tol > 0 else "exact"
        print(f"{item:<35} {str(target):>12} {result_str:>12} {diff_str:>12} {tol_str:>10} {status:>7}")

    print(sep)
    print(f"Total: {n_ok + n_fail} item | OK: {n_ok} | FAIL: {n_fail}")
    if failures:
        print(f"\nItem yang GAGAL ({len(failures)}):")
        for item in failures:
            target, tol, kind = TARGETS[item]
            result = actual.get(item, "N/A")
            if result != "N/A" and isinstance(result, (int, float)):
                diff = abs(float(result) - float(target))
                print(f"  {item}: target={target}, hasil={result}, selisih={diff:.6f}, toleransi={tol}")
            else:
                print(f"  {item}: target={target}, hasil={result}")
        sys.exit(1)
    else:
        print("\n✓ Semua angka dalam toleransi. Verifikasi LULUS.")
        sys.exit(0)


if __name__ == "__main__":
    main()
