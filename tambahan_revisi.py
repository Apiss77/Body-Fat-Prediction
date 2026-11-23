"""
Analisis tambahan revisi skripsi (tanpa IQR, k dipilih dengan CV, intersep & CI koefisien).

Pemakaian:  python tambahan_revisi.py bodyfat.csv
Butuh: pandas, numpy, scipy, scikit-learn, statsmodels
Hasil dicetak ke layar dan disimpan ke tambahan_revisi.json
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings("ignore")
RS = 42

def ci_ols(X, y, level=0.95):
    """Intersep, koefisien (satuan asli), galat baku, dan CI dari OLS: b +- t * SE,
    SE = sqrt(diag(s^2 (X'X)^-1)), s^2 = RSS / (n - p - 1)."""
    Xd = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    beta = np.linalg.solve(Xd.T @ Xd, Xd.T @ np.asarray(y, float))
    resid = np.asarray(y, float) - Xd @ beta
    dof = len(y) - Xd.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(s2 * np.diag(np.linalg.inv(Xd.T @ Xd)))
    t = stats.t.ppf(1 - (1 - level) / 2, dof)
    return beta, se, beta - t * se, beta + t * se, dof, t

def lin(k=None):
    s = [("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]
    if k: s.append(("k", SelectKBest(f_regression, k=k)))
    return Pipeline(s + [("r", LinearRegression())])

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "bodyfat.csv")
X = df.drop(columns=["BodyFat", "Density"]); y = df["BodyFat"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS)  # tanpa IQR
cv5 = KFold(5, shuffle=True, random_state=RS)
out = {"n_train": len(Xtr), "n_test": len(Xte)}

# 1) Pemilihan k (CV 5-lipatan) dan fitur terpilih
cvk = {k: float(cross_val_score(lin(k), Xtr, ytr, cv=cv5, scoring="r2").mean()) for k in [3, 5, 7, 10]}
best_k = max(cvk, key=cvk.get); out["cv_r2_per_k"] = cvk; out["best_k"] = best_k
sel = list(Xtr.columns[SelectKBest(f_regression, k=best_k).fit(Xtr, ytr).get_support()])
out["selected"] = sel
print("CV R2 per k:", {k: round(v, 4) for k, v in cvk.items()}, "-> k =", best_k, sel)

# 2) Intersep dan CI 95% koefisien (satuan asli)
b, se, lo, hi, dof, t = ci_ols(Xtr[sel].values, ytr.values)
names = ["Intersep"] + sel
out["ci"] = {n: dict(koef=b[i], se=se[i], bawah=lo[i], atas=hi[i]) for i, n in enumerate(names)}
out["t_kritis"], out["dof"] = float(t), int(dof)
print(f"\nCI 95% OLS (dof={dof}, t={t:.4f})")
for i, n in enumerate(names):
    print(f"{n:10s} {b[i]:9.4f}  SE {se[i]:.4f}  [{lo[i]:9.4f}, {hi[i]:9.4f}]")
chk = sm.OLS(ytr.values, sm.add_constant(Xtr[sel].values)).fit()   # pemeriksaan silang
assert np.allclose(chk.conf_int(), np.c_[lo, hi]), "CI tidak sama dengan statsmodels"

# 3) VIF 13 fitur dan fitur terpilih (dengan konstanta, data latih)
vif13 = {c: float(variance_inflation_factor(sm.add_constant(Xtr).values, i + 1)) for i, c in enumerate(Xtr.columns)}
vifs = {c: float(variance_inflation_factor(sm.add_constant(Xtr[sel]).values, i + 1)) for i, c in enumerate(sel)}
out["vif_13"], out["vif_selected"] = vif13, vifs
print("\nVIF 13 fitur:", {k: round(v, 2) for k, v in sorted(vif13.items(), key=lambda x: -x[1])})
print("VIF terpilih:", {k: round(v, 2) for k, v in vifs.items()})
print("Korelasi fitur terpilih:\n", Xtr[sel].corr().round(4))

# 4) CV berulang (5-lipatan x 20 ulangan) untuk kestabilan pemilihan k
rk = RepeatedKFold(n_splits=5, n_repeats=20, random_state=RS)
out["cv_berulang"] = {}
print("\nCV berulang (rata-rata R2 +- sd):")
for name, k in [("baseline", None), ("k=3", 3), ("k=5", 5), ("k=7", 7), ("k=10", 10)]:
    s = cross_val_score(lin(k), Xtr, ytr, cv=rk, scoring="r2")
    out["cv_berulang"][name] = [float(s.mean()), float(s.std())]
    print(f"  {name:9s} {s.mean():.4f} +- {s.std():.4f}")

# 5) Data uji dengan dan tanpa sampel ekstrem (BodyFat tertinggi)
ie = yte.idxmax(); mask = (yte.index != ie)
print("\nData uji penuh vs tanpa sampel ekstrem (R2, RMSE):")
out["uji"] = {}
for name, k in [("baseline", None), ("k=3", 3), ("k=5", 5), ("k=7", 7), ("k=10", 10)]:
    p = lin(k).fit(Xtr, ytr).predict(Xte)
    r = dict(R2_51=r2_score(yte, p), RMSE_51=mean_squared_error(yte, p) ** .5,
             R2_50=r2_score(yte[mask], p[mask]), RMSE_50=mean_squared_error(yte[mask], p[mask]) ** .5)
    out["uji"][name] = {a: float(v) for a, v in r.items()}
    print(f"  {name:9s} " + "  ".join(f"{a}={v:.4f}" for a, v in r.items()))
json.dump(out, open("tambahan_revisi.json", "w"), indent=1, default=float)
