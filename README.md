# Prediksi Persentase Body Fat
### Regresi Linear + SelectKBest + Interpretasi SHAP
Hafiz Akbar Habibie Wijaya · 23.53.028035 · Ilmu Komputer UMPR · 2026

---

## Isi Folder
```
skripsi-bodyfat/
├── train_model.py     <- skrip latih model dan evaluasi
├── app.py             <- aplikasi Streamlit (prediksi + SHAP)
├── bodyfat.csv        <- dataset (252 sampel, 15 kolom)
├── requirements.txt   <- daftar library
└── README.md          <- file ini
```
Folder `outputs/` akan dibuat OTOMATIS setelah menjalankan `train_model.py`.
`notebook_bodyfat.ipynb` adalah catatan analisis terdahulu; untuk hasil yang konsisten dengan perbaikan metodologi terbaru, gunakan `train_model.py` sebagai sumber training utama.

---

## Cara Menjalankan di Visual Studio Code

Gunakan **Python 3.12.10** (disarankan) dan versi pustaka yang tercantum di `requirements.txt`.

### 1. Buka folder
VS Code → **File → Open Folder** → pilih folder `skripsi-bodyfat`.

### 2. Buka terminal
Menu **Terminal → New Terminal** (atau tekan Ctrl + `).

### 3. Install library (sekali saja)
```bash
pip install -r requirements.txt
```
> Kalau `pip` tidak dikenal, coba `pip3` atau `python -m pip install -r requirements.txt`

### 4. Latih model (WAJIB dijalankan duluan)
```bash
python train_model.py
```
Berhasil jika muncul output `[TAHAP 1] ... ===== SELESAI =====` dan
folder `outputs/` terisi (model + grafik).

### 5. Jalankan aplikasi
```bash
streamlit run app.py
```
Di formulir, berat badan dimasukkan dalam kg dan lingkar perut dalam cm. Aplikasi mengonversi berat ke pound secara internal karena model dilatih dengan satuan dataset tersebut.
Browser terbuka otomatis di http://localhost:8501
Masukkan data antropometri → klik **Prediksi**.

Hentikan aplikasi: klik di terminal lalu tekan **Ctrl + C**.

---

## Jika Ada Error

| Error | Solusi |
|-------|--------|
| `FileNotFoundError: bodyfat.csv` | Pastikan terminal berada di dalam folder `skripsi-bodyfat` |
| `ModuleNotFoundError` | Ulangi langkah 3 |
| `streamlit: command not found` | Pakai `python -m streamlit run app.py` |
| App: artefak model tidak cocok atau tidak lengkap | Jalankan ulang `python train_model.py` untuk membuat seluruh model dan laporan |

---

## Metodologi dan hasil
- `Density` dihapus karena berhubungan deterministik dengan target dan dapat menyebabkan kebocoran target.
- Data dibagi menjadi train dan test. Baris outlier tidak dihapus agar keputusan filtrasi tidak menggunakan informasi lintas-fold pada evaluasi CV.
- Imputasi, scaling, dan seleksi fitur dilakukan di dalam pipeline. Pada evaluasi CV Ridge dan Random Forest, preprocessing dan seleksi fitur di-fit ulang pada setiap fold.
- Skor CV utama dihitung dengan nested cross-validation: inner CV memilih parameter, outer CV mengukur proses pemilihan. Evaluasi test tetap merupakan estimasi terpisah pada split ini.
- Aplikasi Streamlit memakai jumlah fitur tetap `K=7`. Skrip tetap menghitung kandidat `[3, 5, 7, 10]` sebagai analisis pembanding; model dan artefak final untuk aplikasi dilatih dengan K=7.
- Laporan dan model di `outputs/` hanya valid setelah dibuat ulang dengan `python train_model.py`; jalankan ulang setelah memperbarui kode. Artefak lama mungkin tidak cocok dengan aplikasi.

---

## Lisensi
MIT License

Copyright (c) 2026 Hafiz Akbar Habibie Wijaya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
