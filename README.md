# IF3170_G14_LalapKaoBole

Repository Tugas Besar 2 IF3170 Inteligensi Artifisial, Kelompok LalapKaoBole, klasifikasi pembatalan pemesanan penginapan.

## Struktur Proyek

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── docs/
│   ├── laporan.tex
│   ├── laporan.pdf
│   └── logo-itb.png
├── src/
│   ├── models/
│   │   ├── c45/
│   │   ├── knn/
│   │   ├── logreg/
│   │   ├── naive_bayes/
│   │   └── svm/
│   └── notebooks/
│       └── main.ipynb
├── output/  # Lima CSV submission dan lima berkas model final
└── README.md
```

## Prasyarat

- [uv](https://docs.astral.sh/uv/getting-started/installation/) untuk mengelola versi Python, virtual environment, dan dependensi.

## Setup

Jalankan dari direktori utama repository:

```bash
uv venv --python 3.12
uv pip install jupyterlab notebook pandas numpy matplotlib seaborn 'scikit-learn==1.7.1' scipy imageio-ffmpeg
source .venv/bin/activate
```

Notebook hanya menerima kernel dengan `scikit-learn==1.7.1`. Versi lain
dapat menghasilkan pembagian data latih dan validasi yang berbeda meskipun
nilai seed sama. Jika `.venv` sudah ada, perbarui versi yang terpasang:

```bash
uv pip install --python .venv/bin/python 'scikit-learn==1.7.1'
```

## Menjalankan Notebook

Jalankan Jupyter dari direktori notebook agar data ditemukan:

```bash
cd src/notebooks
jupyter lab main.ipynb
```

Pilih kernel dari `.venv`, lalu **Restart Kernel and Run All Cells** untuk
membuat lima CSV submisi dan lima berkas PKL di `output/`. Hanya sel ekspor
akhir yang menulis ke direktori tersebut:

| Jenis | Nama berkas |
| --- | --- |
| CSV | `submission_dtl.csv`, `submission_knn.csv`, `submission_logreg.csv`, `submission_svm.csv`, `submission_bayes.csv` |
| PKL | `dtl.pkl`, `knn.pkl`, `logreg.pkl`, `svm.pkl`, `bayes.pkl` |

Pemilihan model final memakai CV pengembangan yang sudah ada. DTL/C4.5 dan SVM
menggunakan pemenang eksperimen berpasangan; eksperimen SVM tersebut memakai
16.000 baris. KNN, regresi logistik, dan Bayes menggunakan pemenang pencarian
fitur lintas model. Tabel dan plot eksperimen tetap tampil di notebook;
gambar dan video yang dirujuk laporan tersimpan di `docs/figures/`.

Setiap PKL berisi sebuah dictionary dengan model terlatih yang tepat untuk CSV
pasangannya serta konfigurasi atau referensi prapemrosesan yang diperlukan
untuk prediksi. Struktur prapemrosesan mengikuti kebutuhan model.
Artefak yang sudah tersimpan berasal dari eksekusi dengan scikit-learn 1.9.1.
Jalankan ulang notebook untuk menggantinya dengan hasil dari versi 1.7.1;
kompatibilitas PKL lintas versi tidak dijamin.

Pencarian fitur lintas model membandingkan lima algoritma memakai F1 makro
dari empat fold pengembangan. Eksperimen berpasangan memilih C4.5 dengan
pruning tanpa oversampling dan SVM iterasi 04 (16.000 baris). Holdout hanya
dipakai untuk melaporkan hasil, bukan untuk memilih model final.

## Kontributor

| NIM      | Nama                           |
| -------- | ------------------------------ |
| 13524037 | Nicholas Wise Saragih Sumbayak |
| 13524062 | Nathan E C Marpaung            |
| 13524065 | Kurt Mikhael Purba             |
| 13524089 | Aurelia Jennifer Gunawan       |
| 13524131 | Amanda Aurellia Salsabila      |
