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
├── output/  # Model, hasil eksperimen, dan CSV submission
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

Gunakan scikit-learn 1.7.1 untuk memuat preprocessor yang disimpan dan
mengulang hasil submission. Versi lain dapat menghasilkan pembagian data
latih dan validasi yang berbeda meskipun nilai seed sama.

## Menjalankan Notebook

Jalankan Jupyter dari direktori notebook agar data ditemukan:

```bash
cd src/notebooks
jupyter lab main.ipynb
```

Pilih **Restart Kernel and Run All Cells**. Notebook melatih model dan menulis
hasilnya ke `output/`.

| Submission | Model dan parameter prapemrosesan |
| --- | --- |
| `submission_c45.csv` | `c45_final.pkl` dan `c45_rare_categories.pkl` |
| `submission_knn.csv` | `knn.pkl` dan `knn_preprocessor.pkl` |
| `submission_logreg.csv` | `logreg_bundle.pkl` (model dan preprocessor) |
| `submission_naive_bayes.csv` | `naive_bayes.pkl` dan `naive_bayes_preprocessing.pkl` |
| `submission_svm.csv` | `svm.pkl` dan `svm_preprocessor.pkl` |

`logreg_l2.pkl` dan `svm_smo_gaussian.pkl` adalah model contoh untuk
demonstrasi penyimpanan, bukan model yang menghasilkan submission.

## Kontributor

| NIM      | Nama                           |
| -------- | ------------------------------ |
| 13524037 | Nicholas Wise Saragih Sumbayak |
| 13524062 | Nathan E C Marpaung            |
| 13524065 | Kurt Mikhael Purba             |
| 13524089 | Aurelia Jennifer Gunawan       |
| 13524131 | Amanda Aurellia Salsabila      |
