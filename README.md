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
│   │   └── knn.py
│   └── notebooks/
│       └── v1.ipynb
└── README.md
```

## Prasyarat

- [uv](https://docs.astral.sh/uv/getting-started/installation/) untuk mengelola versi Python, virtual environment, dan dependensi.

## Setup

Jalankan dari direktori utama repository:

```bash
uv venv --python 3.12
uv pip install jupyterlab notebook pandas numpy matplotlib seaborn scikit-learn scipy
source .venv/bin/activate
```

## Menjalankan Notebook EDA

Jalankan Jupyter dari direktori notebook agar data ditemukan:

```bash
cd src/notebooks
jupyter lab v1.ipynb
```

Pilih **Run All** untuk menjalankan notebook.

## Kontributor

| NIM      | Nama                           |
| -------- | ------------------------------ |
| 13524037 | Nicholas Wise Saragih Sumbayak |
| 13524062 | Nathan E C Marpaung            |
| 13524065 | Kurt Mikhael Purba             |
| 13524089 | Aurelia Jennifer Gunawan       |
| 13524131 | Amanda Aurellia Salsabila      |
