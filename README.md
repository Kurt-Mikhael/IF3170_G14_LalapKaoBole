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

Pilih kernel dari `.venv`, lalu **Restart Kernel and Run All Cells**. Notebook
menampilkan keputusan CV pengembangan, menyelesaikan laporan holdout, lalu
melatih ulang model terpilih dan memprediksi baris test pada tahap `final-fit`.
Sel `pair-publish` terakhir memvalidasi ID/label dan menulis lima CSV submisi
serta lima PKL ke `output/`; hanya sel ini yang menulis berkas model/submisi:

| Jenis | Nama berkas |
| --- | --- |
| CSV | `submission_dtl.csv`, `submission_knn.csv`, `submission_logreg.csv`, `submission_svm.csv`, `submission_bayes.csv` |
| PKL | `dtl.pkl`, `knn.pkl`, `logreg.pkl`, `svm.pkl`, `bayes.pkl` |

Pemilihan final memakai hasil CV pengembangan: DTL/C4.5 memakai pemenang CV
berpasangan dengan REP tanpa oversampling; SVM memakai keputusan CV berpasangan
dengan baseline iterasi 04 (16.000 baris); KNN, regresi logistik, dan Naive
Bayes memakai pemenang `fe_winners`. Tabel keputusan menampilkan kandidat,
fitur, ambang bila relevan, dan sumber sebelum refit. Holdout hanya untuk
pelaporan. Notebook menyimpan visualisasi pohon sebagai `output/c45_tree.png`.
Laporan memakai aset di `docs/figures/` dan gambar TAN di `output/tan_structure.png`.

Notebook tidak membaca `ground_truth.csv` atau mengevaluasi prediksi submisi
terhadap label test. Evaluasi tersebut dilakukan terpisah di luar notebook
saat label test tersedia. ID test tetap divalidasi terhadap
`sample_submission.csv` sebelum berkas ditulis.

Artefak eksperimen lama dan bonus TAN yang sudah ada di `output/` bukan
bagian dari lima pasangan CSV/PKL final.

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
