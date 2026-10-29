# Deteksi dan Diagnosis Gangguan Motor Induksi (KNN, Decision Tree, Random Forest)

Proyek 01, mata kuliah **Machine Learning dalam Sistem Kendali dan Energi**, Teknik Elektro, Universitas Bung Hatta.

> Ganti baris ini dengan nama, NPM, dan link aplikasi live Anda:
> **Nama:** ... | **NPM:** ... | **Live app:** https://<nama-app>.streamlit.app

## Ringkasan

Aplikasi web ini membaca lima sensor motor induksi (arus RMS, suhu stator, getaran, faktor daya, kecepatan putar) dan bekerja dua tingkat:

1. **Deteksi**: motor **Normal** atau **Fault**.
2. **Diagnosis**: bila Fault, jenisnya **Electrical**, **Mechanical**, atau **Thermal**, beserta rekomendasi pemeriksaan.

Tiga algoritme dibandingkan di Google Colab: KNN (dengan StandardScaler), Decision Tree, dan Random Forest. Model terbaik tiap tingkat dipakai di aplikasi Streamlit.

## Struktur repo

```
├── app.py                                   # aplikasi Streamlit
├── requirements.txt                         # versi library dari Colab
├── model/
│   ├── detector.joblib                      # Normal vs Fault
│   ├── diagnosis.joblib                     # jenis gangguan
│   └── metadata.json                        # fitur, rentang, metrik, versi
├── data/
│   ├── M6-dataset_kondisi_motor.csv         # dataset (1.200 pengukuran, 12 motor)
│   └── sample_log.csv                       # 60 pengukuran contoh
└── notebooks/
    └── 01_deteksi_gangguan_motor.ipynb      # training di Google Colab
```

## Hasil model (data uji)

| Tingkat | Model terpilih | Metrik |
|---|---|---|
| Deteksi | ... | Accuracy ..., Recall Fault ..., F1 macro ... |
| Diagnosis | ... | Accuracy ..., F1 macro ... |

*Isi dengan angka dari notebook Anda.*

## Menjalankan di komputer sendiri (Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push ke GitHub, buka https://share.streamlit.io, **Create app**, pilih repo dan `app.py`. Di **Advanced settings** pilih **Python 3.12** sebelum klik Deploy.

## Penggunaan AI-assisted coding

Tuliskan tools AI yang dipakai, untuk bagian apa, dan contoh prompt penting.
