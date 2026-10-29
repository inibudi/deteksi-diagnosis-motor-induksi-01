"""
Aplikasi Streamlit: Deteksi dan Diagnosis Gangguan Motor Induksi
Proyek 01, Machine Learning dalam Sistem Kendali dan Energi, Teknik Elektro Universitas Bung Hatta

Jalankan lokal:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
SAMPLE_CSV = BASE / "data" / "sample_log.csv"

st.set_page_config(page_title="Diagnosis Gangguan Motor", page_icon="⚙️", layout="wide")

# Rekomendasi pemeriksaan umum per jenis gangguan (untuk pembelajaran, bukan prosedur resmi)
REKOMENDASI = {
    "Electrical": [
        "Ukur tegangan suplai dan keseimbangan arus tiga fasa.",
        "Periksa tahanan isolasi belitan stator (uji megger).",
        "Periksa sambungan terminal yang longgar atau terbakar.",
    ],
    "Mechanical": [
        "Periksa kondisi dan pelumasan bearing.",
        "Cek kelurusan (alignment) poros dan kopling.",
        "Periksa ketidakseimbangan rotor dan baut dudukan motor.",
    ],
    "Thermal": [
        "Periksa kipas pendingin dan kebersihan sirip motor.",
        "Bandingkan beban aktual dengan rating motor (overload).",
        "Cek suhu ruangan dan ventilasi di sekitar motor.",
    ],
}
WARNA = {"Normal": "#2E75B6", "Electrical": "#E67E22", "Mechanical": "#C0392B", "Thermal": "#7D3C98"}


@st.cache_resource
def load_models():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    detector = joblib.load(MODEL_DIR / "detector.joblib")
    diagnosis = joblib.load(MODEL_DIR / "diagnosis.joblib")
    return detector, diagnosis, meta


detector, diagnosis, meta = load_models()
FEATURES, UNITS = meta["features"], meta["units"]
LABEL = {
    "current_rms_a": "Arus RMS",
    "stator_temperature_c": "Suhu stator",
    "vibration_mm_s": "Getaran",
    "power_factor": "Faktor daya",
    "rotational_speed_rpm": "Kecepatan putar",
}


def predict(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    X = df[FEATURES].astype(float)
    p_fault = detector.predict_proba(X)[:, list(detector.classes_).index("Fault")]
    jenis = diagnosis.predict(X)
    proba_jenis = diagnosis.predict_proba(X).max(axis=1)
    status = np.where(p_fault >= threshold, "FAULT", "Normal")
    return pd.DataFrame({
        "status": status,
        "prob_fault": p_fault.round(3),
        "jenis_gangguan": np.where(status == "FAULT", jenis, "-"),
        "keyakinan_jenis": np.where(status == "FAULT", proba_jenis.round(3), np.nan),
    })


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    threshold = st.slider("Ambang probabilitas fault", 0.05, 0.95, 0.50, 0.05,
                          help="Motor dinyatakan FAULT bila probabilitas ≥ ambang. Ambang lebih rendah "
                               "menangkap lebih banyak gangguan, tetapi false alarm bertambah.")
    st.divider()
    st.subheader("Model yang dipakai")
    d, g = meta["detector"], meta["diagnosis"]
    st.metric(f"Deteksi: {d['model']}", f"F1 {d['f1_macro']:.3f}", help=f"Parameter: {d['params']}")
    st.metric(f"Diagnosis: {g['model']}", f"F1 {g['f1_macro']:.3f}", help=f"Parameter: {g['params']}")
    st.caption("F1 macro pada data uji.")

st.title("⚙️ Deteksi dan Diagnosis Gangguan Motor Induksi")
st.write("Masukkan lima pengukuran sensor. Model **deteksi** menentukan apakah motor normal atau mengalami "
         "gangguan; bila gangguan, model **diagnosis** menebak jenisnya: **Electrical**, **Mechanical**, "
         "atau **Thermal**.")

tab_cek, tab_csv, tab_info = st.tabs(["🔧 Cek kondisi motor", "📄 Analisis log CSV", "ℹ️ Tentang model"])

# --- Tab 1: input sensor ---
with tab_cek:
    preset = st.radio("Isi otomatis dengan rata-rata kelas", ["Normal", "Electrical", "Mechanical", "Thermal"],
                      horizontal=True)
    base = meta["class_means"][preset]
    with st.form("sensor"):
        cols = st.columns(5)
        nilai = {}
        for c, f in zip(cols, FEATURES):
            lo, hi = meta["ranges"][f]
            step = 0.001 if f == "power_factor" else 0.1
            nilai[f] = c.number_input(f"{LABEL[f]} ({UNITS[f]})" if UNITS[f] else LABEL[f],
                                      min_value=float(np.floor(lo * 0.8)), max_value=float(np.ceil(hi * 1.2)),
                                      value=float(base[f]), step=step, format="%.3f" if f == "power_factor" else "%.2f",
                                      key=f"{preset}-{f}")
        ok = st.form_submit_button("🔍 Analisis", type="primary")

    if ok:
        r = predict(pd.DataFrame([nilai]), threshold).iloc[0]
        if r["status"] == "FAULT":
            st.error(f"### ⚠️ FAULT terdeteksi (probabilitas {r['prob_fault']:.0%})\n"
                     f"Jenis gangguan paling mungkin: **{r['jenis_gangguan']}** "
                     f"(keyakinan {r['keyakinan_jenis']:.0%})")
            proba = pd.Series(diagnosis.predict_proba(pd.DataFrame([nilai])[FEATURES])[0], index=diagnosis.classes_)
            left, right = st.columns(2)
            with left:
                st.write("**Probabilitas tiap jenis gangguan**")
                st.bar_chart(proba.rename("probabilitas"))
            with right:
                st.write("**Rekomendasi pemeriksaan**")
                for t in REKOMENDASI.get(r["jenis_gangguan"], []):
                    st.write(f"- {t}")
                st.caption("Rekomendasi umum untuk pembelajaran; ikuti prosedur K3 dan manual pabrikan.")
        else:
            st.success(f"### ✅ Motor NORMAL (probabilitas fault {r['prob_fault']:.0%})")

        st.write("**Perbandingan dengan rentang normal** (persentil 5 sampai 95 data Normal)")
        band = meta["normal_band"]
        rows = []
        for f in FEATURES:
            lo, hi = band[f]
            v = nilai[f]
            rows.append({"Sensor": LABEL[f], "Nilai": v, "Rentang normal": f"{lo:.2f} sampai {hi:.2f} {UNITS[f]}",
                         "Keterangan": "di atas normal" if v > hi else "di bawah normal" if v < lo else "normal"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# --- Tab 2: CSV ---
with tab_csv:
    st.write("Unggah log pengukuran (CSV) dengan kolom: " + ", ".join(f"`{f}`" for f in FEATURES) +
             ". Kolom lain seperti `motor_id` dan `timestamp` boleh ada.")
    up = st.file_uploader("File CSV", type="csv")
    use_sample = st.toggle("Gunakan log contoh (60 pengukuran)", value=up is None)
    df = pd.read_csv(SAMPLE_CSV) if use_sample else (pd.read_csv(up) if up is not None else None)

    if df is not None:
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
        else:
            res = predict(df, threshold)
            out = pd.concat([df.reset_index(drop=True), res], axis=1)
            n_fault = int((res["status"] == "FAULT").sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Jumlah pengukuran", len(out))
            k2.metric("Terdeteksi FAULT", n_fault)
            k3.metric("Persentase FAULT", f"{n_fault / len(out):.1%}")

            if "condition" in df.columns:
                acc = (res["status"].str.upper() == df["condition"].str.upper()).mean()
                st.info(f"File memuat kolom `condition`, jadi hasil bisa dicek: akurasi deteksi = **{acc:.1%}**")

            left, right = st.columns(2)
            with left:
                st.subheader("Jenis gangguan terdeteksi")
                jenis = res.loc[res["status"] == "FAULT", "jenis_gangguan"].value_counts()
                if len(jenis):
                    st.bar_chart(jenis.rename("jumlah"))
                else:
                    st.write("Tidak ada gangguan terdeteksi.")
            with right:
                if "motor_id" in out.columns:
                    st.subheader("FAULT per motor")
                    st.bar_chart(out.groupby("motor_id")["status"].apply(lambda s: (s == "FAULT").sum()).rename("jumlah fault"))

            st.subheader("Hasil per pengukuran")
            st.dataframe(out, width="stretch", height=360)
            st.download_button("⬇️ Unduh hasil (CSV)", out.to_csv(index=False).encode(), "hasil_diagnosis.csv", "text/csv")

# --- Tab 3: model ---
with tab_info:
    st.subheader("Perbandingan tiga model: deteksi Normal vs Fault")
    st.dataframe(pd.DataFrame(meta["comparison_detection"]).set_index("Model"), width="stretch")
    st.subheader("Perbandingan tiga model: diagnosis jenis gangguan")
    st.dataframe(pd.DataFrame(meta["comparison_diagnosis"]).set_index("Model"), width="stretch")
    st.subheader("Feature importance (Random Forest)")
    imp = pd.DataFrame(meta["importance"])
    imp.index = [LABEL[i] for i in imp.index]
    st.bar_chart(imp, stack=False)
    ts = meta["two_stage"]
    st.caption(f"Sistem dua tingkat pada data uji: akurasi {ts['accuracy']:.1%}, F1 macro {ts['f1_macro']:.3f}. "
               f"Dilatih dengan Python {meta['versions']['python']}, scikit-learn {meta['versions']['scikit-learn']}, "
               f"numpy {meta['versions']['numpy']}.")
