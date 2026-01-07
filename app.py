import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import my_transformers
from my_transformers import DateTransformer # Memanggil DateTransformer kustom Anda

# Konfigurasi Halaman
st.set_page_config(page_title="FraudGuard: E-commerce Protection", layout="wide")

# 1. Load Model & Threshold Table
@st.cache_resource
def load_assets():
    model = joblib.load("fraud_model.pkl")
    threshold_data = pd.read_csv("threshold_table.csv")
    return model, threshold_data

model, threshold_table = load_assets()

# ---------------------------------------------------------
# SIDEBAR: 13 Fitur Input (Mirip Logika App Mobil)
# ---------------------------------------------------------
st.sidebar.header('📝 Detail Transaksi')

def get_user_input():
    # Fitur Numerik
    acc_age = st.sidebar.number_input('Account Age (Days)', 0, 10000, 365)
    total_trans = st.sidebar.number_input('Total Transactions User', 0, 5000, 10)
    avg_amount = st.sidebar.number_input('Average Amount User', 0.0, 50000.0, 500.0)
    shipping_dist = st.sidebar.number_input('Shipping Distance (KM)', 0.0, 20000.0, 15.0)
    
    # Fitur Kategorikal
    country = st.sidebar.selectbox('Country', ['USA', 'Germany', 'Turkey', 'Berlin', 'New York', 'London'])
    bin_country = st.sidebar.selectbox('BIN Country', ['USA', 'Germany', 'Turkey', 'UK'])
    channel = st.sidebar.selectbox('Channel', ['Web', 'App'])
    merchant = st.sidebar.selectbox('Merchant Category', ['Electronics', 'Fashion', 'Food', 'Travel', 'Health'])
    
    # Fitur Boolean (Flag)
    promo = st.sidebar.checkbox('Promo Used?')
    avs = st.sidebar.checkbox('AVS Match?')
    cvv = st.sidebar.checkbox('CVV Result Correct?')
    three_ds = st.sidebar.checkbox('3DS Flag Active?')
    
    # Fitur Waktu (Sangat penting untuk DateTransformer Anda)
    trans_date = st.sidebar.date_input('Transaction Date', datetime.now())
    trans_time = st.sidebar.time_input('Transaction Time', datetime.now())
    # Gabungkan menjadi format string yang dikenali DateTransformer
    full_time = f"{trans_date} {trans_time}"

    # Susun ke DataFrame (Pastikan urutan & nama kolom sama dengan saat training)
    data = {
        'account_age_days': acc_age,
        'total_transactions_user': total_trans,
        'avg_amount_user': avg_amount,
        'country': country,
        'bin_country': bin_country,
        'channel': channel,
        'merchant_category': merchant,
        'promo_used': 1 if promo else 0,
        'avs_match': 1 if avs else 0,
        'cvv_result': 1 if cvv else 0,
        'three_ds_flag': 1 if three_ds else 0,
        'transaction_time': full_time,
        'shipping_distance_km': shipping_dist
    }
    return pd.DataFrame([data])

user_data_df = get_user_input()

# ---------------------------------------------------------
# HALAMAN UTAMA
# ---------------------------------------------------------
st.title("🛡️ E-commerce Fraud Detection")
st.markdown("Aplikasi ini menggunakan model XGBoost dengan optimasi biaya bisnis.")

col1, col2 = st.columns([1, 2])

# Kolom Kiri: Konfigurasi Biaya (Logika Ekonomi Anda)
with col1:
    st.subheader("💰 Business Cost Logic")
    cost_fn = st.number_input("Cost of False Negative (Loss)", value=650)
    cost_fp = st.number_input("Cost of False Positive (Operational)", value=30)
    
    # Hitung Threshold Optimal
    threshold_table["total_cost"] = (threshold_table["FN"] * cost_fn) + (threshold_table["FP"] * cost_fp)
    best_threshold = threshold_table.loc[threshold_table["total_cost"].idxmin()]["threshold_model"]
    
    st.metric("Optimal Threshold", round(best_threshold, 3))
    st.info("Threshold ini meminimalkan kerugian finansial, bukan hanya akurasi.")

# Kolom Kanan: Hasil Prediksi & Visualisasi
with col2:
    st.subheader("📊 Prediction Results")
    
    if st.button("Jalankan Deteksi", use_container_width=True):
        # 1. Dapatkan Probabilitas
        prob = model.predict_proba(user_data_df)[0, 1]
        
        # 2. Bandingkan dengan Threshold Optimal
        is_fraud = prob >= best_threshold
        
        # 3. Tampilkan Hasil
        if is_fraud:
            st.error(f"### HASIL: TERINDIKASI FRAUD")
        else:
            st.success(f"### HASIL: TRANSAKSI AMAN")
            
        st.write(f"**Probabilitas Fraud:** {prob:.2%}")

        # 4. Gauge Chart untuk visualisasi (Mirip Plotly di app mobil)
        fig = px.pie(
            values=[prob, 1-prob], 
            names=['Risk', 'Safe'],
            hole=0.7,
            color_discrete_sequence=['#e74c3c', '#2ecc71']
        )
        st.plotly_chart(fig, use_container_width=True)
# Tambahkan grafik garis untuk melihat kurva biaya
fig_cost = px.line(
    threshold_table, 
    x='threshold_model', 
    y='total_cost', 
    title='Total Cost vs Model Threshold'
)
# Tambahkan titik merah di lokasi threshold optimal
fig_cost.add_vline(x=best_threshold, line_dash="dash", line_color="red", annotation_text="Optimal")
st.plotly_chart(fig_cost, use_container_width=True)

# Footer info
st.markdown("---")
st.caption("Developed by Muhammad Jundy Andymurti - Data Science Portfolio")