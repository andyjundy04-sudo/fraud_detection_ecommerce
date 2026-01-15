import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime
import plotly.express as px 
from pathlib import Path

# --- HACK UNTUK PICKLE IMPORT ERROR ---
#sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# ---------------------------------------------------------
# LOAD ASSETS
# ---------------------------------------------------------

#BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

@st.cache_resource
def load_assets():
    model = joblib.load(BASE_DIR / "fraud_model.pkl")
    threshold_table = pd.read_csv(BASE_DIR / "threshold_table.csv")
    fi_df = pd.read_csv(BASE_DIR / "feature_importance_permutation.csv")
    return model, threshold_table, fi_df

model, threshold_table, fi_df = load_assets()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header('📝 Detail Transaksi')

def get_user_input():
    acc_age = st.sidebar.number_input('Account Age (Days)', 0, 10000, 365)
    total_trans = st.sidebar.number_input('Total Transactions User', 0, 5000, 10)
    
    amount = st.sidebar.number_input('Amount User', 0.0, 50000.0, 500.0)
    shipping_dist = st.sidebar.number_input('Shipping Distance (KM)', 0.0, 20000.0, 15.0)
    
    country = st.sidebar.selectbox('Country', ['USA', 'Germany', 'Turkey', 'Berlin', 'New York', 'London'])
    channel = st.sidebar.selectbox('Channel', ['Web', 'App'])
    merchant = st.sidebar.selectbox('Merchant Category', ['Electronics', 'Fashion', 'Food', 'Travel', 'Health'])
    
    promo = st.sidebar.checkbox('Promo Used?')
    three_ds = st.sidebar.checkbox('3DS Flag Active?')
    
    trans_date = st.sidebar.date_input('Transaction Date', datetime.now())
    trans_time = st.sidebar.time_input('Transaction Time', datetime.now())

    dt = datetime.combine(trans_date, trans_time)

    # SESUAIKAN NAMA KEY DENGAN KOLOM TRAINING (15 KOLOM)
    data = {
        'account_age_days': acc_age,
        'total_transactions_user': total_trans,
        'amount': amount, # Sesuai dataset asli Anda
        'country': country,
        'channel': channel,
        'merchant_category': merchant,
        'promo_used': 1 if promo else 0,
        'three_ds_flag': 1 if three_ds else 0,
        'shipping_distance_km': shipping_dist,
        'month': dt.month,
        'day': dt.weekday(),
        'hour': dt.hour
    }
    return pd.DataFrame([data])

user_data_df = get_user_input()

# --- Auto handle transaction_time ---
if 'transaction_time' not in user_data_df.columns:
    # kalau kolom tidak ada, isi dengan current datetime
    user_data_df['transaction_time'] = datetime.now()
else:
    # kalau ada, convert ke datetime, salah format jadi NaT
    user_data_df['transaction_time'] = pd.to_datetime(
        user_data_df['transaction_time'], errors='coerce'
    )

# ---------------------------------------------------------
# HALAMAN UTAMA
# ---------------------------------------------------------
st.title("🛡️ E-commerce Fraud Detection")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("💰 Business Expected Cost Loss")
    cost_fn = st.number_input("Cost of False Negative ($)", value=650)
    cost_fp = st.number_input("Cost of False Positive ($)", value=30)

    threshold_tmp = threshold_table.copy()
    threshold_tmp["total_cost"] = (threshold_tmp["FN"] * cost_fn) + (threshold_tmp["FP"] * cost_fp)
    best_threshold = threshold_tmp.loc[threshold_tmp["total_cost"].idxmin()]["threshold_model"]
    
    st.metric("Threshold Model", round(best_threshold, 3))

    # st.subheader("🔍 Feature Importance (Permutation, Global)")

    top_fi = fi_df.head(10)

    fig_fi = px.bar(
        top_fi,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance"
    )

    st.plotly_chart(fig_fi, width="stretch")
    
with col2:
    st.subheader("📊 Prediction Results")
    if st.button("Jalankan Deteksi", use_container_width=True):
        prob = model.predict_proba(user_data_df)[0, 1]
        is_fraud = prob >= best_threshold
        
        if is_fraud:
            st.error(f"### HASIL: TERINDIKASI FRAUD")
        else:
            st.success(f"### HASIL: TRANSAKSI AMAN")
        st.write(f"**Probabilitas Fraud:** {prob:.2%}")

        fig = px.pie(values=[prob, 1-prob], names=['Risk', 'Safe'], hole=0.4,
                     color_discrete_sequence=['#e74c3c', '#2ecc71'])
        st.plotly_chart(fig, width="stretch")

# Grafik Biaya di bagian bawah
st.markdown("---")
fig_cost = px.line(threshold_tmp, x='threshold_model', y='total_cost', 
                  title='Analisis Biaya (Semakin Rendah Semakin Baik)')
fig_cost.add_vline(x=best_threshold, line_dash="dash", line_color="red")
st.plotly_chart(fig_cost, width="stretch")