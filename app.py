import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime
import plotly.express as px 

# --- HACK UNTUK PICKLE IMPORT ERROR ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# ---------------------------------------------------------
# LOAD ASSETS
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    curr_dir = os.path.dirname(__file__)
    model_path = os.path.join(curr_dir, "fraud_model.pkl")
    table_path = os.path.join(curr_dir, "threshold_table.csv")
    
    model = joblib.load(model_path)
    threshold_data = pd.read_csv(table_path)
    return model, threshold_data

model, threshold_table = load_assets()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header('📝 Detail Transaksi')

def get_user_input():
    acc_age = st.sidebar.number_input('Account Age (Days)', 0, 10000, 365)
    total_trans = st.sidebar.number_input('Total Transactions User', 0, 5000, 10)
    # Pastikan nama variabel di sini 'amount' tapi di dictionary 'avg_amount_user'
    amount = st.sidebar.number_input('Average Amount User', 0.0, 50000.0, 500.0)
    shipping_dist = st.sidebar.number_input('Shipping Distance (KM)', 0.0, 20000.0, 15.0)
    
    country = st.sidebar.selectbox('Country', ['USA', 'Germany', 'Turkey', 'Berlin', 'New York', 'London'])
    bin_country = st.sidebar.selectbox('BIN Country', ['USA', 'Germany', 'Turkey', 'UK'])
    channel = st.sidebar.selectbox('Channel', ['Web', 'App'])
    merchant = st.sidebar.selectbox('Merchant Category', ['Electronics', 'Fashion', 'Food', 'Travel', 'Health'])
    
    promo = st.sidebar.checkbox('Promo Used?')
    avs = st.sidebar.checkbox('AVS Match?')
    cvv = st.sidebar.checkbox('CVV Result Correct?')
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
        'bin_country': bin_country,
        'channel': channel,
        'merchant_category': merchant,
        'promo_used': 1 if promo else 0,
        'avs_match': 1 if avs else 0,
        'cvv_result': 1 if cvv else 0,
        'three_ds_flag': 1 if three_ds else 0,
        'shipping_distance_km': shipping_dist,
        'month': dt.month,
        'day': dt.weekday(),
        'hour': dt.hour
    }
    return pd.DataFrame([data])

user_data_df = get_user_input()

# ---------------------------------------------------------
# HALAMAN UTAMA
# ---------------------------------------------------------
st.title("🛡️ E-commerce Fraud Detection")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("💰 Business Cost Logic")
    cost_fn = st.number_input("Cost of False Negative ($)", value=650)
    cost_fp = st.number_input("Cost of False Positive ($)", value=30)

    threshold_tmp = threshold_table.copy()
    threshold_tmp["total_cost"] = (threshold_tmp["FN"] * cost_fn) + (threshold_tmp["FP"] * cost_fp)
    best_threshold = threshold_tmp.loc[threshold_tmp["total_cost"].idxmin()]["threshold_model"]
    
    st.metric("Optimal Threshold", round(best_threshold, 3))

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

        fig = px.pie(values=[prob, 1-prob], names=['Risk', 'Safe'], hole=0.7,
                     color_discrete_sequence=['#e74c3c', '#2ecc71'])
        st.plotly_chart(fig, use_container_width=True)

# Grafik Biaya di bagian bawah
st.markdown("---")
fig_cost = px.line(threshold_table, x='threshold_model', y='total_cost', 
                  title='Analisis Biaya (Semakin Rendah Semakin Baik)')
fig_cost.add_vline(x=best_threshold, line_dash="dash", line_color="red")
st.plotly_chart(fig_cost, use_container_width=True)