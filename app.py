import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------- LOAD DATA ----------------
df = pd.read_csv("ncr_ride_bookings.csv")
df.columns = df.columns.str.strip()

# ---------------- TARGET ----------------
df['is_cancelled'] = df['Booking Status'].apply(
    lambda x: 0 if x == 'Completed' else 1
)

# ---------------- FEATURES ----------------
X = df.select_dtypes(include=['number'])
X = X.drop(columns=['Cancelled Rides by Driver'], errors='ignore')
X = X.loc[:, X.nunique() > 1]

y = df['is_cancelled']

# ---------------- MODEL ----------------
model = RandomForestClassifier()
model.fit(X, y)

# ---------------- PAGE ----------------
st.set_page_config(page_title="Ride Prediction", layout="wide")

st.title("🚗 Ride Cancellation Prediction System")
st.markdown("### Smart Prediction Dashboard")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🎛️ Enter Ride Details")

inputs = []

def safe_slider(col):
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())

    if min_val == max_val:
        st.sidebar.write(f"{col}: {min_val}")
        return min_val
    else:
        return st.sidebar.slider(col, min_val, max_val, mean_val)

# Main features
for col in ['Avg VTAT', 'Avg CTAT', 'Booking Value',
            'Ride Distance', 'Driver Ratings', 'Customer Rating']:
    if col in X.columns:
        inputs.append(safe_slider(col))

# Other features
for col in X.columns:
    if col not in ['Avg VTAT','Avg CTAT','Booking Value',
                   'Ride Distance','Driver Ratings','Customer Rating']:
        inputs.append(safe_slider(col))

# ---------------- MAIN RIGHT SIDE ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Rides", f"{len(df):,}")
col2.metric("Cancellation Rate", f"{df['is_cancelled'].mean()*100:.2f}%")
col3.metric("Avg Booking Value", f"{df['Booking Value'].mean():.2f}")

st.markdown("---")

# Button on RIGHT SIDE
if st.button("🚀 Predict Cancellation"):

    input_array = np.array(inputs).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    proba = model.predict_proba(input_array)
    probability = proba[0][1] if proba.shape[1] > 1 else proba[0][0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"🚨 High Cancellation Risk: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low Cancellation Risk: {probability*100:.2f}%")