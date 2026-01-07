import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
df = pd.read_csv("house_data.csv")
X = df[["size", "bedrooms"]]
y = df["price"]
model = LinearRegression()
model.fit(X, y)
with open("model/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)
import streamlit as st
import pandas as pd
import pickle
# Load model
with open("model/linear_model.pkl", "rb") as f:
    model = pickle.load(f)
st.title("🏠 House Price Prediction App")
st.write("Prediksi harga rumah menggunakan **Linear Regression**")
# Sidebar input
st.sidebar.header("Input Rumah")
size = st.sidebar.slider("Luas Rumah (m²)", 30, 250, 100)
bedrooms = st.sidebar.slider("Jumlah Kamar Tidur", 1, 6, 3)
# Create input dataframe
input_data = pd.DataFrame({
    "size": [size],
    "bedrooms": [bedrooms]
})
st.subheader("Data Input")
st.dataframe(input_data)
# Prediction
prediction = model.predict(input_data)[0]
st.subheader("💰 Estimasi Harga Rumah")
st.metric(
    label="Harga (Juta Rupiah)",
    value=f"{prediction:,.0f}"
)
# Explanation
st.info(
    "Model ini menggunakan Linear Regression sederhana "
    "berdasarkan luas rumah dan jumlah kamar."
)
