# ============================
# AQI LSTM-GRU + Streamlit
# ============================

# Install dependencies (uncomment if running fresh)
# !pip install lime tensorflow scikit-learn pandas requests matplotlib streamlit imbalanced-learn --quiet

# ============================
# Imports
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from lime import lime_tabular

# ============================
# Scrape AQI Data (Tamil Nadu Cities)
# ============================
cities = {
    "Dindigul": "https://www.aqi.in/dashboard/india/tamil-nadu/dindigul",
    "Madurai": "https://www.aqi.in/dashboard/india/tamil-nadu/madurai",
    "Karur": "https://www.aqi.in/dashboard/india/tamil-nadu/karur",
    "Salem": "https://www.aqi.in/dashboard/india/tamil-nadu/salem",
    "Erode": "https://www.aqi.in/dashboard/india/tamil-nadu/erode",
    "Coimbatore": "https://www.aqi.in/dashboard/india/tamil-nadu/coimbatore",
    "Tiruppur": "https://www.aqi.in/dashboard/india/tamil-nadu/tiruppur",
    "Tuticorin": "https://www.aqi.in/dashboard/india/tamil-nadu/tuticorin",
    "Ooty": "https://www.aqi.in/dashboard/india/tamil-nadu/ooty",
    "Chennai": "https://www.aqi.in/dashboard/india/tamil-nadu/chennai"
    # Add remaining cities as before
}

all_data = []
for city, url in cities.items():
    try:
        response = requests.get(url, timeout=10)
        tables = pd.read_html(response.text)
        df_city = tables[0]
        df_city.columns = ["Time","AQI"]
        df_city["City"] = city
        all_data.append(df_city)
    except:
        print(f"Skipped {city}")

if len(all_data) == 0:
    st.error("No AQI data could be scraped. Check URLs or internet connection.")
    st.stop()

df_all = pd.concat(all_data, ignore_index=True)
df_all["Time"] = pd.to_datetime(df_all["Time"], errors='coerce')
df_all["AQI"] = pd.to_numeric(df_all["AQI"], errors='coerce')
df_all = df_all.dropna()
df_all = df_all[df_all["AQI"] > 0].sort_values(["City","Time"])

# ============================
# Feature Engineering
# ============================
np.random.seed(42)
df_all["PM2.5"] = df_all["AQI"]*np.random.uniform(0.4,0.6,len(df_all))
df_all["PM10"]  = df_all["AQI"]*np.random.uniform(0.6,0.9,len(df_all))
df_all["NO2"]   = df_all["AQI"]*np.random.uniform(0.2,0.4,len(df_all))
df_all["SO2"]   = df_all["AQI"]*np.random.uniform(0.1,0.3,len(df_all))
df_all["O3"]    = df_all["AQI"]*np.random.uniform(0.2,0.5,len(df_all))
df_all["CO"]    = df_all["AQI"]*np.random.uniform(0.01,0.05,len(df_all))
df_all["temperature"] = 25 + df_all["AQI"]*0.02 + np.random.normal(0,1,len(df_all))
df_all["humidity"] = 50 + df_all["AQI"]*0.1 + np.random.normal(0,5,len(df_all))
df_all["wind_speed"] = np.random.uniform(0.5,5,len(df_all))
df_all["traffic"] = (df_all["AQI"] // 50).clip(1,5)

df_all["AQI_Category"] = pd.cut(df_all["AQI"], bins=[0,50,100,300], labels=["Good","Moderate","Unhealthy"])
le = LabelEncoder()
df_all["AQI_Label"] = le.fit_transform(df_all["AQI_Category"])

features = ['PM2.5','PM10','SO2','O3','NO2','CO','temperature','humidity','wind_speed','traffic']
scaler = MinMaxScaler()
df_all[features] = scaler.fit_transform(df_all[features])

# ============================
# Sequence Creation
# ============================
def create_sequences(df, time_steps=5):
    X, y = [], []
    for city in df["City"].unique():
        city_df = df[df["City"]==city]
        data = city_df[features].values
        labels = city_df["AQI_Label"].values
        for i in range(len(data)-time_steps):
            X.append(data[i:i+time_steps])
            y.append(labels[i+time_steps])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(df_all)

# Train/test split
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
y_train = to_categorical(y_train_raw)
y_test = to_categorical(y_test_raw)

# ============================
# LSTM + GRU Model
# ============================
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=8, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# ============================
# Streamlit UI
# ============================
st.title("🌫️ Tamil Nadu AQI Prediction")
st.markdown("Predict AQI levels using LSTM-GRU deep learning model")

city = st.selectbox("Select a City:", df_all["City"].unique())

st.markdown("### Enter Environmental Parameters")
def user_input():
    PM25 = st.slider("PM2.5", 5, 300, 50)
    PM10 = st.slider("PM10", 10, 500, 80)
    NO2  = st.slider("NO2", 1, 150, 20)
    SO2  = st.slider("SO2", 0.1, 50, 10)
    O3   = st.slider("O3", 5, 100, 20)
    CO   = st.slider("CO", 0.01, 5.0, 0.5)
    temperature = st.slider("Temperature (°C)", 15, 45, 30)
    humidity    = st.slider("Humidity (%)", 20, 100, 60)
    wind_speed  = st.slider("Wind Speed (m/s)", 0.1, 10.0, 2.0)
    traffic     = st.slider("Traffic Level (1-5)", 1, 5, 3)
    return pd.DataFrame({
        "PM2.5":[PM25], "PM10":[PM10], "SO2":[SO2], "O3":[O3],
        "NO2":[NO2], "CO":[CO], "temperature":[temperature],
        "humidity":[humidity], "wind_speed":[wind_speed], "traffic":[traffic]
    })

input_df = user_input()
input_scaled = scaler.transform(input_df)
X_input = input_scaled.reshape(1, X_train.shape[1], X_train.shape[2])
pred = model.predict(X_input)
pred_index = np.argmax(pred, axis=1)[0]
pred_category = le.inverse_transform([pred_index])[0]
pred_prob = pred[0][pred_index]

st.markdown("### Predicted AQI Level")
st.write(f"**{pred_category}** with probability {pred_prob:.2f}")

st.markdown("### Input Features")
st.dataframe(input_df.T.rename(columns={0:"Value"}))
