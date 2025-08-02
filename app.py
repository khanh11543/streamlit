import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import numpy as np

st.title("📊 Sales Forecasting for ABC Manufacturing")
st.markdown("### A Data Science Solution with Visualizations and Prediction")

# Load data
df = pd.read_csv("data/sales_data.csv")

# Hiển thị dữ liệu
st.subheader("1️⃣ Raw Data")
st.dataframe(df.head())

# --- Step 1: Data Preprocessing ---
st.subheader("2️⃣ Data Preprocessing")

# Missing value handling
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month')

# Scaling
scaler = MinMaxScaler()
cols_to_scale = ['Sales', 'Defects', 'MaintenanceCost']
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

st.code("Applied: Missing value imputation, datetime conversion, MinMax scaling")

# --- Step 2: Visualization ---
st.subheader("3️⃣ Visualizations")

fig1, ax1 = plt.subplots()
sns.lineplot(data=df, x='Month', y='Sales', ax=ax1)
plt.title("📈 Total Sales Over Time")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.barplot(data=df, x='Product', y='Sales', ax=ax2)
plt.title("🛒 Sales by Product")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
sns.lineplot(data=df, x='Month', y='Defects', label='Defects', ax=ax3)
sns.lineplot(data=df, x='Month', y='MaintenanceCost', label='Maintenance Cost', ax=ax3)
plt.title("⚙️ Defects and Maintenance")
plt.legend()
st.pyplot(fig3)

fig4, ax4 = plt.subplots()
sns.lineplot(data=df, x='Month', y='Satisfaction', ax=ax4)
plt.title("😊 Customer Satisfaction")
st.pyplot(fig4)

# --- Step 3: Forecast with Linear Regression ---
st.subheader("4️⃣ Sales Forecast (Linear Regression)")

df_model = df.copy()
df_model['Month_num'] = df_model['Month'].dt.month + (df_model['Month'].dt.year - df_model['Month'].dt.year.min()) * 12

X = df_model[['Month_num']]
y = df_model['Sales']

model = LinearRegression()
model.fit(X, y)

# Predict next month
next_month = X['Month_num'].max() + 1
predicted_sales = model.predict([[next_month]])
r2 = r2_score(y, model.predict(X))

st.markdown(f"📅 **Forecast for next month:** `{round(predicted_sales[0])} units`")
st.markdown(f"🎯 **Model R² Score:** `{round(r2, 3)}`")

# Plot regression
fig5, ax5 = plt.subplots()
sns.lineplot(x=X['Month_num'], y=y, label='Actual Sales', ax=ax5)
sns.lineplot(x=X['Month_num'], y=model.predict(X), label='Regression Line', ax=ax5)
plt.scatter(next_month, predicted_sales, color='red', label='Forecast')
plt.title("🔮 Linear Regression Forecast")
plt.legend()
st.pyplot(fig5)
