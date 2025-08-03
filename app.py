import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Tiêu đề
st.title("📊 Diamond Sales Data Analysis")

# Đọc dữ liệu
df = pd.read_csv("diamonds.csv")
st.subheader("1. Dữ liệu gốc")
st.write(df.head())

# Biểu đồ phân tích
st.subheader("2. Biểu đồ giá theo carat")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='carat', y='price', ax=ax)
st.pyplot(fig)

# Huấn luyện mô hình đơn giản
st.subheader("3. Mô hình dự đoán giá")
X = df[['carat']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
st.write(f"🔍 MAE: {mae:.2f}")
