# import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -------- Sidebar --------
st.sidebar.header("Determine values ")
st.sidebar.image("stock_img.png")

st.sidebar.write("")
st.sidebar.markdown("Made by [Mohamed Ahmed Elfikey](https://www.linkedin.com/in/mohamed-elfikey/)")
st.sidebar.write("")
stock_name = st.sidebar.text_input("Enter stock name:")

# ----------- code related to model ----------------
if stock_name:  # Check if user has entered a stock name
    # Scrape and Read data
    stock = yf.Ticker(stock_name)
    stock = stock.history(start="2022-01-01", end="2024-01-01")

    last_row_values = stock.iloc[-1, [0, 1, 2, 4]].values.reshape(1, -1)  # Reshape for prediction

    # load model
    model_path = r"C:\Users\ELFEKY\Desktop\projects\ML\Dashboard of stock price predication\linear_regression_model.pkl"
    loaded_model = joblib.load(model_path)
    last_row_values = stock.iloc[-1, [0, 1, 2, 4]].values.reshape(1, -1)  # Reshape for prediction

    # Make a prediction
    predication = loaded_model.predict(last_row_values)

    # Display a header
    st.write("""  # Stock Price Prediction
       We built an app to predict the closing stock price.
       """)
    st.write(f" ##  For {stock_name}")
    st.write(f"## {stock_name} will close at {predication[0]:.2f}")

    # Metrics
    a1, a2, a3 = st.columns(3)
    a1.metric("Max Close Price", f"{stock.Close.max():.2f}")
    a2.metric("Max Open Price", f"{stock.Open.max():.2f}")
    a3.metric("Max Volume", f"{stock.Volume.max():.2f}")

    # Display DataFrame
    st.dataframe(stock.head())

    # Visualizations
    st.write(""" ### Close Price Over Time """)
    st.line_chart(stock.Close)

    st.write(""" ### Open Price Over Time """)
    st.line_chart(stock.Open)

    st.write(""" ### Volume of Stock Over Time """)
    st.line_chart(stock.Volume)

else:
    st.write(" ")
