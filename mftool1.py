import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from mftool import Mftool
from datetime import datetime

# Initialize Mftool for fetching mutual fund data
mf = Mftool()


# Function to fetch mutual fund NAV data and scheme name
def fetch_nav_data_and_name(scheme_code, start_date='2015-01-01'):
    nav_data = mf.get_scheme_historical_nav(scheme_code, start_date, datetime.now().strftime('%d-%m-%Y'))
    scheme_name = mf.get_scheme_details(scheme_code).get('scheme_name', 'Unknown Scheme Name')
    if nav_data and 'data' in nav_data:
        nav_df = pd.DataFrame(nav_data['data'])
        nav_df['date'] = pd.to_datetime(nav_df['date'], format='%d-%m-%Y')
        nav_df['nav'] = pd.to_numeric(nav_df['nav'], errors='coerce')
        nav_df = nav_df.sort_values('date').reset_index(drop=True)
        nav_df['return'] = nav_df['nav'].pct_change() * 100  # Calculate daily returns
        return nav_df, scheme_name
    else:
        return pd.DataFrame(), scheme_name


# Function to fetch benchmark data from Yahoo Finance
def fetch_benchmark_data(ticker, start_date='2015-01-01'):
    benchmark_data = yf.download(ticker, start=start_date)
    benchmark_data.reset_index(inplace=True)
    return benchmark_data


# Function to calculate returns for specific periods
def calculate_period_returns(nav_df):
    periods = [1, 2, 3, 5, 10]  # Years
    results = {}
    for period in periods:
        # Calculate returns for the period, assuming 250 trading days in a year
        period_return = nav_df['nav'].pct_change(periods=period * 250).dropna().iloc[-1] * 100
        results[f'{period} Year Return'] = period_return
    return results


# UI layout and inputs
st.sidebar.title("Settings")
mode = st.sidebar.radio("Mode", ("Analysis Mode", "Comparison Mode"))
benchmark_ticker = st.sidebar.text_input("Benchmark Ticker (Yahoo Finance)", "SPY")

if mode == "Analysis Mode":
    fund_code = st.sidebar.text_input("Enter Mutual Fund Code", "100033")
    fund_codes = [fund_code]
else:
    st.sidebar.subheader("Comparison Mode")
    fund_codes_input = st.sidebar.text_area("Enter Mutual Fund Codes (comma-separated)", "100033,102885")
    fund_codes = [code.strip() for code in fund_codes_input.split(",")]

# Main panel for displaying analysis results
st.title("Mutual Fund Analysis Dashboard")

# Fetch benchmark data
benchmark_df = fetch_benchmark_data(benchmark_ticker)

if mode == "Analysis Mode" and fund_codes[0]:
    fund_codes = [fund_codes[0]]  # Ensure only one fund is processed in analysis mode

for scheme_code in fund_codes:
    nav_df, scheme_name = fetch_nav_data_and_name(scheme_code)
    if not nav_df.empty:
        st.markdown(f"### {scheme_name} (Code: {scheme_code})")
        period_returns = calculate_period_returns(nav_df)

        # Display period returns
        for period, value in period_returns.items():
            st.markdown(f"**{period}:** {value:.2f}%")

        # Plot NAV data and benchmark using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nav_df['date'], y=nav_df['nav'], mode='lines', name='NAV'))
        fig.add_trace(
            go.Scatter(x=benchmark_df['Date'], y=benchmark_df['Adj Close'], mode='lines', name=benchmark_ticker))
        fig.update_layout(title=f'NAV vs. Benchmark ({benchmark_ticker}) Performance', xaxis_title='Date',
                          yaxis_title='Value', legend_title='Legend', height=400)
        st.plotly_chart(fig, use_container_width=True)

        expander = st.expander("Show NAV Data")
        with expander:
            # Display NAV data alongside returns
            st.dataframe(nav_df[['date', 'nav', 'return']].tail())

    else:
        st.error(f"Data not available for scheme code {scheme_code}.")
