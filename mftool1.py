import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from mftool import Mftool
from datetime import datetime
import plotly.graph_objs as go
from scipy.stats import linregress

# Initialize Mftool for fetching mutual fund data
mf = Mftool()

RISK_FREE_RATE = 6.7  # Constant Risk-Free Rate

# Function to fetch mutual fund NAV data and scheme name
def fetch_nav_data_and_name(scheme_code, start_date='2015-01-01'):
    nav_data = mf.get_scheme_historical_nav(scheme_code, start_date, datetime.now().strftime('%Y-%m-%d'))
    scheme_name = mf.get_scheme_details(scheme_code).get('scheme_name', 'Unknown Scheme Name')
    if nav_data and 'data' in nav_data:
        nav_df = pd.DataFrame(nav_data['data'])
        nav_df['date'] = pd.to_datetime(nav_df['date'], format='%d-%m-%Y')
        nav_df['nav'] = pd.to_numeric(nav_df['nav'], errors='coerce')
        nav_df = nav_df.sort_values('date').reset_index(drop=True)
        return nav_df, scheme_name
    else:
        return pd.DataFrame(), scheme_name

# Function to fetch benchmark data from Yahoo Finance
def fetch_benchmark_data(ticker, start_date='2015-01-01'):
    benchmark_data = yf.download(ticker, start=start_date)
    benchmark_data.reset_index(inplace=True)
    return benchmark_data

# Modified function to calculate annualized returns for mutual fund and benchmark
def calculate_annualized_returns(df, column_name='nav'):
    def calculate_returns(latest_value, series, n):
        if n == 250:  # For the first year
            return (((latest_value / series.shift(n)) - 1) * 100)
        else:  # For periods greater than 1 year
            return (((latest_value / series.shift(n)) ** (1 / (n / 250))) - 1) * 100

    periods = {'1 Year': 250, '2 Years': 500, '3 Years': 750, '5 Years': 1250, '10 Years': 2500}
    results = {}
    for period, days in periods.items():
        period_return = calculate_returns(df[column_name].iloc[-1], df[column_name], days).iloc[-1]
        results[period + ' Return'] = period_return
    return pd.DataFrame([results])

# Function to calculate financial metrics
def calculate_financial_metrics(returns_df, benchmark_returns_df):
    # Convert percentages to decimals for calculations
    annual_returns = returns_df.iloc[0].values / 100
    benchmark_annual_returns = benchmark_returns_df.iloc[0].values / 100

    # Risk-Free Rate (as a decimal for calculations)
    risk_free_rate_decimal = RISK_FREE_RATE / 100

    # Calculate average return, benchmark return, average risk, downside risk, Sharpe Ratio, and Alpha
    average_return = np.mean(annual_returns)
    benchmark_return = np.mean(benchmark_annual_returns)
    average_risk = np.std(annual_returns)
    negative_returns = annual_returns[annual_returns < 0]
    downside_risk = np.std(negative_returns) if len(negative_returns) > 0 else 0
    sharpe_ratio = (average_return - risk_free_rate_decimal) / average_risk if average_risk != 0 else np.nan
    slope, intercept, _, _, _ = linregress(benchmark_annual_returns, annual_returns)
    alpha = average_return - (benchmark_return * slope)

    metrics = {
        'Average Return (%)': average_return * 100,
        'Benchmark Return (%)': benchmark_return * 100,
        'Average Risk': average_risk,
        'Downside Risk': downside_risk,
        'Sharpe Ratio': sharpe_ratio,
        'Alpha': alpha
    }

    return pd.DataFrame([metrics])


import streamlit as st

# Page setup and navigation
st.sidebar.title("Navigation")
page = st.sidebar.select_slider("Go to", ("Home", "Mutual Fund Analysis","Scheme Codes"))
# Define schemes_df in a broader scope to ensure availability
try:
    schemes_df = pd.read_csv("scheme_codes.csv")
except FileNotFoundError:
    schemes_df = pd.DataFrame()  # In case the file is not found, define an empty DataFrame

if page == "Home":
    st.title("Welcome to the Mutual Fund Analysis Dashboard")

    # Home page content
    st.header("Features")
    st.markdown("""
    - **Comprehensive Data**: Access historical NAV data, benchmark comparisons, and more.
    - **Downloadable Reports**: Easily download data for offline analysis.
    - **Weighted Returns**: Evaluate performance with weighted returns calculations.
    - **Financial Ratios**: Gain insights with calculated financial ratios like Sharpe Ratio, Alpha, and more.
    - **Easy Comparison**: Compare multiple mutual funds simultaneously to make informed decisions.
    """)

    st.header("Benefits")
    st.markdown("""
    - **Make Informed Investment Decisions**: Leverage detailed analysis and comparisons.
    - **Understand Market Trends**: Historical data and benchmark comparisons provide market context.
    - **Assess Risk and Return**: Financial ratios help in assessing risk and potential return.
    - **Customizable Analysis**: Tailor the analysis to your specific investment goals.
    """)
elif page == "Scheme Codes":
    if not schemes_df.empty:
        st.title("Mutual Fund Scheme Codes")

        # Button to download the mutual fund schemes CSV
        st.download_button(
            label="Download Mutual Fund Scheme Codes",
            data=schemes_df.to_csv(index=False).encode('utf-8'),  # Ensure proper encoding for download
            file_name='scheme_codes.csv',
            mime='text/csv',
        )

        # Display the entire DataFrame
        st.write("List of Mutual Fund Scheme Codes:")
        st.dataframe(schemes_df)
    else:
        st.error("Scheme codes data not available.")

elif page == "Mutual Fund Analysis":
    # Insert all your mutual fund analysis code here
    st.title("Mutual Fund Analysis Dashboard")
    # Your mutual fund analysis dashboard code...

    # UI layout and inputs
    st.sidebar.title("Settings")
    mode = st.sidebar.select_slider("Mode", ("Analysis Mode", "Comparison Mode"))
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker (Yahoo Finance)", "SPY")

    fund_codes = []
    if mode == "Analysis Mode":
        fund_code = st.sidebar.text_input("Enter Mutual Fund Code", "100033")
        fund_codes = [fund_code]
    else:
        st.sidebar.subheader("Comparison Mode")
        fund_codes_input = st.sidebar.text_area("Enter Mutual Fund Codes (comma-separated)", "100033,102885")
        fund_codes = [code.strip() for code in fund_codes_input.split(",")]

    #st.title("Mutual Fund Analysis Dashboard")
    benchmark_df = fetch_benchmark_data(benchmark_ticker)

    for scheme_code in fund_codes:
        nav_df, scheme_name = fetch_nav_data_and_name(scheme_code)
        if not nav_df.empty:
            st.markdown(f"### {scheme_name} (Code: {scheme_code})")

            # Calculations and display logic
            annualized_returns_df = calculate_annualized_returns(nav_df, 'nav')
            benchmark_annualized_returns_df = calculate_annualized_returns(benchmark_df, 'Adj Close')
            financial_metrics_df = calculate_financial_metrics(annualized_returns_df, benchmark_annualized_returns_df)

            # Plotting and displaying information
            expander = st.expander(f"Show NAV Data and Financial Metrics for {scheme_name}")
            with expander:
                st.dataframe(nav_df[['date', 'nav']])
                st.table(annualized_returns_df)
                st.table(financial_metrics_df)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav_df['date'], y=nav_df['nav'], mode='lines', name='NAV'))
            fig.add_trace(go.Scatter(x=benchmark_df['Date'], y=benchmark_df['Adj Close'], mode='lines', name=benchmark_ticker))
            fig.update_layout(title=f'NAV vs. Benchmark ({benchmark_ticker}) Performance', xaxis_title='Date', yaxis_title='Value', legend_title='Legend', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Data not available for scheme code {scheme_code}.")
