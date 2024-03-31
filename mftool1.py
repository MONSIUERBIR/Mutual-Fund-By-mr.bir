import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from mftool import Mftool
from datetime import datetime
import plotly.graph_objs as go
from scipy.stats import linregress
import babel.numbers

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


def fetch_benchmark_data(ticker, start_date, end_date=datetime.now().strftime('%Y-%m-%d')):
    benchmark_data = yf.download(ticker, start=start_date, end=end_date)
    benchmark_data.reset_index(inplace=True)
    benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'])
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


# Fetch scheme codes data
try:
    schemes_df = pd.read_csv("scheme_codes.csv")
except FileNotFoundError:
    schemes_df = pd.DataFrame()  # In case the file is not found, define an empty DataFrame

# Set page config
st.set_page_config('Mutual Fund Calculator', page_icon=':bar_chart:', layout="wide")
# st.write('Phone users: swipe on tabs to see all options.')

# Define tabs
tabs = st.tabs(tabs=['Home', 'Mutual Fund Analysis', 'Scheme Codes', 'MF Guide'])


# Detailed summary
def calculate_returns1(latest_nav, nav_series, n):
    return (((latest_nav / nav_series.shift(n))) - 1) * 100


def calculate_returns(latest_nav, nav_series, n):
    return (((latest_nav / nav_series.shift(n)) ** (1 / n)) - 1) * 100

# Load the scheme codes and names from the uploaded CSV file
scheme_codes_df = pd.read_csv('scheme_codes.csv')

# Adjust column names based on your actual CSV structure
scheme_codes = scheme_codes_df['Scheme_Code'].tolist()  # Placeholder for 'Scheme Code' column
scheme_names = scheme_codes_df['Scheme_Name'].tolist()  # Placeholder for 'Scheme Name' column

# Map scheme names to codes for display in the selectbox
scheme_options = {name: code for name, code in zip(scheme_names, scheme_codes)}




# Home tab content
with tabs[0]:
    st.title("Welcome to the Mutual Fund Analyzer Dashboard")

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

# Mutual Fund Analysis tab content
with tabs[1]:
    st.title("Mutual Fund Analysis Dashboard")

    # Define tabs for Mutual Fund Analysis
    analysis_tabs = st.tabs(
        tabs=['Analysis Mode', 'Comparison Mode', 'SIP Calculator', 'Lumpsum Investment Calculator', 'Quick Tools',
              'Detailed Annual Returns'])

    # Analysis Mode
    with analysis_tabs[0]:  # Analysis Mode
        st.subheader("Analysis Mode")
        # Inputs for selecting a single mutual fund scheme and benchmark
        selected_scheme_name = st.selectbox(
            'Select Mutual Fund Scheme',
            options=scheme_names, # Display scheme names in the selectbox
            placeholder="Select Scheme...",
            index=4610,

        )

        # Fetch the selected scheme code based on the selected name
        selected_scheme_code = scheme_options[selected_scheme_name]

        # Display the selected scheme code
        st.write('You selected scheme code:', selected_scheme_code)
        fund_code = selected_scheme_code
        benchmark_ticker = st.text_input("Benchmark Ticker (Yahoo Finance)", "^CNXINFRA")
        st.write('Get Benchmark Ticker names from Yahoo Finance.')

        if fund_code:
            nav_df, scheme_name = fetch_nav_data_and_name(fund_code)
            if not nav_df.empty:
                # Get the oldest and latest NAV date
                oldest_nav_date = nav_df['date'].min().strftime('%Y-%m-%d')
                latest_nav_date = nav_df['date'].max().strftime('%Y-%m-%d')
                # Use these dates as the start and end date for fetching benchmark data
                benchmark_df = fetch_benchmark_data(benchmark_ticker, oldest_nav_date, latest_nav_date)

            if not nav_df.empty and not benchmark_df.empty:
                st.subheader(f"Analysis for {scheme_name} against {benchmark_ticker} benchmark")

                # Calculation
                annualized_returns_df = calculate_annualized_returns(nav_df, 'nav')
                benchmark_annualized_returns_df = calculate_annualized_returns(benchmark_df, 'Adj Close')
                financial_metrics_df = calculate_financial_metrics(annualized_returns_df,
                                                                   benchmark_annualized_returns_df)


                # Expander for detailed data and financial metrics
                with st.expander(f"Show NAV Data and Financial Metrics for {scheme_name}"):
                    st.dataframe(nav_df[['date', 'nav']])


                    # Function to determine performance indicator
                    def performance_indicator(value):
                        if value >= 10:
                            return "🟢 Excellent"
                        elif 5 <= value < 10:
                            return "🟡 Good"
                        else:
                            return "🔴 Needs Improvement"


                    # Display annualized returns with performance indicators
                    st.write("Annualized Returns:")
                    cols_returns = st.columns(len(annualized_returns_df.columns))
                    for col, metric in zip(cols_returns, annualized_returns_df.columns):
                        metric_value = annualized_returns_df[metric].values[0]
                        performance = performance_indicator(metric_value)
                        col.metric(label=metric, value=f"{metric_value:.2f}%", delta=performance)

                    # Display financial metrics with performance indicators
                    st.write("Financial Metrics:")
                    cols_metrics = st.columns(len(financial_metrics_df.columns))
                    for col, metric in zip(cols_metrics, financial_metrics_df.columns):
                        value = financial_metrics_df[metric].values[0]
                        formatted_value = f"{value:.2f}" if isinstance(value, float) else value
                        performance = performance_indicator(value)
                        col.metric(label=metric, value=formatted_value, delta=performance)

                # Calculate cumulative returns for mutual fund NAV
                nav_df['Cumulative Returns'] = (1 + nav_df['nav'].pct_change()).cumprod() - 1

                # Ensure benchmark_df covers the same dates as nav_df
                if not benchmark_df.empty:
                    benchmark_df = benchmark_df.set_index('Date').reindex(nav_df['date']).interpolate(
                        method='time').reset_index()

                # Calculate cumulative returns for both NAV and benchmark
                nav_df['Cumulative Returns'] = (1 + nav_df['nav'].pct_change()).cumprod() - 1
                benchmark_df['Cumulative Returns'] = (1 + benchmark_df['Adj Close'].pct_change()).cumprod() - 1

                # Prepare the final DataFrame for plotting
                plot_df = pd.DataFrame({
                    'Date': nav_df['date'],
                    f'{scheme_name} Cumulative Returns': nav_df['Cumulative Returns'],
                    f'{benchmark_ticker} Cumulative Returns': benchmark_df['Cumulative Returns']
                }).set_index('Date')

                # Using Streamlit to plot the line chart
                st.line_chart(plot_df,color=["#52DDED", "#ED7052"])

    with analysis_tabs[1]:  # Comparison Mode
        st.subheader("Comparison Mode")

        # Inputs for selecting multiple mutual fund schemes and a benchmark
        selected_scheme_names = st.multiselect(
            'Select Mutual Fund Schemes',
            options=scheme_names,  # Display scheme names in the multiselect
            default=[scheme_names[4610]] if scheme_names else []  # Default selection
        )

        # Fetch the selected scheme codes based on the selected names and ensure they are strings
        selected_scheme_codes = [str(scheme_options[name]) for name in selected_scheme_names]

        # Show selected scheme codes for confirmation
        st.write('You selected scheme codes:', ', '.join(selected_scheme_codes))

        benchmark_ticker_comparison = st.text_input("Benchmark Ticker for Comparison (Yahoo Finance)", "^CNXINFRA")
        st.write('Get Benchmark Ticker names from Yahoo Finance.')

        if selected_scheme_codes and benchmark_ticker_comparison:
            # Assume the first selected scheme for benchmark comparison
            if selected_scheme_codes:  # Check if any scheme is selected
                first_scheme_code = selected_scheme_codes[0]
                first_nav_df, first_scheme_name = fetch_nav_data_and_name(first_scheme_code)
                if not first_nav_df.empty:
                    # Get the oldest and latest NAV date for the first selected scheme
                    oldest_nav_date = first_nav_df['date'].min().strftime('%Y-%m-%d')
                    latest_nav_date = first_nav_df['date'].max().strftime('%Y-%m-%d')

                    # Fetch benchmark data for the same date range
                    benchmark_df_comparison = fetch_benchmark_data(benchmark_ticker_comparison, oldest_nav_date,
                                                                   latest_nav_date)

                    # Calculate cumulative returns for benchmark
                    benchmark_df_comparison['Cumulative Returns'] = (1 + benchmark_df_comparison[
                        'Adj Close'].pct_change()).cumprod() - 1

                fig = go.Figure()

                # Plot each selected mutual fund scheme
                for scheme_code in selected_scheme_codes:
                    nav_df, scheme_name = fetch_nav_data_and_name(scheme_code)
                    if not nav_df.empty:
                        # Calculate cumulative returns for the mutual fund NAV
                        nav_df['Cumulative Returns'] = (1 + nav_df['nav'].pct_change()).cumprod() - 1
                        fig.add_trace(go.Scatter(x=nav_df['date'], y=nav_df['Cumulative Returns'], mode='lines',
                                                 name=scheme_name))

                # Plot the benchmark comparison
                if not benchmark_df_comparison.empty:
                    fig.add_trace(
                        go.Scatter(x=benchmark_df_comparison['Date'], y=benchmark_df_comparison['Cumulative Returns'],
                                   mode='lines', name='Benchmark'))

                # Update plot layout
                fig.update_layout(
                    width=1300,
                    title='Cumulative Returns: Mutual Funds vs. Benchmark',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Returns',
                    legend_title='Series'
                )

                # Set custom colors if needed
                colors = ['52DDED', 'EDA752', '6EED52', 'ED7052', '5281ED']  # Example colors
                for i, trace in enumerate(fig.data):
                    trace.line.color = colors[i % len(colors)]

                st.plotly_chart(fig)
            else:
                st.error("Please select at least one mutual fund scheme for comparison.")

    # Calculator Pages
    with analysis_tabs[2]:
        st.subheader("SIP Calculator")
        sip_amount = st.number_input('SIP Amount', 100, 100000, 1000)
        # rate_of_return = st.number_input('Expected Rate of Return (in %)',0.0,100.0,12.0,0.01)
        rate_of_return = st.slider('Expected Rate of Return (in %)', 1, 30, 12)
        duration = st.number_input('Duration of Investment (in years)', 1, 100, 10)

        st.markdown('##')
        # checkbox for adjusting inflation
        checkbox = st.checkbox('Adjust SIP for Inflation ? (Assumed annual inflation rate is 6%)', False)

        # if inflation checkbox if off
        if checkbox == False:

            monthly_rate = rate_of_return / 12 / 100
            months = duration * 12

            invested_value = sip_amount * months
            invested_value_inwords = babel.numbers.format_currency(invested_value, 'INR', locale='en_IN')

            future_value = sip_amount * ((((1 + monthly_rate) ** (months)) - 1) * (1 + monthly_rate)) / monthly_rate
            future_value_inwords = babel.numbers.format_currency(future_value, 'INR', locale='en_IN')

            gain = round(float(future_value) - float(invested_value), 2)
            gain_inwords = babel.numbers.format_currency(gain, 'INR', locale='en_IN')

            st.subheader(f'Amount Invested: {invested_value_inwords}')
            st.subheader(f'Final Amount: {future_value_inwords}')
            st.subheader(f'Gain: {gain_inwords}')

            # plot pie chart
            fig = go.Figure(data=[go.Pie(labels=['Investment', 'Gain'], values=[invested_value, gain])])
            fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15,
                              marker=dict(colors=['52ED5D', 'ED7052'], line=dict(color='#000000', width=2)))
            st.plotly_chart(fig)

        elif checkbox == True:

            try:
                monthly_rate = (rate_of_return - 6) / 12 / 100
                months = duration * 12

                invested_value = sip_amount * months
                invested_value_inwords = babel.numbers.format_currency(sip_amount * months, 'INR', locale='en_IN')

                future_value = sip_amount * ((((1 + monthly_rate) ** (months)) - 1) * (1 + monthly_rate)) / monthly_rate
                future_value_inwords = babel.numbers.format_currency(future_value, 'INR', locale='en_IN')

                gain_after_inflation = round(float(future_value) - float(invested_value), 2)
                gain_after_inflation_inwords = babel.numbers.format_currency(gain_after_inflation, 'INR',
                                                                             locale='en_IN')

                st.subheader(f'Amount Invested: {invested_value_inwords}')
                st.subheader(f'Final Amount: {future_value_inwords}')
                st.subheader(f'Gain: {gain_after_inflation_inwords}')

                fig = go.Figure(
                    data=[go.Pie(labels=['Investment', 'Gain'], values=[invested_value, gain_after_inflation])])
                fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15,
                                  marker=dict(colors=['52ED5D', 'ED7052'], line=dict(color='#000000', width=2)))
                st.plotly_chart(fig)
            except Exception as e:
                st.warning('Please change the expcted rate of return')

        st.subheader('About SIP & SIP Calculator')
        st.write('''
            Systematic Investment Plan (SIP) is a kind of investment scheme offeED7052 by mutual fund companies. Using SIP one can invest small amount peridically (weekly, monthly, quaterly) into a selected mutual fund. For retail investors, SIP offers a well disciplined and passive approach to investing, to create wealth in long term (using the power of compounding). Since, the amount is invested on regular intervals (usually on monthly basis), it also ED7052uces the impact of market volatility.

        This calculator helps you calculate the wealth gain and expected returns for your monthly SIP investment.''')

    with analysis_tabs[3]:
        st.subheader("Lumpsum Investment Calculator")
        lumpsum_amount = st.number_input('Investment Amount', 100, 9999999999, 1000)
        lumpsum_amount_inwords = babel.numbers.format_currency(lumpsum_amount, 'INR', locale='en_IN')

        # lumpsum_rate_of_return = st.number_input('Expected Rate of Return (in %)',1.00,100.0,12.0,0.01)
        lumpsum_rate_of_return = st.slider('Expected Rate of Return (in %) ', 1, 30, 12)
        lumpsum_duration = st.number_input('Duration of Investment (in years)', 1, 99, 10)

        cagr = lumpsum_amount * (pow((1 + lumpsum_rate_of_return / 100), lumpsum_duration))
        cagr_inwords = babel.numbers.format_currency(cagr, 'INR', locale='en_IN')

        st.markdown('##')
        lumpsum_checkbox = st.checkbox('Adjust Investment for Inflation ? (Assumed annual inflation rate is 6%)', False)

        if lumpsum_checkbox == False:

            lumpsum_gain = round(float(cagr) - float(lumpsum_amount), 2)
            lumpsum_gain_inwords = babel.numbers.format_currency(lumpsum_gain, 'INR', locale='en_IN')

            st.subheader(f'Amount Invested: {lumpsum_amount_inwords}')
            st.subheader(f'Final Amount: {cagr_inwords}')
            st.subheader(f'Gain: {lumpsum_gain_inwords}')

            # plot pie chart
            fig = go.Figure(data=[go.Pie(labels=['Investment', 'Gain'], values=[lumpsum_amount, lumpsum_gain])])
            fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15,
                              marker=dict(colors=['52ED5D', 'ED7052'], line=dict(color='#000000', width=2)))
            st.plotly_chart(fig)

        elif lumpsum_checkbox == True:
            cagr_after_inflation = lumpsum_amount * (pow((1 + (lumpsum_rate_of_return - 6) / 100), lumpsum_duration))
            cagr_after_inflation_inwords = babel.numbers.format_currency(cagr_after_inflation, 'INR', locale='en_IN')

            lumpsum_gain_after_inflation = round(float(cagr_after_inflation) - float(lumpsum_amount), 2)
            lumpsum_gain_after_inflation_inwords = babel.numbers.format_currency(lumpsum_gain_after_inflation, 'INR',
                                                                                 locale='en_IN')

            st.subheader(f'Amount Invested: {lumpsum_amount_inwords}')
            st.subheader(f'Final Amount: {cagr_after_inflation_inwords}')
            st.subheader(f'Gain: {lumpsum_gain_after_inflation_inwords}')

            # plot pie chart
            fig = go.Figure(
                data=[go.Pie(labels=['Investment', 'Gain'], values=[lumpsum_amount, lumpsum_gain_after_inflation])])
            fig.update_traces(hoverinfo='value', textinfo='label+value', textfont_size=15,
                              marker=dict(colors=['52ED5D', 'ED7052'], line=dict(color='#000000', width=2)))
            st.plotly_chart(fig)

    with analysis_tabs[4]:
        st.subheader("Quick Tools")
        calc1, calc2 = st.columns(2)

        with calc1:
            st.subheader('Future Value Calculator')

            principal_amount = st.number_input('Today\'s Value', 1, 1000000000, 50000)
            after_years = st.number_input('After Years', 1, 100, 10)
            expected_increase = st.number_input('Expected Rate of Increase (in %)', 1, 20, 6)
            expected_increase = expected_increase / 100

            future_value_calc = principal_amount * (pow(1 + expected_increase, after_years))
            future_value_calc_inwords = babel.numbers.format_currency(future_value_calc, 'INR', locale='en_IN')

            st.subheader(f'Future Value: {future_value_calc_inwords}')

        with calc2:
            st.subheader('Present Value Calculator')

            principal_amount1 = st.number_input('Future\'s Value', 1, 1000000000, 100000)
            after_years1 = st.number_input('After Years ', 1, 100, 5)
            expected_decrease = st.number_input('Expected Rate of Decrease (in %)', 1, 20, 6)
            expected_decrease = expected_decrease / 100

            present_value_calc = principal_amount1 / (pow(1 + expected_decrease, after_years1))
            present_value_calc_inwords = babel.numbers.format_currency(present_value_calc, 'INR', locale='en_IN')

            st.subheader(f'Present Value: {present_value_calc_inwords}')
    with analysis_tabs[5]:
        st.subheader("Detailed Annual Returns")

        # Inputs for mutual fund scheme code and start date
        selected_scheme_name = st.selectbox(
            'Select Mutual Fund Scheme',
            options=scheme_names,  # Display scheme names in the selectbox
            index= 4610,
        )

        # Fetch the selected scheme code based on the selected name
        selected_scheme_code = scheme_options[selected_scheme_name]

        # Display the selected scheme code
        st.write('You selected scheme code:', selected_scheme_code)
        scheme_code = selected_scheme_code
        start_date = st.text_input("Enter start date (DD-MM-YYYY)", "03-04-2006")

        if scheme_code and start_date:
            try:
                start_date_obj = datetime.strptime(start_date, "%d-%m-%Y")

                # Fetch historical NAV data
                nav_data = mf.get_scheme_historical_nav(scheme_code, start_date, datetime.now().strftime('%Y-%m-%d'))

                if nav_data and 'data' in nav_data:
                    # Process and calculate returns
                    nav_df = pd.DataFrame(nav_data['data'])
                    nav_df["nav"] = pd.to_numeric(nav_df["nav"], errors="coerce")
                    nav_df["date"] = pd.to_datetime(nav_df["date"], format="%d-%m-%Y")
                    nav_df.sort_values("date", inplace=True)

                    # Calculate detailed annual returns
                    nav_df["1 Year Return"] = calculate_returns1(nav_df["nav"], nav_df["nav"], 250)
                    nav_df["2 Year Return"] = calculate_returns(nav_df["nav"], nav_df["nav"].shift(250 * 2), 2)
                    nav_df["3 Year Return"] = calculate_returns(nav_df["nav"], nav_df["nav"].shift(250 * 3), 3)
                    nav_df["5 Year Return"] = calculate_returns(nav_df["nav"], nav_df["nav"].shift(250 * 5), 5)
                    nav_df["10 Year Return"] = calculate_returns(nav_df["nav"], nav_df["nav"].shift(250 * 10), 10)

                    # Display the processed data
                    st.write(f"Showing returns for {scheme_code} starting from {start_date}:")
                    st.info(f"Detailed Summary for {scheme_name} ")
                    st.dataframe(nav_df)
                else:
                    st.error("No data found for the provided scheme code and date range.")
            except ValueError:
                st.error("Invalid date format. Please ensure the date is in DD-MM-YYYY format.")
    # Scheme Codes tab content
with tabs[2]:
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
        st.write("List of Mutual Fund Scheme Codes")
        st.dataframe(schemes_df)
    else:
        st.write("No scheme codes data available. Please ensure you have the correct file or contact support.")

# MF Guide tab content
with tabs[3]:
    st.title("Mutual Fund Guide")
    st.write("MF Guide Content Goes Here")
