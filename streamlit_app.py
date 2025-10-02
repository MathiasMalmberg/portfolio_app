import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Portfolio Analysis", layout="wide", page_icon="📈")

# Title
st.title("📊 Portfolio Performance Analysis")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

# Date range selector
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2025, 5, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime(2025, 9, 18))

# Risk-free rate
risk_free_rate_annual = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)", 
    min_value=0.0, 
    max_value=10.0, 
    value=2.6, 
    step=0.1
) / 100

# Tickers and weights
st.sidebar.subheader("Portfolio Holdings")

# Default tickers and weights
default_tickers = [
    "INVE-B.ST", "TFBANK.ST", "NVO", "FREJA.ST", "LUND-B.ST", 
    "VOLV-B.ST", "VEFAB.ST", "VOLCAR-B.ST", "SHB-A.ST", "ABB.ST",
    "MC.PA", "CELH", "EMBRAC-B.ST", "DYVOX.ST", "EVO.ST", 
    "ABEV", "FLAT-B.ST"
]

default_weights = [
    12.1, 10.1, 9.1, 7.1, 6.5, 6.3, 5.8, 5.6, 5.4, 
    5.1, 5.0, 4.7, 4.3, 4.3, 3.8, 2.4, 2.4
]

# Option to edit portfolio
edit_portfolio = st.sidebar.checkbox("Edit Portfolio", value=False)

if edit_portfolio:
    num_stocks = st.sidebar.number_input("Number of stocks", min_value=1, max_value=50, value=len(default_tickers))
    
    tickers = []
    weights_percent = []
    
    for i in range(int(num_stocks)):
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            ticker = st.text_input(f"Ticker {i+1}", value=default_tickers[i] if i < len(default_tickers) else "", key=f"ticker_{i}")
        with col2:
            weight = st.number_input(f"Weight %", min_value=0.0, max_value=100.0, 
                                    value=float(default_weights[i]) if i < len(default_weights) else 0.0, 
                                    key=f"weight_{i}")
        if ticker:
            tickers.append(ticker)
            weights_percent.append(weight)
else:
    tickers = default_tickers
    weights_percent = default_weights

# Run analysis button
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    
    # Validate weights
    if abs(sum(weights_percent) - 100) > 0.1:
        st.error(f"⚠️ Weights must sum to 100%. Current sum: {sum(weights_percent):.1f}%")
        st.stop()
    
    # Show loading spinner
    with st.spinner("Downloading data and calculating portfolio metrics..."):
        try:
            # Download data
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
            
            # Handle single ticker case
            if len(tickers) == 1:
                data = data.to_frame()
                data.columns = [tickers[0]]
            
            # Fill missing data
            data = data.ffill().dropna()
            
            if data.empty:
                st.error("❌ No data available for the selected date range and tickers.")
                st.stop()
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Convert weights to decimal
            weights = np.array(weights_percent) / 100
            
            # Portfolio returns
            portfolio_returns = returns.dot(weights)
            
            # Risk-free rate (daily)
            risk_free_rate_daily = risk_free_rate_annual / 252
            
            # Excess returns
            excess_returns = portfolio_returns - risk_free_rate_daily
            
            # Sharpe ratio
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod() * 100
            
            # Display key metrics at the top
            st.markdown("## 📈 Key Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_return = (cumulative_returns.iloc[-1] - 100)
            annualized_return = portfolio_returns.mean() * 252 * 100
            annualized_volatility = portfolio_returns.std() * np.sqrt(252) * 100
            
            with col1:
                st.metric("Total Return", f"{total_return:.2f}%")
            with col2:
                st.metric("Annualized Return", f"{annualized_return:.2f}%")
            with col3:
                st.metric("Annualized Volatility", f"{annualized_volatility:.2f}%")
            with col4:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
            
            st.markdown("---")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Returns", "🥧 Composition", "📋 Statistics"])
            
            with tab1:
                st.subheader("Cumulative Portfolio Value")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                        linewidth=2.5, color='#2E86AB')
                ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Initial Value')
                ax1.set_xlabel('Date', fontsize=12)
                ax1.set_ylabel('Portfolio Value (Base = 100)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=10)
                
                # Annotation
                final_value = cumulative_returns.iloc[-1]
                ax1.annotate(f'Final: {final_value:.2f}', 
                            xy=(cumulative_returns.index[-1], final_value),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                plt.tight_layout()
                st.pyplot(fig1)
                
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Daily Returns Distribution")
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    ax2.hist(portfolio_returns * 100, bins=30, color='#A23B72', 
                            alpha=0.7, edgecolor='black')
                    ax2.axvline(x=portfolio_returns.mean() * 100, color='red', 
                               linestyle='--', linewidth=2, 
                               label=f'Mean: {portfolio_returns.mean()*100:.3f}%')
                    ax2.set_xlabel('Daily Return (%)', fontsize=10)
                    ax2.set_ylabel('Frequency', fontsize=10)
                    ax2.legend(fontsize=9)
                    ax2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                with col2:
                    st.subheader("Daily Returns Over Time")
                    fig3, ax3 = plt.subplots(figsize=(6, 5))
                    ax3.plot(portfolio_returns.index, portfolio_returns * 100, 
                            linewidth=1, color='#F18F01', alpha=0.7)
                    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                    ax3.fill_between(portfolio_returns.index, 0, portfolio_returns * 100, 
                                    where=(portfolio_returns > 0), color='green', alpha=0.3)
                    ax3.fill_between(portfolio_returns.index, 0, portfolio_returns * 100, 
                                    where=(portfolio_returns <= 0), color='red', alpha=0.3)
                    ax3.set_xlabel('Date', fontsize=10)
                    ax3.set_ylabel('Daily Return (%)', fontsize=10)
                    ax3.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig3)
            
            with tab3:
                st.subheader("Portfolio Composition")
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
                wedges, texts, autotexts = ax4.pie(weights_percent, labels=tickers, 
                                                    autopct='%1.1f%%', colors=colors, 
                                                    startangle=90)
                plt.setp(autotexts, size=9, weight="bold")
                plt.setp(texts, size=10)
                plt.tight_layout()
                st.pyplot(fig4)
                
                # Display weights table
                st.subheader("Holdings Breakdown")
                holdings_df = pd.DataFrame({
                    'Ticker': tickers,
                    'Weight (%)': weights_percent
                }).sort_values('Weight (%)', ascending=False)
                st.dataframe(holdings_df, use_container_width=True)
            
            with tab4:
                st.subheader("Detailed Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Return Metrics")
                    stats_df1 = pd.DataFrame({
                        'Metric': [
                            'Total Return',
                            'Annualized Return',
                            'Mean Daily Return',
                            'Best Day',
                            'Worst Day'
                        ],
                        'Value': [
                            f"{total_return:.2f}%",
                            f"{annualized_return:.2f}%",
                            f"{portfolio_returns.mean() * 100:.3f}%",
                            f"{portfolio_returns.max() * 100:.2f}%",
                            f"{portfolio_returns.min() * 100:.2f}%"
                        ]
                    })
                    st.dataframe(stats_df1, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("### Risk Metrics")
                    stats_df2 = pd.DataFrame({
                        'Metric': [
                            'Annualized Volatility',
                            'Daily Volatility',
                            'Sharpe Ratio',
                            'Max Drawdown',
                            'Trading Days'
                        ],
                        'Value': [
                            f"{annualized_volatility:.2f}%",
                            f"{portfolio_returns.std() * 100:.3f}%",
                            f"{sharpe_ratio:.3f}",
                            f"{((cumulative_returns / cumulative_returns.cummax() - 1).min() * 100):.2f}%",
                            f"{len(portfolio_returns)}"
                        ]
                    })
                    st.dataframe(stats_df2, use_container_width=True, hide_index=True)
                
                # Period information
                st.markdown("### Analysis Period")
                st.info(f"**Start:** {portfolio_returns.index[0].strftime('%Y-%m-%d')} | **End:** {portfolio_returns.index[-1].strftime('%Y-%m-%d')}")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please check your ticker symbols and date range.")

else:
    st.info("👈 Configure your portfolio in the sidebar and click 'Run Analysis' to begin.")
    
    # Show sample portfolio
    st.markdown("### Current Portfolio Preview")
    preview_df = pd.DataFrame({
        'Ticker': tickers[:5] + ['...'] if len(tickers) > 5 else tickers,
        'Weight (%)': weights_percent[:5] + ['...'] if len(weights_percent) > 5 else weights_percent
    })
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Data from Yahoo Finance*")