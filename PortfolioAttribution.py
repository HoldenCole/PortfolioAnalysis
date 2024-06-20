import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define tickers and their sectors
tickers = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'NASDAQ',
    '^DJI': 'Dow Jones',
    'TLT': 'ETF',
    'ACWI': 'Global'
}

# Download historical data
data = yf.download(list(tickers.keys()), start='2020-01-01', end='2023-01-01')['Adj Close']

# Step 2: Data Preprocessing
# Calculate daily returns
returns = data.pct_change().dropna()

# Step 3: Performance Calculation and Benchmark Comparison
# Calculate cumulative returns
cumulative_returns = (1 + returns).cumprod()

# Calculate total return for each asset
total_returns = cumulative_returns.iloc[-1] - 1

# Example weights for each asset
weights = {
    '^GSPC': 0.2,
    '^IXIC': 0.2,
    '^DJI': 0.2,
    'TLT': 0.2,
    'ACWI': 0.2
}

# Calculate portfolio return
portfolio_return = sum(total_returns[ticker] * weight for ticker, weight in weights.items())

# Normalize returns to calculate attribution
normalized_returns = returns / returns.std()

# Calculate the contribution of each asset
contribution = normalized_returns.mul(list(weights.values()), axis=1).sum()

# Create a DataFrame to summarize the results
attribution_df = pd.DataFrame({
    'Total Return': total_returns,
    'Contribution': contribution
})

# Step 4: Sector Analysis
# Assign sectors to the tickers
sectors = {
    '^GSPC': 'Equity',
    '^IXIC': 'Equity',
    '^DJI': 'Equity',
    'TLT': 'Bond',
    'ACWI': 'Global'
}

# Add sector information to the attribution DataFrame
attribution_df['Sector'] = [sectors[ticker] for ticker in attribution_df.index]

# Group by sector and calculate sector contribution
sector_contribution = attribution_df.groupby('Sector')['Contribution'].sum()

# Step 5: Risk Analysis
# Calculate portfolio beta using S&P 500 as market proxy
market_returns = returns['^GSPC']
betas = returns.apply(lambda x: x.cov(market_returns) / market_returns.var())

# Add beta information to the attribution DataFrame
attribution_df['Beta'] = betas

# Step 6: Visualization
# Plot cumulative returns
cumulative_returns.plot(figsize=(10, 6))
plt.title('Cumulative Returns of Selected Assets')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend(loc='upper left')
plt.show()

# Plot attribution analysis
attribution_df.plot(kind='bar', y='Contribution', figsize=(10, 6))
plt.title('Attribution Analysis')
plt.xlabel('Assets')
plt.ylabel('Contribution')
plt.show()

# Plot sector contribution
sector_contribution.plot(kind='bar', figsize=(10, 6))
plt.title('Sector Contribution Analysis')
plt.xlabel('Sector')
plt.ylabel('Contribution')
plt.show()

# Print the final attribution DataFrame with beta values
print(attribution_df)

