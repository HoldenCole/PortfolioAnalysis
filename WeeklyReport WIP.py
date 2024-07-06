import pandas as pd
import requests
from jinja2 import Template
from google.oauth2 import service_account
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64

# Load the Excel file
file_path = 'path_to_your/Report Signup.xlsx'  # Replace with the actual path to your Excel file
df = pd.read_excel(file_path)

# Your Alpha Vantage API key
api_key = 'your_alpha_vantage_api_key'  # Replace with your Alpha Vantage API key

# Function to fetch real stock data
def fetch_real_stock_data(ticker):
    try:
        # Fetch daily time series data
        ts_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}'
        ts_response = requests.get(ts_url).json()
        time_series = ts_response.get('Time Series (Daily)', {})
        
        # Fetch earnings data
        earnings_url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={api_key}'
        earnings_response = requests.get(earnings_url).json()
        earnings_dates = earnings_response.get('quarterlyEarnings', [])

        # Fetch company news
        news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
        news_response = requests.get(news_url).json()
        news_articles = news_response.get('feed', [])
        big_news = news_articles[0]['title'] if news_articles else "No recent news"

        # Get the latest close price and calculate weekly performance
        if len(time_series) < 5:
            return None
        dates = sorted(time_series.keys(), reverse=True)
        current_price = float(time_series[dates[0]]['4. close'])
        last_week_price = float(time_series[dates[5]]['4. close'])
        weekly_performance = (current_price - last_week_price) / last_week_price * 100

        # Get YTD return
        first_price = float(time_series[dates[-1]]['4. close'])
        ytd_return = (current_price - first_price) / first_price * 100

        # Get earnings information
        next_earnings_date = earnings_dates[0]['reportDate'] if earnings_dates else "N/A"
        earnings = f"{earnings_dates[0]['reportedEPS']} EPS" if earnings_dates else "N/A"

        return {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'weekly_performance': f"{weekly_performance:.2f}%",
            'ytd_return': f"{ytd_return:.2f}%",
            'earnings': earnings,
            'next_earnings_date': next_earnings_date,
            'big_news': big_news,
            'corporate_announcements': "Check company website for announcements.",
            'insider_transactions': "Check SEC filings for insider transactions.",
            'analyst_ratings': "Check financial websites for analyst ratings."
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# HTML template for the report
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Weekly Stock Report</title>
</head>
<body>
    <h1>Weekly Stock Report for {{ name }}</h1>
    <p>Email: {{ email }}</p>
    {% for stock in stocks %}
    <h2>{{ stock.ticker }}</h2>
    <p>Current Price: {{ stock.current_price }}</p>
    <p>Weekly Performance: {{ stock.weekly_performance }}</p>
    <p>YTD Return: {{ stock.ytd_return }}</p>
    <p>Earnings: {{ stock.earnings }}</p>
    <p>Next Earnings Date: {{ stock.next_earnings_date }}</p>
    <p>Big News: {{ stock.big_news }}</p>
    <p>Corporate Announcements: {{ stock.corporate_announcements }}</p>
    <p>Insider Transactions: {{ stock.insider_transactions }}</p>
    <p>Analyst Ratings: {{ stock.analyst_ratings }}</p>
    {% endfor %}
</body>
</html>
"""

def generate_report(user_data, stock_data):
    template = Template(html_template)
    return template.render(name=user_data['name'], email=user_data['email'], stocks=stock_data)

# Function to create a message
def create_message(to, subject, message_text):
    message = MIMEText(message_text, 'html')
    message['to'] = to
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes())
    return {'raw': raw.decode()}

# Function to send email using Gmail API
def send_email(service, user_id, message):
    try:
        message = service.users().messages().send(userId=user_id, body=message).execute()
        print(f"Message Id: {message['id']}")
        return message
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Authenticate and build the Gmail API service
def get_gmail_service():
    creds = service_account.Credentials.from_service_account_file(
        'path_to_your/credentials.json', 
        scopes=['https://www.googleapis.com/auth/gmail.send']
    )
    service = build('gmail', 'v1', credentials=creds)
    return service

# Process each user in the Excel file
gmail_service = get_gmail_service()
for index, row in df.iterrows():
    user_data = {
        'name': row['Name'],
        'email': row['Email'],
        'stocks': [row['Stock 1'], row['Stock 2'], row['Stock 3']]
    }

    # Fetch stock data for each ticker
    stock_data = [fetch_real_stock_data(ticker) for ticker in user_data['stocks'] if fetch_real_stock_data(ticker) is not None]

    # Generate the report
    report = generate_report(user_data, stock_data)

    # Create and send the email
    message = create_message(user_data['email'], 'Weekly Stock Report', report)
    send_email(gmail_service, 'me', message)
