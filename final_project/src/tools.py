import datetime as dt 
from typing import Union, Dict, Set, List, TypedDict, Annotated
import pandas as pd
from langchain_core.tools import tool
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price

@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicator for a given ticker."""
    try:
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=24*3),
            end=dt.datetime.now(),
            interval='1wk'
        )
        df= data.copy()
        data.reset_index(inplace=True)
        data.Date = data.Date.astype(str)
        
        indicators = {}
        
        rsi_series = RSIIndicator(df['Close'].squeeze(), window=14).rsi().iloc[-12:]
        indicators["RSI"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in rsi_series.dropna().to_dict().items()}
        
        sto_series = StochasticOscillator(
            df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze(), window=14).stoch().iloc[-12:]
        indicators["Stochastic_Oscillator"] = {
                    date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in sto_series.dropna().to_dict().items()}

        macd = MACD(df['Close'].squeeze())
        macd_series = macd.macd().iloc[-12:]
        indicators["MACD"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in macd_series.to_dict().items()}
        
        macd_signal_series = macd.macd_signal().iloc[-12:]
        indicators["MACD_Signal"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in macd_signal_series.to_dict().items()}
        
        vwap_series = volume_weighted_average_price(
            high=df['High'].squeeze(), low=df['Low'].squeeze(), close=df['Close'].squeeze(), 
            volume=df['Volume'].squeeze(),
        ).iloc[-12:]
        indicators["vwap"] = {date.strftime('%Y-%m-%d'): int(value) 
                    for date, value in vwap_series.to_dict().items()}
        
        return {'stock_price': data.to_dict(orient='records'),
                'indicators': indicators}

    except Exception as e:
        return f"Error fetching price data: {str(e)}"
    
@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'pe_ratio': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'profit_margins': info.get('profitMargins')
        }
    except Exception as e:
        return f"Error fetching ratios: {str(e)}"
  
@tool
def compare_stocks(ticker1: str, ticker2: str) -> Union[Dict, str]:
    """Compares two stocks based on key financial metrics."""
    try:
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        
        info1 = stock1.info
        info2 = stock2.info
        
        return {
            ticker1: {
                'pe_ratio': info1.get('forwardPE'),
                'price_to_book': info1.get('priceToBook'),
                'debt_to_equity': info1.get('debtToEquity'),
                'profit_margins': info1.get('profitMargins')
            },
            ticker2: {
                'pe_ratio': info2.get('forwardPE'),
                'price_to_book': info2.get('priceToBook'),
                'debt_to_equity': info2.get('debtToEquity'),
                'profit_margins': info2.get('profitMargins')
            }
        }
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"
    
@tool
def get_stock_analysis(ticker: str) -> Union[Dict, str]:
    """Performs a detailed analysis of the stock based on historical data."""
    try:
        data = yf.download(
            ticker,
            start=dt.datetime.now() - dt.timedelta(weeks=24*3),
            end=dt.datetime.now(),
            interval='1wk'
        )
        df = data.copy()
        df['Date'] = df.index
        
        # Menghitung rata-rata harga penutupan
        average_close = df['Close'].mean()
        
        # Menghitung RSI
        rsi = RSIIndicator(df['Close'].squeeze(), window=14).rsi().iloc[-1]
        
        # Rekomendasi berdasarkan RSI
        recommendation = "Hold"
        if rsi < 30:
            recommendation = "Buy"
        elif rsi > 70:
            recommendation = "Sell"
        
        return {
            'average_close': average_close,
            'current_rsi': rsi,
            'recommendation': recommendation
        }
    
    except Exception as e:
        return f"Error performing stock analysis: {str(e)}"
    

@tool
def compare_stocks_advanced(ticker1: str, ticker2: str) -> Union[Dict, str]:
    """Compares two stocks based on technical analysis and price metrics."""
    try:
        # Mengambil data untuk kedua saham
        data1 = yf.download(ticker1, start=dt.datetime.now() - dt.timedelta(weeks=24*3), end=dt.datetime.now(), interval='1wk')
        data2 = yf.download(ticker2, start=dt.datetime.now() - dt.timedelta(weeks=24*3), end=dt.datetime.now(), interval='1wk')
        
        # Menghitung RSI untuk kedua saham
        rsi1 = RSIIndicator(data1['Close'].squeeze(), window=14).rsi().iloc[-1]
        rsi2 = RSIIndicator(data2['Close'].squeeze(), window=14).rsi().iloc[-1]
        
        # Menghitung MACD untuk kedua saham
        macd1 = MACD(data1['Close'].squeeze())
        macd_signal1 = macd1.macd_signal().iloc[-1]
        
        macd2 = MACD(data2['Close'].squeeze())
        macd_signal2 = macd2.macd_signal().iloc[-1]
        
        # Menghitung rata-rata harga penutupan
        avg_close1 = data1['Close'].mean()
        avg_close2 = data2['Close'].mean()
        
        # Menentukan mana yang lebih baik berdasarkan analisis
        better_stock = ticker1 if (rsi1 < 30 and avg_close1 < avg_close2) else ticker2
        
        return {
            ticker1: {
                'rsi': rsi1,
                'macd_signal': macd_signal1,
                'average_close': avg_close1
            },
            ticker2: {
                'rsi': rsi2,
                'macd_signal': macd_signal2,
                'average_close': avg_close2
            },
            'better_stock': better_stock
        }
    
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"
    

@tool
def get_stock_performance_summary(ticker: str) -> Union[Dict, str]:
    """Provides a performance summary of the stock including price change, volatility, and trend analysis."""
    try:
        # Mengambil data historis saham
        data = yf.download(ticker, start=dt.datetime.now() - dt.timedelta(weeks=24*3), end=dt.datetime.now(), interval='1wk')
        
        # Menghitung persentase perubahan harga
        price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        
        # Menghitung volatilitas (standar deviasi dari harga penutupan)
        volatility = data['Close'].std()
        
        # Analisis tren menggunakan SMA (Simple Moving Average)
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        trend = "Bullish" if data['Close'].iloc[-1] > data['SMA_20'].iloc[-1] else "Bearish"
        
        return {
            'ticker': ticker,
            'price_change_percentage': price_change,
            'volatility': volatility,
            'current_trend': trend
        }
    
    except Exception as e:
        return f"Error fetching performance summary: {str(e)}"
    
@tool
def get_fair_value_recommendation(ticker: str) -> Union[Dict, str]:
    """Provides a fair value recommendation for a stock based on financial reports and technical analysis."""
    try:
        # Mengambil data saham
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Mengambil rasio keuangan yang relevan
        pe_ratio = info.get('forwardPE')
        price_to_book = info.get('priceToBook')
        profit_margin = info.get('profitMargins')
        
        # Menghitung estimasi laba per saham (EPS) berdasarkan profit margin
        estimated_eps = info.get('regularMarketPrice') * profit_margin
        
        # Menghitung harga wajar menggunakan metode Gordon Growth Model
        growth_rate = 0.05  # Asumsi pertumbuhan tahunan 5%
        fair_value = estimated_eps * (1 + growth_rate) / (0.10 - growth_rate)  # Menggunakan discount rate 10%
        
        # Analisis teknis: Menghitung RSI dan SMA
        historical_data = yf.download(ticker, start=dt.datetime.now() - dt.timedelta(weeks=24*3), end=dt.datetime.now(), interval='1wk')
        rsi = RSIIndicator(historical_data['Close'].squeeze(), window=14).rsi().iloc[-1]
        sma_50 = historical_data['Close'].rolling(window=50).mean().iloc[-1]
        
        # Rekomendasi berdasarkan harga wajar dan analisis teknis
        current_price = info.get('regularMarketPrice')
        recommendation = "Hold"
        
        if current_price < fair_value and rsi < 30:
            recommendation = "Buy"
        elif current_price > fair_value and rsi > 70:
            recommendation = "Sell"
        
        return {
            'ticker': ticker,
            'fair_value': fair_value,
            'current_price': current_price,
            'pe_ratio': pe_ratio,
            'price_to_book': price_to_book,
            'profit_margin': profit_margin,
            'rsi': rsi,
            'sma_50': sma_50,
            'recommendation': recommendation
        }
    
    except Exception as e:
        return f"Error calculating fair value recommendation: {str(e)}"