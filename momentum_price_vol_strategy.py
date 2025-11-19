import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy:
    """
    Momentum Breakout Trading Strategy based on price and volume.
    Long-only strategy that generates BUY/SELL/HOLD signals.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        volume_threshold: float = 1.5,
        breakout_threshold: float = 0.02,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ):
        """
        Initialize the Momentum Breakout Strategy.

        Args:
            lookback_period: Number of periods to look back for high/low (default: 20)
            volume_threshold: Volume multiplier vs average (default: 1.5x)
            breakout_threshold: Percentage above high to confirm breakout (default: 2%)
            stop_loss_pct: Stop loss percentage below entry (default: 5%)
            take_profit_pct: Take profit percentage above entry (default: 10%)
        """
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        self.breakout_threshold = breakout_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def fetch_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol
            period: Time period for historical data

        Returns:
            DataFrame with historical price and volume data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for momentum breakout strategy.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with calculated indicators
        """
        # Calculate rolling high and low
        df['rolling_high'] = df['High'].rolling(window=self.lookback_period).max()
        df['rolling_low'] = df['Low'].rolling(window=self.lookback_period).min()

        # Calculate average volume
        df['avg_volume'] = df['Volume'].rolling(window=self.lookback_period).mean()

        # Calculate volume ratio
        df['volume_ratio'] = df['Volume'] / df['avg_volume']

        # Calculate breakout level
        df['breakout_level'] = df['rolling_high'] * (1 + self.breakout_threshold)

        # Calculate breakdown level
        df['breakdown_level'] = df['rolling_low'] * (1 - self.breakout_threshold)

        # Price momentum (rate of change)
        df['momentum'] = df['Close'].pct_change(self.lookback_period)

        return df

    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        Generate trading signal based on momentum breakout logic.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Signal: 'BUY', 'SELL', or 'HOLD'
        """
        if df.empty or len(df) < self.lookback_period + 1:
            return 'HOLD'

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # BUY Signal Conditions:
        # 1. Price breaks above rolling high
        # 2. Volume is above threshold
        # 3. Positive momentum
        if (
            latest['Close'] > previous['rolling_high'] and
            latest['volume_ratio'] >= self.volume_threshold and
            latest['momentum'] > 0
        ):
            return 'BUY'

        # SELL Signal Conditions:
        # 1. Price breaks below rolling low
        # 2. High volume on breakdown
        # 3. Negative momentum
        if (
            latest['Close'] < previous['rolling_low'] and
            latest['volume_ratio'] >= self.volume_threshold and
            latest['momentum'] < 0
        ):
            return 'SELL'

        return 'HOLD'

    def analyze_stock(self, symbol: str) -> Dict:
        """
        Analyze a single stock and generate signal.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with analysis results
        """
        try:
            df = self.fetch_data(symbol)
            if df is None or df.empty:
                return {
                    'symbol': symbol,
                    'signal': 'HOLD',
                    'error': 'No data available'
                }

            df = self.calculate_indicators(df)
            signal = self.generate_signal(df)

            latest = df.iloc[-1]

            result = {
                'symbol': symbol,
                'signal': signal,
                'close_price': round(float(latest['Close']), 2),
                'rolling_high': round(float(latest['rolling_high']), 2),
                'rolling_low': round(float(latest['rolling_low']), 2),
                'volume_ratio': round(float(latest['volume_ratio']), 2),
                'momentum': round(float(latest['momentum']) * 100, 2),
                'date': latest.name.strftime('%Y-%m-%d') if hasattr(latest.name, 'strftime') else str(latest.name),
                'stop_loss': round(float(latest['Close']) * (1 - self.stop_loss_pct), 2),
                'take_profit': round(float(latest['Close']) * (1 + self.take_profit_pct), 2)
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'error': str(e)
            }

    def get_sp500_tickers(self) -> List[str]:
        """
        Get list of S&P 500 ticker symbols.

        Returns:
            List of ticker symbols
        """
        try:
            # Fetch S&P 500 tickers from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            # Clean up tickers (replace dots with dashes for Yahoo Finance)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {str(e)}")
            # Return a subset of common S&P 500 stocks as fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V', 'WMT']

    def get_nasdaq100_tickers(self) -> List[str]:
        """
        Get list of NASDAQ 100 ticker symbols.

        Returns:
            List of ticker symbols
        """
        try:
            # Fetch NASDAQ 100 tickers from Wikipedia
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(url)
            nasdaq_table = tables[4]  # The ticker table is usually the 5th table
            tickers = nasdaq_table['Ticker'].tolist()
            return tickers
        except Exception as e:
            logger.error(f"Error fetching NASDAQ 100 tickers: {str(e)}")
            # Return a subset of common NASDAQ 100 stocks as fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'PYPL']

    def analyze_sp500_stocks(self, output_file: str = 'sp500_momentum_signals.csv') -> pd.DataFrame:
        """
        Analyze all S&P 500 stocks and save results to CSV.

        Args:
            output_file: Output CSV file path

        Returns:
            DataFrame with analysis results
        """
        logger.info("Starting S&P 500 momentum analysis...")
        tickers = self.get_sp500_tickers()
        results = []

        for i, symbol in enumerate(tickers, 1):
            logger.info(f"Analyzing {symbol} ({i}/{len(tickers)})...")
            result = self.analyze_stock(symbol)
            results.append(result)

        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        return df_results

    def analyze_nasdaq100_stocks(self, output_file: str = 'nasdaq100_momentum_signals.csv') -> pd.DataFrame:
        """
        Analyze all NASDAQ 100 stocks and save results to CSV.

        Args:
            output_file: Output CSV file path

        Returns:
            DataFrame with analysis results
        """
        logger.info("Starting NASDAQ 100 momentum analysis...")
        tickers = self.get_nasdaq100_tickers()
        results = []

        for i, symbol in enumerate(tickers, 1):
            logger.info(f"Analyzing {symbol} ({i}/{len(tickers)})...")
            result = self.analyze_stock(symbol)
            results.append(result)

        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        return df_results


if __name__ == "__main__":
    # Example usage
    strategy = MomentumBreakoutStrategy(
        lookback_period=20,
        volume_threshold=1.5,
        breakout_threshold=0.02,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )

    # Test with a single stock
    result = strategy.analyze_stock('AAPL')
    print(f"\nSingle Stock Analysis for AAPL:")
    print(result)

    # Analyze S&P 500 stocks
    # sp500_results = strategy.analyze_sp500_stocks()
    # print(f"\nS&P 500 Analysis Complete. Buy signals: {len(sp500_results[sp500_results['signal'] == 'BUY'])}")

    # Analyze NASDAQ 100 stocks
    # nasdaq_results = strategy.analyze_nasdaq100_stocks()
    # print(f"\nNASDAQ 100 Analysis Complete. Buy signals: {len(nasdaq_results[nasdaq_results['signal'] == 'BUY'])}")