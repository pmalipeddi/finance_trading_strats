import yfinance as yf
import pandas as pd
import numpy as np
import requests
import sys
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from yahoo_finance_api import YahooFinanceAPI


class TradingStrategies:
    """
    A class for implementing various trading strategies.
    
    Args:
        symbol (str, optional): Stock ticker symbol for strategy analysis
        symbols (List[str], optional): List of stock ticker symbols for analysis
    """
    
    def __init__(self, symbol: Optional[str] = None, symbols: Optional[List[str]] = None):
        """
        Initialize the TradingStrategies instance.
        
        Args:
            symbol (str, optional): Single stock ticker symbol (e.g., 'AAPL')
            symbols (List[str], optional): List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        """
        self.symbol = symbol.upper() if symbol else None
        self.symbols = [s.upper() for s in symbols] if symbols else None
        self.api = YahooFinanceAPI()
        
        # If both are provided, prefer symbols list
        if symbols and symbol:
            self.symbol = None
    
    def get_historical_data(
        self,
        symbol: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical stock price data.
        
        Args:
            symbol (str, optional): Stock ticker symbol. If not provided, uses the symbol
                                    from initialization.
            period (str): Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval (str): Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        
        Returns:
            pd.DataFrame: Historical price data with columns: Open, High, Low, Close, Volume
        
        Raises:
            ValueError: If no symbol is provided and no symbol was set during initialization.
        """
        if symbol is None:
            if self.symbol is None:
                raise ValueError("No symbol provided. Either pass a symbol to this method or initialize the class with a symbol.")
            symbol = self.symbol
        else:
            symbol = symbol.upper()
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data retrieved for symbol {symbol}")
            
            return data
        except Exception as e:
            raise ValueError(f"Error fetching historical data for {symbol}: {str(e)}")
    
    def calculate_moving_average(
        self,
        data: pd.DataFrame,
        window: int,
        column: str = "Close"
    ) -> pd.Series:
        """
        Calculate moving average for a given window.
        
        Args:
            data (pd.DataFrame): Historical price data
            window (int): Number of periods for moving average
            column (str): Column name to calculate MA for (default: "Close")
        
        Returns:
            pd.Series: Moving average values
        """
        return data[column].rolling(window=window).mean()
    
    def moving_average_crossover(
        self,
        symbol: Optional[str] = None,
        fast_period: int = 50,
        slow_period: int = 200,
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict:
        """
        Implement Moving Average Crossover strategy.
        
        Buy Signal: When fast MA crosses above slow MA (Golden Cross)
        Sell Signal: When fast MA crosses below slow MA (Death Cross)
        
        Args:
            symbol (str, optional): Stock ticker symbol. If not provided, uses the symbol
                                    from initialization.
            fast_period (int): Period for fast moving average (default: 50)
            slow_period (int): Period for slow moving average (default: 200)
            period (str): Historical data period (default: "2y")
            interval (str): Data interval (default: "1d")
        
        Returns:
            dict: Dictionary containing:
                - data: DataFrame with price data and moving averages
                - signals: DataFrame with buy/sell signals
                - current_signal: Current trading signal (BUY/SELL/HOLD)
                - fast_ma: Current fast moving average value
                - slow_ma: Current slow moving average value
                - price: Current price
        
        Raises:
            ValueError: If no symbol is provided and no symbol was set during initialization.
        """
        if symbol is None:
            if self.symbol is None:
                raise ValueError("No symbol provided. Either pass a symbol to this method or initialize the class with a symbol.")
            symbol = self.symbol
        else:
            symbol = symbol.upper()
        
        # Get historical data
        data = self.get_historical_data(symbol, period=period, interval=interval)
        
        # Calculate moving averages
        data['Fast_MA'] = self.calculate_moving_average(data, fast_period, 'Close')
        data['Slow_MA'] = self.calculate_moving_average(data, slow_period, 'Close')
        
        # Identify crossovers
        # Signal: 1 for buy (fast crosses above slow), -1 for sell (fast crosses below slow), 0 for hold
        data['Signal'] = 0
        data['Position'] = 0
        
        # Find crossovers
        data.loc[data['Fast_MA'] > data['Slow_MA'], 'Position'] = 1  # Bullish
        data.loc[data['Fast_MA'] < data['Slow_MA'], 'Position'] = -1  # Bearish
        
        # Generate signals at crossover points
        data['Signal'] = data['Position'].diff()
        
        # Create signals dataframe
        signals = data[data['Signal'] != 0].copy()
        signals['Action'] = signals['Signal'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
        
        # Determine current signal
        current_price = data['Close'].iloc[-1]
        current_fast_ma = data['Fast_MA'].iloc[-1]
        current_slow_ma = data['Slow_MA'].iloc[-1]
        previous_fast_ma = data['Fast_MA'].iloc[-2] if len(data) > 1 else current_fast_ma
        previous_slow_ma = data['Slow_MA'].iloc[-2] if len(data) > 1 else current_slow_ma
        
        # Check for recent crossover
        if pd.isna(current_fast_ma) or pd.isna(current_slow_ma):
            current_signal = "HOLD (Insufficient Data)"
        elif current_fast_ma > current_slow_ma and previous_fast_ma <= previous_slow_ma:
            current_signal = "BUY (Golden Cross)"
        elif current_fast_ma < current_slow_ma and previous_fast_ma >= previous_slow_ma:
            current_signal = "SELL (Death Cross)"
        elif current_fast_ma > current_slow_ma:
            current_signal = "HOLD (Bullish)"
        else:
            current_signal = "HOLD (Bearish)"
        
        return {
            'symbol': symbol,
            'data': data,
            'signals': signals,
            'current_signal': current_signal,
            'current_price': float(current_price),
            'fast_ma': float(current_fast_ma) if not pd.isna(current_fast_ma) else None,
            'slow_ma': float(current_slow_ma) if not pd.isna(current_slow_ma) else None,
            'fast_period': fast_period,
            'slow_period': slow_period,
            'total_signals': len(signals),
            'buy_signals': len(signals[signals['Signal'] > 0]) if len(signals) > 0 else 0,
            'sell_signals': len(signals[signals['Signal'] < 0]) if len(signals) > 0 else 0
        }
    
    def get_crossover_signals(
        self,
        symbol: Optional[str] = None,
        fast_period: int = 50,
        slow_period: int = 200,
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get only the crossover signals (buy/sell points) for a symbol.
        
        Args:
            symbol (str, optional): Stock ticker symbol
            fast_period (int): Period for fast moving average
            slow_period (int): Period for slow moving average
            period (str): Historical data period
            interval (str): Data interval
        
        Returns:
            pd.DataFrame: DataFrame with only the dates where crossovers occurred
        """
        result = self.moving_average_crossover(
            symbol=symbol,
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
            interval=interval
        )
        
        if len(result['signals']) > 0:
            return result['signals'][['Close', 'Fast_MA', 'Slow_MA', 'Action']]
        else:
            return pd.DataFrame()
    
    def backtest_strategy(
        self,
        symbol: Optional[str] = None,
        fast_period: int = 50,
        slow_period: int = 200,
        period: str = "2y",
        interval: str = "1d",
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Backtest the moving average crossover strategy.
        
        Args:
            symbol (str, optional): Stock ticker symbol
            fast_period (int): Period for fast moving average
            slow_period (int): Period for slow moving average
            period (str): Historical data period
            interval (str): Data interval
            initial_capital (float): Starting capital for backtest
        
        Returns:
            dict: Backtest results including returns, trades, and performance metrics
        """
        result = self.moving_average_crossover(
            symbol=symbol,
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
            interval=interval
        )
        
        data = result['data'].copy()
        signals = result['signals']
        
        # Initialize backtest variables
        capital = initial_capital
        shares = 0
        trades = []
        position = 0  # 0: no position, 1: long position
        
        # Simulate trading
        for idx, row in data.iterrows():
            # Check if there's a signal at this date
            if idx in signals.index:
                signal = signals.loc[idx, 'Signal']
                price = row['Close']
                
                if signal > 0 and position == 0:  # Buy signal
                    shares = capital / price
                    capital = 0
                    position = 1
                    trades.append({
                        'Date': idx,
                        'Action': 'BUY',
                        'Price': price,
                        'Shares': shares,
                        'Capital': capital
                    })
                elif signal < 0 and position == 1:  # Sell signal
                    capital = shares * price
                    shares = 0
                    position = 0
                    trades.append({
                        'Date': idx,
                        'Action': 'SELL',
                        'Price': price,
                        'Shares': shares,
                        'Capital': capital
                    })
        
        # Close final position if still holding
        final_price = data['Close'].iloc[-1]
        if position == 1:
            capital = shares * final_price
            trades.append({
                'Date': data.index[-1],
                'Action': 'SELL (Final)',
                'Price': final_price,
                'Shares': shares,
                'Capital': capital
            })
        
        # Calculate returns
        final_value = capital
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # Buy and hold return for comparison
        buy_hold_return = ((final_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        
        # Calculate number of trades
        num_trades = len([t for t in trades if t['Action'] in ['BUY', 'SELL']])
        
        return {
            'symbol': result['symbol'],
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'num_trades': num_trades,
            'trades': trades,
            'outperformed_buy_hold': total_return > buy_hold_return
        }

    def _fetch_constituent_tickers(
        self,
        url: str,
        symbol_candidates: Sequence[str],
        headers: Optional[Dict[str, str]] = None,
        table_indices: Optional[Sequence[int]] = None
    ) -> List[str]:
        """
        Retrieve a list of tickers from a data table on a web page.

        Args:
            url (str): URL containing the constituents table.
            symbol_candidates (Sequence[str]): Lower-case column name candidates that may contain tickers.
            headers (dict, optional): HTTP headers to include in the request.
            table_indices (Sequence[int], optional): Specific table indices to inspect. If None, all tables are checked.

        Returns:
            List[str]: Parsed list of ticker symbols.
        """
        request_headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            response = requests.get(url, headers=request_headers, timeout=30)
            response.raise_for_status()
            tables = pd.read_html(response.text)
            if not tables:
                raise ValueError(f"No tables found at {url}")
        except Exception as exc:
            raise ValueError(f"Failed to retrieve constituents from {url}: {exc}")

        indices_to_check = table_indices or range(len(tables))
        for index in indices_to_check:
            if index >= len(tables):
                continue

            table = tables[index]
            if isinstance(table.columns, pd.MultiIndex):
                normalized_columns = [
                    " ".join(
                        str(part).strip()
                        for part in column
                        if part and str(part).strip().lower() != "nan"
                    ).strip()
                    for column in table.columns
                ]
                table.columns = normalized_columns
            else:
                table.columns = [str(col).strip() for col in table.columns]

            lower_column_map = {col.lower(): col for col in table.columns}

            for candidate in symbol_candidates:
                if candidate in lower_column_map:
                    column_name = lower_column_map[candidate]
                    tickers = [
                        symbol.replace(".", "-").upper()
                        for symbol in table[column_name].dropna().astype(str).tolist()
                    ]
                    return tickers

        available_columns = [
            str(col)
            for idx in indices_to_check
            if idx < len(tables)
            for col in tables[idx].columns
        ]
        raise ValueError(
            f"Unable to find a ticker column at {url}. "
            f"Columns inspected: {available_columns}"
        )

    def _process_single_ticker(
        self,
        ticker: str,
        fast_period: int,
        slow_period: int,
        period: str,
        interval: str
    ) -> Dict[str, Optional[float]]:
        """
        Process a single ticker and return its crossover summary.

        Args:
            ticker (str): Ticker symbol to process.
            fast_period (int): Fast moving average period.
            slow_period (int): Slow moving average period.
            period (str): Historical data period.
            interval (str): Historical data interval.

        Returns:
            Dict: Dictionary containing ticker crossover summary.
        """
        try:
            result = self.moving_average_crossover(
                symbol=ticker,
                fast_period=fast_period,
                slow_period=slow_period,
                period=period,
                interval=interval
            )
            data_frame = result.get("data")
            last_close_price = (
                float(data_frame["Close"].iloc[-1])
                if data_frame is not None and not data_frame.empty
                else result.get("current_price")
            )

            if data_frame is not None and not data_frame.empty:
                fast_series = data_frame["Fast_MA"].dropna()
                slow_series = data_frame["Slow_MA"].dropna()
                current_fast_ma = float(fast_series.iloc[-1]) if not fast_series.empty else None
                previous_fast_ma = float(fast_series.iloc[-2]) if len(fast_series) > 1 else current_fast_ma
                current_slow_ma = float(slow_series.iloc[-1]) if not slow_series.empty else None
                previous_slow_ma = float(slow_series.iloc[-2]) if len(slow_series) > 1 else current_slow_ma
            else:
                current_fast_ma = result.get("fast_ma")
                previous_fast_ma = current_fast_ma
                current_slow_ma = result.get("slow_ma")
                previous_slow_ma = current_slow_ma

            return {
                "Ticker": ticker,
                "Last Close Price": last_close_price,
                "Current Price": result.get("current_price"),
                "previous_fast_ma": previous_fast_ma,
                "previous_slow_ma": previous_slow_ma,
                "current_fast_ma": current_fast_ma,
                "current_slow_ma": current_slow_ma,
                "Signal": result.get("current_signal")
            }
        except Exception as exc:
            return {
                "Ticker": ticker,
                "Last Close Price": np.nan,
                "Current Price": np.nan,
                "previous_fast_ma": np.nan,
                "previous_slow_ma": np.nan,
                "current_fast_ma": np.nan,
                "current_slow_ma": np.nan,
                "Signal": f"ERROR: {str(exc)[:100]}"  # Truncate long error messages
            }

    def _generate_crossover_summary_for_tickers(
        self,
        tickers: Sequence[str],
        fast_period: int,
        slow_period: int,
        period: str,
        interval: str,
        csv_path: str,
        max_workers: int = 20
    ) -> pd.DataFrame:
        """
        Generate moving average crossover summary for a sequence of tickers using parallel processing.

        Args:
            tickers (Sequence[str]): Iterable of ticker symbols.
            fast_period (int): Fast moving average period.
            slow_period (int): Slow moving average period.
            period (str): Historical data period.
            interval (str): Historical data interval.
            csv_path (str): Path to write the resulting CSV.
            max_workers (int): Maximum number of concurrent workers (default: 20).

        Returns:
            pd.DataFrame: Summary DataFrame containing crossover signals and metrics.
        """
        results: List[Dict[str, Optional[float]]] = []
        total_tickers = len(tickers)
        
        print(f"Processing {total_tickers} tickers with {max_workers} concurrent workers...")
        
        # Process tickers in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    self._process_single_ticker,
                    ticker,
                    fast_period,
                    slow_period,
                    period,
                    interval
                ): ticker
                for ticker in tickers
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 50 == 0 or completed == total_tickers:
                        print(f"Progress: {completed}/{total_tickers} tickers processed ({completed/total_tickers*100:.1f}%)")
                except Exception as exc:
                    # Fallback error handling
                    results.append({
                        "Ticker": ticker,
                        "Last Close Price": np.nan,
                        "Current Price": np.nan,
                        "previous_fast_ma": np.nan,
                        "previous_slow_ma": np.nan,
                        "current_fast_ma": np.nan,
                        "current_slow_ma": np.nan,
                        "Signal": f"ERROR: {str(exc)[:100]}"
                    })
                    completed += 1

        # Sort results by ticker to maintain consistent order
        results.sort(key=lambda x: x["Ticker"])
        
        df = pd.DataFrame(
            results,
            columns=[
                "Ticker",
                "Last Close Price",
                "Current Price",
                "previous_fast_ma",
                "previous_slow_ma",
                "current_fast_ma",
                "current_slow_ma",
                "Signal",
            ]
        )
        
        # Try to save CSV, but don't fail if it doesn't work
        try:
            df.to_csv(csv_path, index=False)
            print(f"Completed! Results saved to {csv_path}")
        except Exception as csv_error:
            print(f"Warning: Could not save CSV to {csv_path}: {csv_error}", file=sys.stderr)
            print("Results are still available in the returned DataFrame", file=sys.stderr)
        
        return df

    def moving_average_crossover_sp500(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        period: str = "2y",
        interval: str = "1d",
        csv_path: str = "sp500_moving_average_signals.csv",
        max_workers: int = 20
    ) -> pd.DataFrame:
        """
        Calculate the moving average crossover for all S&P 500 stocks.

        Args:
            fast_period (int): Period for fast moving average (default: 50)
            slow_period (int): Period for slow moving average (default: 200)
            period (str): Historical data period (default: "2y")
            interval (str): Data interval (default: "1d")
            csv_path (str): File path to export the results CSV (default: "sp500_moving_average_signals.csv")
            max_workers (int): Maximum number of concurrent workers for parallel processing (default: 20)

        Returns:
            pd.DataFrame: DataFrame containing the crossover results for each S&P 500 stock
        """
        tickers = self._fetch_constituent_tickers(
            url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            symbol_candidates=("symbol", "ticker symbol", "ticker", "root")
        )
        return self._generate_crossover_summary_for_tickers(
            tickers=tickers,
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
            interval=interval,
            csv_path=csv_path,
            max_workers=max_workers
        )

    def moving_average_crossover_nasdaq100(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        period: str = "2y",
        interval: str = "1d",
        csv_path: str = "nasdaq100_moving_average_signals.csv",
        max_workers: int = 20
    ) -> pd.DataFrame:
        """
        Calculate the moving average crossover for all NASDAQ-100 stocks.

        Args:
            fast_period (int): Period for fast moving average (default: 50)
            slow_period (int): Period for slow moving average (default: 200)
            period (str): Historical data period (default: "2y")
            interval (str): Data interval (default: "1d")
            csv_path (str): File path to export the results CSV (default: "nasdaq100_moving_average_signals.csv")
            max_workers (int): Maximum number of concurrent workers for parallel processing (default: 20)

        Returns:
            pd.DataFrame: DataFrame containing the crossover results for each NASDAQ-100 stock
        """
        tickers = self._fetch_constituent_tickers(
            url="https://en.wikipedia.org/wiki/NASDAQ-100",
            symbol_candidates=("ticker", "ticker symbol", "symbol"),
            table_indices=(3, 4, 5)  # the constituents table typically appears later on the page
        )
        return self._generate_crossover_summary_for_tickers(
            tickers=tickers,
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
            interval=interval,
            csv_path=csv_path,
            max_workers=max_workers
        )


# Example usage
if __name__ == "__main__":
    # Create an instance
    strategies = TradingStrategies(symbol="AAPL")
    
    # Example 1: Moving Average Crossover
    print("="*60)
    print("Moving Average Crossover Strategy - AAPL")
    print("="*60)
    
    result = strategies.moving_average_crossover(fast_period=50, slow_period=200)
    
    print(f"\nSymbol: {result['symbol']}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Fast MA ({result['fast_period']}): ${result['fast_ma']:.2f}" if result['fast_ma'] else f"Fast MA ({result['fast_period']}): N/A")
    print(f"Slow MA ({result['slow_period']}): ${result['slow_ma']:.2f}" if result['slow_ma'] else f"Slow MA ({result['slow_period']}): N/A")
    print(f"Current Signal: {result['current_signal']}")
    print(f"Total Crossover Signals: {result['total_signals']}")
    print(f"Buy Signals: {result['buy_signals']}")
    print(f"Sell Signals: {result['sell_signals']}")
    
    # Show recent signals
    if len(result['signals']) > 0:
        print("\nRecent Crossover Signals:")
        print(result['signals'][['Close', 'Fast_MA', 'Slow_MA', 'Action']].tail(10))
    
    # Example 2: Backtest the strategy
    print("\n" + "="*60)
    print("Backtest Results")
    print("="*60)
    
    backtest = strategies.backtest_strategy(initial_capital=10000.0)
    
    print(f"\nInitial Capital: ${backtest['initial_capital']:,.2f}")
    print(f"Final Value: ${backtest['final_value']:,.2f}")
    print(f"Total Return: {backtest['total_return']:.2f}%")
    print(f"Buy & Hold Return: {backtest['buy_hold_return']:.2f}%")
    print(f"Number of Trades: {backtest['num_trades']}")
    print(f"Outperformed Buy & Hold: {backtest['outperformed_buy_hold']}")
    
    # Show last few trades
    if len(backtest['trades']) > 0:
        print("\nLast 5 Trades:")
        for trade in backtest['trades'][-5:]:
            print(f"  {trade['Date'].strftime('%Y-%m-%d')}: {trade['Action']} at ${trade['Price']:.2f}")

    # Example 3: Generate S&P 500 crossover summary and export to CSV
    print("\n" + "="*60)
    print("S&P 500 Moving Average Crossover Summary")
    print("="*60)

    sp500_df = strategies.moving_average_crossover_sp500(
        fast_period=50,
        slow_period=200,
        period="2y",
        interval="1d",
        csv_path="sp500_moving_average_signals.csv"
    )

    print(f"\nGenerated crossover summary for {len(sp500_df)} S&P 500 tickers.")
    print("Saved results to sp500_moving_average_signals.csv")

    # Example 4: Generate NASDAQ-100 crossover summary and export to CSV
    print("\n" + "="*60)
    print("NASDAQ-100 Moving Average Crossover Summary")
    print("="*60)

    nasdaq_df = strategies.moving_average_crossover_nasdaq100(
        fast_period=50,
        slow_period=200,
        period="2y",
        interval="1d",
        csv_path="nasdaq100_moving_average_signals.csv"
    )

    print(f"\nGenerated crossover summary for {len(nasdaq_df)} NASDAQ-100 tickers.")
    print("Saved results to nasdaq100_moving_average_signals.csv")

