import yfinance as yf
import pandas as pd
import numpy as np
import sys
import requests
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


class MeanReversionStrategy:
    """
    A class for implementing mean reversion trading strategies.
    
    Mean reversion strategies are based on the assumption that prices will
    revert to their mean/average after deviating from it.
    """
    
    def __init__(self, symbol: Optional[str] = None, symbols: Optional[List[str]] = None):
        """
        Initialize the MeanReversionStrategy instance.
        
        Args:
            symbol (str, optional): Single stock ticker symbol (e.g., 'AAPL')
            symbols (List[str], optional): List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        """
        self.symbol = symbol.upper() if symbol else None
        self.symbols = [s.upper() for s in symbols] if symbols else None
        
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
    
    def calculate_standard_deviation(
        self,
        data: pd.DataFrame,
        window: int,
        column: str = "Close"
    ) -> pd.Series:
        """
        Calculate standard deviation for a given window.
        
        Args:
            data (pd.DataFrame): Historical price data
            window (int): Number of periods for standard deviation
            column (str): Column name to calculate SD for (default: "Close")
        
        Returns:
            pd.Series: Standard deviation values
        """
        return data[column].rolling(window=window).std()
    
    def calculate_bollinger_bands(
        self,
        data: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
        column: str = "Close"
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands consist of:
        - Middle Band: Simple Moving Average (SMA)
        - Upper Band: SMA + (num_std * Standard Deviation)
        - Lower Band: SMA - (num_std * Standard Deviation)
        
        Args:
            data (pd.DataFrame): Historical price data
            window (int): Number of periods for moving average (default: 20)
            num_std (float): Number of standard deviations (default: 2.0)
            column (str): Column name to calculate bands for (default: "Close")
        
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (Upper Band, Middle Band, Lower Band)
        """
        middle_band = self.calculate_moving_average(data, window, column)
        std = self.calculate_standard_deviation(data, window, column)
        
        upper_band = middle_band + (num_std * std)
        lower_band = middle_band - (num_std * std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_rsi(
        self,
        data: pd.DataFrame,
        window: int = 14,
        column: str = "Close"
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI ranges from 0 to 100:
        - RSI > 70: Overbought (potential sell signal)
        - RSI < 30: Oversold (potential buy signal)
        
        Args:
            data (pd.DataFrame): Historical price data
            window (int): Number of periods for RSI calculation (default: 14)
            column (str): Column name to calculate RSI for (default: "Close")
        
        Returns:
            pd.Series: RSI values
        """
        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_z_score(
        self,
        data: pd.DataFrame,
        window: int = 20,
        column: str = "Close"
    ) -> pd.Series:
        """
        Calculate Z-score (standard score).
        
        Z-score measures how many standard deviations a price is from the mean.
        - Z-score > 2: Price is significantly above mean (potential sell)
        - Z-score < -2: Price is significantly below mean (potential buy)
        
        Args:
            data (pd.DataFrame): Historical price data
            window (int): Number of periods for mean and std calculation (default: 20)
            column (str): Column name to calculate Z-score for (default: "Close")
        
        Returns:
            pd.Series: Z-score values
        """
        mean = self.calculate_moving_average(data, window, column)
        std = self.calculate_standard_deviation(data, window, column)
        
        z_score = (data[column] - mean) / std
        
        return z_score
    
    def mean_reversion_signal(
        self,
        symbol: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d",
        method: str = "bollinger",
        window: int = 20,
        upper_threshold: float = 2.0,
        lower_threshold: float = -2.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0
    ) -> Dict:
        """
        Generate mean reversion trading signals.
        
        Args:
            symbol (str, optional): Stock ticker symbol
            period (str): Historical data period (default: "1y")
            interval (str): Data interval (default: "1d")
            method (str): Method to use - "bollinger", "rsi", "zscore", or "combined" (default: "bollinger")
            window (int): Window size for calculations (default: 20)
            upper_threshold (float): Upper threshold for Z-score/Bollinger (default: 2.0)
            lower_threshold (float): Lower threshold for Z-score/Bollinger (default: -2.0)
            rsi_overbought (float): RSI overbought threshold (default: 70.0)
            rsi_oversold (float): RSI oversold threshold (default: 30.0)
        
        Returns:
            dict: Dictionary containing signals, indicators, and current recommendations
        """
        if symbol is None:
            if self.symbol is None:
                raise ValueError("No symbol provided. Either pass a symbol to this method or initialize the class with a symbol.")
            symbol = self.symbol
        else:
            symbol = symbol.upper()
        
        # Get historical data
        data = self.get_historical_data(symbol, period=period, interval=interval)
        
        # Initialize signal columns
        data['Signal'] = 0
        data['Position'] = 0
        
        # Calculate indicators based on method
        if method == "bollinger" or method == "combined":
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(
                data, window=window, num_std=upper_threshold
            )
            data['Upper_Band'] = upper_band
            data['Middle_Band'] = middle_band
            data['Lower_Band'] = lower_band
            
            # Bollinger Bands signals
            # Buy when price touches or goes below lower band
            data.loc[data['Close'] <= data['Lower_Band'], 'Signal'] = 1
            # Sell when price touches or goes above upper band
            data.loc[data['Close'] >= data['Upper_Band'], 'Signal'] = -1
        
        if method == "rsi" or method == "combined":
            rsi = self.calculate_rsi(data, window=window)
            data['RSI'] = rsi
            
            # RSI signals
            # Buy when RSI < oversold threshold
            data.loc[data['RSI'] < rsi_oversold, 'Signal'] = 1
            # Sell when RSI > overbought threshold
            data.loc[data['RSI'] > rsi_overbought, 'Signal'] = -1
        
        if method == "zscore" or method == "combined":
            z_score = self.calculate_z_score(data, window=window)
            data['Z_Score'] = z_score
            
            # Z-score signals
            # Buy when Z-score < lower threshold (price below mean)
            data.loc[data['Z_Score'] < lower_threshold, 'Signal'] = 1
            # Sell when Z-score > upper threshold (price above mean)
            data.loc[data['Z_Score'] > upper_threshold, 'Signal'] = -1
        
        # For combined method, use majority vote or require multiple confirmations
        if method == "combined":
            # Count signals from each method
            bollinger_signal = (data['Close'] <= data['Lower_Band']).astype(int) - (data['Close'] >= data['Upper_Band']).astype(int)
            rsi_signal = (data['RSI'] < rsi_oversold).astype(int) - (data['RSI'] > rsi_overbought).astype(int)
            zscore_signal = (data['Z_Score'] < lower_threshold).astype(int) - (data['Z_Score'] > upper_threshold).astype(int)
            
            # Combined signal (at least 2 out of 3 methods agree)
            combined = bollinger_signal + rsi_signal + zscore_signal
            data['Signal'] = 0
            data.loc[combined >= 2, 'Signal'] = 1  # Buy if 2+ methods say buy
            data.loc[combined <= -2, 'Signal'] = -1  # Sell if 2+ methods say sell
        
        # Create signals dataframe (only where signals occur)
        signals = data[data['Signal'] != 0].copy()
        signals['Action'] = signals['Signal'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
        
        # Get current values
        current_price = data['Close'].iloc[-1]
        current_signal = "HOLD"
        
        # Determine current signal
        if method == "bollinger":
            if pd.notna(data['Lower_Band'].iloc[-1]) and pd.notna(data['Upper_Band'].iloc[-1]):
                if data['Close'].iloc[-1] <= data['Lower_Band'].iloc[-1]:
                    current_signal = "BUY (Price below Lower Bollinger Band)"
                elif data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
                    current_signal = "SELL (Price above Upper Bollinger Band)"
                else:
                    current_signal = "HOLD (Price within Bollinger Bands)"
            else:
                current_signal = "HOLD (Insufficient Data)"
        
        elif method == "rsi":
            if pd.notna(data['RSI'].iloc[-1]):
                rsi_value = data['RSI'].iloc[-1]
                if rsi_value < rsi_oversold:
                    current_signal = f"BUY (RSI={rsi_value:.2f} < {rsi_oversold})"
                elif rsi_value > rsi_overbought:
                    current_signal = f"SELL (RSI={rsi_value:.2f} > {rsi_overbought})"
                else:
                    current_signal = f"HOLD (RSI={rsi_value:.2f})"
            else:
                current_signal = "HOLD (Insufficient Data)"
        
        elif method == "zscore":
            if pd.notna(data['Z_Score'].iloc[-1]):
                z_value = data['Z_Score'].iloc[-1]
                if z_value < lower_threshold:
                    current_signal = f"BUY (Z-Score={z_value:.2f} < {lower_threshold})"
                elif z_value > upper_threshold:
                    current_signal = f"SELL (Z-Score={z_value:.2f} > {upper_threshold})"
                else:
                    current_signal = f"HOLD (Z-Score={z_value:.2f})"
            else:
                current_signal = "HOLD (Insufficient Data)"
        
        elif method == "combined":
            # Check all three indicators
            bollinger_buy = pd.notna(data['Lower_Band'].iloc[-1]) and data['Close'].iloc[-1] <= data['Lower_Band'].iloc[-1]
            bollinger_sell = pd.notna(data['Upper_Band'].iloc[-1]) and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]
            
            rsi_buy = pd.notna(data['RSI'].iloc[-1]) and data['RSI'].iloc[-1] < rsi_oversold
            rsi_sell = pd.notna(data['RSI'].iloc[-1]) and data['RSI'].iloc[-1] > rsi_overbought
            
            zscore_buy = pd.notna(data['Z_Score'].iloc[-1]) and data['Z_Score'].iloc[-1] < lower_threshold
            zscore_sell = pd.notna(data['Z_Score'].iloc[-1]) and data['Z_Score'].iloc[-1] > upper_threshold
            
            buy_count = sum([bollinger_buy, rsi_buy, zscore_buy])
            sell_count = sum([bollinger_sell, rsi_sell, zscore_sell])
            
            if buy_count >= 2:
                current_signal = f"BUY ({buy_count}/3 indicators agree)"
            elif sell_count >= 2:
                current_signal = f"SELL ({sell_count}/3 indicators agree)"
            else:
                current_signal = "HOLD (No clear consensus)"
        
        # Prepare return dictionary
        result = {
            'symbol': symbol,
            'data': data,
            'signals': signals,
            'current_signal': current_signal,
            'current_price': float(current_price),
            'method': method,
            'total_signals': len(signals),
            'buy_signals': len(signals[signals['Signal'] > 0]) if len(signals) > 0 else 0,
            'sell_signals': len(signals[signals['Signal'] < 0]) if len(signals) > 0 else 0
        }
        
        # Add current indicator values
        if method == "bollinger" or method == "combined":
            result['upper_band'] = float(data['Upper_Band'].iloc[-1]) if pd.notna(data['Upper_Band'].iloc[-1]) else None
            result['middle_band'] = float(data['Middle_Band'].iloc[-1]) if pd.notna(data['Middle_Band'].iloc[-1]) else None
            result['lower_band'] = float(data['Lower_Band'].iloc[-1]) if pd.notna(data['Lower_Band'].iloc[-1]) else None
        
        if method == "rsi" or method == "combined":
            result['rsi'] = float(data['RSI'].iloc[-1]) if pd.notna(data['RSI'].iloc[-1]) else None
        
        if method == "zscore" or method == "combined":
            result['z_score'] = float(data['Z_Score'].iloc[-1]) if pd.notna(data['Z_Score'].iloc[-1]) else None
        
        return result
    
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
    
    def _process_single_ticker_mean_reversion(
        self,
        ticker: str,
        method: str,
        window: int,
        period: str,
        interval: str,
        upper_threshold: float,
        lower_threshold: float,
        rsi_overbought: float,
        rsi_oversold: float
    ) -> Dict[str, Optional[float]]:
        """
        Process a single ticker and return its mean reversion summary.

        Args:
            ticker (str): Ticker symbol to process.
            method (str): Method to use - "bollinger", "rsi", "zscore", or "combined"
            window (int): Window size for calculations
            period (str): Historical data period
            interval (str): Historical data interval
            upper_threshold (float): Upper threshold for Z-score/Bollinger
            lower_threshold (float): Lower threshold for Z-score/Bollinger
            rsi_overbought (float): RSI overbought threshold
            rsi_oversold (float): RSI oversold threshold

        Returns:
            Dict: Dictionary containing ticker mean reversion summary.
        """
        try:
            result = self.mean_reversion_signal(
                symbol=ticker,
                method=method,
                window=window,
                period=period,
                interval=interval,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                rsi_overbought=rsi_overbought,
                rsi_oversold=rsi_oversold
            )
            
            data_frame = result.get("data")
            last_close_price = (
                float(data_frame["Close"].iloc[-1])
                if data_frame is not None and not data_frame.empty
                else result.get("current_price")
            )
            
            # Build result dictionary
            ticker_result = {
                "Ticker": ticker,
                "Last Close Price": last_close_price,
                "Current Price": result.get("current_price"),
                "Signal": result.get("current_signal"),
                "Method": method
            }
            
            # Add method-specific indicators
            if method == "bollinger" or method == "combined":
                ticker_result["Upper_Band"] = result.get("upper_band")
                ticker_result["Middle_Band"] = result.get("middle_band")
                ticker_result["Lower_Band"] = result.get("lower_band")
            
            if method == "rsi" or method == "combined":
                ticker_result["RSI"] = result.get("rsi")
            
            if method == "zscore" or method == "combined":
                ticker_result["Z_Score"] = result.get("z_score")
            
            return ticker_result
        except Exception as exc:
            # Return error result
            error_result = {
                "Ticker": ticker,
                "Last Close Price": np.nan,
                "Current Price": np.nan,
                "Signal": f"ERROR: {str(exc)[:100]}",
                "Method": method
            }
            
            # Add empty indicator fields
            if method == "bollinger" or method == "combined":
                error_result["Upper_Band"] = np.nan
                error_result["Middle_Band"] = np.nan
                error_result["Lower_Band"] = np.nan
            
            if method == "rsi" or method == "combined":
                error_result["RSI"] = np.nan
            
            if method == "zscore" or method == "combined":
                error_result["Z_Score"] = np.nan
            
            return error_result
    
    def _generate_mean_reversion_summary_for_tickers(
        self,
        tickers: Sequence[str],
        method: str,
        window: int,
        period: str,
        interval: str,
        csv_path: str,
        upper_threshold: float = 2.0,
        lower_threshold: float = -2.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        max_workers: int = 20
    ) -> pd.DataFrame:
        """
        Generate mean reversion summary for a sequence of tickers using parallel processing.

        Args:
            tickers (Sequence[str]): Iterable of ticker symbols.
            method (str): Method to use - "bollinger", "rsi", "zscore", or "combined"
            window (int): Window size for calculations
            period (str): Historical data period
            interval (str): Historical data interval
            csv_path (str): Path to write the resulting CSV
            upper_threshold (float): Upper threshold for Z-score/Bollinger
            lower_threshold (float): Lower threshold for Z-score/Bollinger
            rsi_overbought (float): RSI overbought threshold
            rsi_oversold (float): RSI oversold threshold
            max_workers (int): Maximum number of concurrent workers (default: 20)

        Returns:
            pd.DataFrame: Summary DataFrame containing mean reversion signals and metrics.
        """
        results: List[Dict[str, Optional[float]]] = []
        total_tickers = len(tickers)
        
        print(f"Processing {total_tickers} tickers with {max_workers} concurrent workers...")
        
        # Process tickers in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    self._process_single_ticker_mean_reversion,
                    ticker,
                    method,
                    window,
                    period,
                    interval,
                    upper_threshold,
                    lower_threshold,
                    rsi_overbought,
                    rsi_oversold
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
                    error_result = {
                        "Ticker": ticker,
                        "Last Close Price": np.nan,
                        "Current Price": np.nan,
                        "Signal": f"ERROR: {str(exc)[:100]}",
                        "Method": method
                    }
                    if method == "bollinger" or method == "combined":
                        error_result["Upper_Band"] = np.nan
                        error_result["Middle_Band"] = np.nan
                        error_result["Lower_Band"] = np.nan
                    if method == "rsi" or method == "combined":
                        error_result["RSI"] = np.nan
                    if method == "zscore" or method == "combined":
                        error_result["Z_Score"] = np.nan
                    results.append(error_result)
                    completed += 1

        # Sort results by ticker to maintain consistent order
        results.sort(key=lambda x: x["Ticker"])
        
        # Build columns list based on method
        columns = ["Ticker", "Last Close Price", "Current Price", "Signal", "Method"]
        if method == "bollinger" or method == "combined":
            columns.extend(["Upper_Band", "Middle_Band", "Lower_Band"])
        if method == "rsi" or method == "combined":
            columns.append("RSI")
        if method == "zscore" or method == "combined":
            columns.append("Z_Score")
        
        df = pd.DataFrame(results, columns=columns)
        
        # Try to save CSV, but don't fail if it doesn't work
        try:
            df.to_csv(csv_path, index=False)
            print(f"Completed! Results saved to {csv_path}")
        except Exception as csv_error:
            print(f"Warning: Could not save CSV to {csv_path}: {csv_error}", file=sys.stderr)
            print("Results are still available in the returned DataFrame", file=sys.stderr)
        
        return df
    
    def mean_reversion_sp500(
        self,
        method: str = "bollinger",
        window: int = 20,
        period: str = "1y",
        interval: str = "1d",
        csv_path: str = "sp500_mean_reversion_signals.csv",
        upper_threshold: float = 2.0,
        lower_threshold: float = -2.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        max_workers: int = 20
    ) -> pd.DataFrame:
        """
        Calculate mean reversion signals for all S&P 500 stocks.

        Args:
            method (str): Method to use - "bollinger", "rsi", "zscore", or "combined" (default: "bollinger")
            window (int): Window size for calculations (default: 20)
            period (str): Historical data period (default: "1y")
            interval (str): Data interval (default: "1d")
            csv_path (str): File path to export the results CSV (default: "sp500_mean_reversion_signals.csv")
            upper_threshold (float): Upper threshold for Z-score/Bollinger (default: 2.0)
            lower_threshold (float): Lower threshold for Z-score/Bollinger (default: -2.0)
            rsi_overbought (float): RSI overbought threshold (default: 70.0)
            rsi_oversold (float): RSI oversold threshold (default: 30.0)
            max_workers (int): Maximum number of concurrent workers for parallel processing (default: 20)

        Returns:
            pd.DataFrame: DataFrame containing mean reversion results for each S&P 500 stock
        """
        tickers = self._fetch_constituent_tickers(
            url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            symbol_candidates=("symbol", "ticker symbol", "ticker", "root")
        )
        return self._generate_mean_reversion_summary_for_tickers(
            tickers=tickers,
            method=method,
            window=window,
            period=period,
            interval=interval,
            csv_path=csv_path,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            max_workers=max_workers
        )
    
    def mean_reversion_nasdaq100(
        self,
        method: str = "bollinger",
        window: int = 20,
        period: str = "1y",
        interval: str = "1d",
        csv_path: str = "nasdaq100_mean_reversion_signals.csv",
        upper_threshold: float = 2.0,
        lower_threshold: float = -2.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        max_workers: int = 20
    ) -> pd.DataFrame:
        """
        Calculate mean reversion signals for all NASDAQ-100 stocks.

        Args:
            method (str): Method to use - "bollinger", "rsi", "zscore", or "combined" (default: "bollinger")
            window (int): Window size for calculations (default: 20)
            period (str): Historical data period (default: "1y")
            interval (str): Data interval (default: "1d")
            csv_path (str): File path to export the results CSV (default: "nasdaq100_mean_reversion_signals.csv")
            upper_threshold (float): Upper threshold for Z-score/Bollinger (default: 2.0)
            lower_threshold (float): Lower threshold for Z-score/Bollinger (default: -2.0)
            rsi_overbought (float): RSI overbought threshold (default: 70.0)
            rsi_oversold (float): RSI oversold threshold (default: 30.0)
            max_workers (int): Maximum number of concurrent workers for parallel processing (default: 20)

        Returns:
            pd.DataFrame: DataFrame containing mean reversion results for each NASDAQ-100 stock
        """
        tickers = self._fetch_constituent_tickers(
            url="https://en.wikipedia.org/wiki/NASDAQ-100",
            symbol_candidates=("ticker", "ticker symbol", "symbol"),
            table_indices=(3, 4, 5)  # the constituents table typically appears later on the page
        )
        return self._generate_mean_reversion_summary_for_tickers(
            tickers=tickers,
            method=method,
            window=window,
            period=period,
            interval=interval,
            csv_path=csv_path,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            max_workers=max_workers
        )


# Example usage
if __name__ == "__main__":
    # Create an instance
    strategy = MeanReversionStrategy(symbol="AAPL")
    
    # Example 1: Bollinger Bands Mean Reversion
    print("="*60)
    print("Mean Reversion Strategy - Bollinger Bands - AAPL")
    print("="*60)
    
    result = strategy.mean_reversion_signal(method="bollinger", window=20)
    
    print(f"\nSymbol: {result['symbol']}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Upper Band: ${result.get('upper_band', 'N/A'):.2f}" if result.get('upper_band') else f"Upper Band: N/A")
    print(f"Middle Band: ${result.get('middle_band', 'N/A'):.2f}" if result.get('middle_band') else f"Middle Band: N/A")
    print(f"Lower Band: ${result.get('lower_band', 'N/A'):.2f}" if result.get('lower_band') else f"Lower Band: N/A")
    print(f"Current Signal: {result['current_signal']}")
    print(f"Total Signals: {result['total_signals']}")
    print(f"Buy Signals: {result['buy_signals']}")
    print(f"Sell Signals: {result['sell_signals']}")
    
    # Example 2: RSI Mean Reversion
    print("\n" + "="*60)
    print("Mean Reversion Strategy - RSI - AAPL")
    print("="*60)
    
    result_rsi = strategy.mean_reversion_signal(method="rsi", window=14)
    
    print(f"\nSymbol: {result_rsi['symbol']}")
    print(f"Current Price: ${result_rsi['current_price']:.2f}")
    print(f"RSI: {result_rsi.get('rsi', 'N/A'):.2f}" if result_rsi.get('rsi') else f"RSI: N/A")
    print(f"Current Signal: {result_rsi['current_signal']}")
    
    # Example 3: Combined Method
    print("\n" + "="*60)
    print("Mean Reversion Strategy - Combined - AAPL")
    print("="*60)
    
    result_combined = strategy.mean_reversion_signal(method="combined", window=20)
    
    print(f"\nSymbol: {result_combined['symbol']}")
    print(f"Current Price: ${result_combined['current_price']:.2f}")
    print(f"Current Signal: {result_combined['current_signal']}")
    if result_combined.get('rsi'):
        print(f"RSI: {result_combined['rsi']:.2f}")
    if result_combined.get('z_score'):
        print(f"Z-Score: {result_combined['z_score']:.2f}")
    
    # Example 4: Generate S&P 500 mean reversion summary and export to CSV
    print("\n" + "="*60)
    print("S&P 500 Mean Reversion Summary")
    print("="*60)
    
    sp500_df = strategy.mean_reversion_sp500(
        method="bollinger",
        window=20,
        period="1y",
        interval="1d",
        csv_path="sp500_mean_reversion_signals.csv"
    )
    
    print(f"\nGenerated mean reversion summary for {len(sp500_df)} S&P 500 tickers.")
    print("Saved results to sp500_mean_reversion_signals.csv")
    
    # Example 5: Generate NASDAQ-100 mean reversion summary and export to CSV
    print("\n" + "="*60)
    print("NASDAQ-100 Mean Reversion Summary")
    print("="*60)
    
    nasdaq_df = strategy.mean_reversion_nasdaq100(
        method="bollinger",
        window=20,
        period="1y",
        interval="1d",
        csv_path="nasdaq100_mean_reversion_signals.csv"
    )
    
    print(f"\nGenerated mean reversion summary for {len(nasdaq_df)} NASDAQ-100 tickers.")
    print("Saved results to nasdaq100_mean_reversion_signals.csv")

