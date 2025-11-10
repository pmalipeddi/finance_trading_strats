import yfinance as yf
from typing import Dict, Optional, List


class YahooFinanceAPI:
    """
    A class for interacting with Yahoo Finance API to get stock quotes and financial data.
    
    Args:
        symbol (str, optional): Single stock ticker symbol to store for later use
        symbols (List[str], optional): List of stock ticker symbols to store for later use
    """
    
    def __init__(self, symbol: Optional[str] = None, symbols: Optional[List[str]] = None):
        """
        Initialize the YahooFinanceAPI instance.
        
        Args:
            symbol (str, optional): Single stock ticker symbol (e.g., 'AAPL')
            symbols (List[str], optional): List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        """
        self.symbol = symbol.upper() if symbol else None
        self.symbols = [s.upper() for s in symbols] if symbols else None
        
        # If both are provided, prefer symbols list
        if symbols and symbol:
            self.symbol = None
    
    def get_quote(self, symbol: Optional[str] = None) -> Dict:
        """
        Get real-time quote for a stock symbol using Yahoo Finance API.
        
        Args:
            symbol (str, optional): Stock ticker symbol. If not provided, uses the symbol
                                    from initialization.
        
        Returns:
            dict: Quote information containing price, volume, market cap, etc.
                  Returns dict with 'error' key if an error occurs.
        
        Raises:
            ValueError: If no symbol is provided and no symbol was set during initialization.
        """
        # Use provided symbol or fall back to instance symbol
        if symbol is None:
            if self.symbol is None:
                raise ValueError("No symbol provided. Either pass a symbol to this method or initialize the class with a symbol.")
            symbol = self.symbol
        else:
            symbol = symbol.upper()
        
        ticker = yf.Ticker(symbol)
        
        try:
            # Get current info and fast info
            info = ticker.info
            fast_info = ticker.fast_info
            
            quote = {
                'symbol': symbol.upper(),
                'price': fast_info.get('lastPrice') or info.get('currentPrice'),
                'previous_close': fast_info.get('previousClose'),
                'open': fast_info.get('open'),
                'day_high': fast_info.get('dayHigh'),
                'day_low': fast_info.get('dayLow'),
                'volume': fast_info.get('volume'),
                'market_cap': info.get('marketCap'),
                'company_name': info.get('longName') or info.get('shortName'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'dividend_yield': info.get('dividendYield'),
                'pe_ratio': info.get('trailingPE'),
                'beta': info.get('beta')
            }
            
            return quote
        except Exception as e:
            return {'error': str(e), 'symbol': symbol}
    
    def get_quotes(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Get real-time quotes for multiple stock symbols.
        
        Args:
            symbols (List[str], optional): List of stock ticker symbols. If not provided,
                                           uses the symbols list from initialization.
        
        Returns:
            dict: Dictionary mapping symbol to quote information
        
        Raises:
            ValueError: If no symbols are provided and no symbols were set during initialization.
        """
        # Use provided symbols or fall back to instance symbols
        if symbols is None:
            if self.symbols is None:
                raise ValueError("No symbols provided. Either pass a symbols list to this method or initialize the class with symbols.")
            symbols = self.symbols
        else:
            symbols = [s.upper() for s in symbols]
        
        results = {}
        for symbol in symbols:
            results[symbol.upper()] = self.get_quote(symbol)
        return results
    
    def get_quote_simple(self, symbol: Optional[str] = None) -> Optional[float]:
        """
        Get just the current price for a stock symbol.
        
        Args:
            symbol (str, optional): Stock ticker symbol. If not provided, uses the symbol
                                    from initialization.
        
        Returns:
            float: Current price, or None if error
        
        Raises:
            ValueError: If no symbol is provided and no symbol was set during initialization.
        """
        quote = self.get_quote(symbol)
        if 'error' not in quote:
            return quote.get('price')
        return None


# Backward compatibility: Keep standalone functions that use the class
def get_quote(symbol: str) -> Dict:
    """
    Get real-time quote for a stock symbol using Yahoo Finance API.
    (Backward compatibility wrapper)
    """
    api = YahooFinanceAPI()
    return api.get_quote(symbol)


def get_quotes(symbols: List[str]) -> Dict[str, Dict]:
    """
    Get real-time quotes for multiple stock symbols.
    (Backward compatibility wrapper)
    """
    api = YahooFinanceAPI()
    return api.get_quotes(symbols)


def get_quote_simple(symbol: str) -> Optional[float]:
    """
    Get just the current price for a stock symbol.
    (Backward compatibility wrapper)
    """
    api = YahooFinanceAPI()
    return api.get_quote_simple(symbol)


# Example usage
if __name__ == "__main__":
    # Example 1: Initialize with no parameters - methods require symbols to be passed
    print("="*50)
    print("Example 1: Initialize with no parameters")
    print("="*50)
    api1 = YahooFinanceAPI()
    quote = api1.get_quote("PLUG")
    print(f"\nQuote for {quote.get('symbol')}:")
    print(f"Company: {quote.get('company_name')}")
    print(f"Price: ${quote.get('price')}")
    
    # Example 2: Initialize with a single symbol - can call methods without parameters
    print("\n" + "="*50)
    print("Example 2: Initialize with a single symbol")
    print("="*50)
    api2 = YahooFinanceAPI(symbol="MSFT")
    quote = api2.get_quote()  # Uses the symbol from initialization
    print(f"\nQuote for {quote.get('symbol')}:")
    print(f"Company: {quote.get('company_name')}")
    print(f"Price: ${quote.get('price')}")
    print(f"Previous Close: ${quote.get('previous_close')}")
    print(f"Day High: ${quote.get('day_high')}")
    print(f"Day Low: ${quote.get('day_low')}")
    print(f"Volume: {quote.get('volume'):,}" if quote.get('volume') else "Volume: N/A")
    print(f"Market Cap: ${quote.get('market_cap'):,}" if quote.get('market_cap') else "Market Cap: N/A")
    
    # Can still override with a different symbol
    price = api2.get_quote_simple("GOOGL")
    print(f"\nOverride example - GOOGL price: ${price}")
    
    # Example 3: Initialize with a list of symbols
    print("\n" + "="*50)
    print("Example 3: Initialize with a list of symbols")
    print("="*50)
    api3 = YahooFinanceAPI(symbols=["AAPL", "MSFT", "GOOGL"])
    quotes = api3.get_quotes()  # Uses the symbols from initialization
    for symbol, quote_data in quotes.items():
        if 'error' not in quote_data:
            print(f"{symbol}: ${quote_data.get('price')}")
        else:
            print(f"{symbol}: Error - {quote_data.get('error')}")
    
    # Example 4: Simple price lookup with stored symbol
    print("\n" + "="*50)
    print("Example 4: Simple price lookup with stored symbol")
    print("="*50)
    api4 = YahooFinanceAPI(symbol="TSLA")
    price = api4.get_quote_simple()  # Uses the symbol from initialization
    if price:
        print(f"TSLA current price: ${price}")
    else:
        print("Error getting TSLA price")
    
    # Example 5: Still works with explicit parameters
    print("\n" + "="*50)
    print("Example 5: Explicit parameters override stored symbols")
    print("="*50)
    api5 = YahooFinanceAPI(symbol="AAPL")
    quote = api5.get_quote("NVDA")  # Override with different symbol
    print(f"NVDA price: ${quote.get('price')}")
