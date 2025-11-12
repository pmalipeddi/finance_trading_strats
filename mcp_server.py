"""
MCP Server for Trading Strategies - Moving Average Crossover & Mean Reversion
Exposes trading strategy tools via Model Context Protocol
"""
import asyncio
import sys
from typing import Any, Sequence

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError as e:
    print(f"Error: MCP SDK not installed. Please install it with: pip install mcp", file=sys.stderr)
    sys.exit(1)

from moving_average_crossover_strategy import TradingStrategies
from mean_reversion_strategy import MeanReversionStrategy

# Initialize the trading strategies instances
strategies = TradingStrategies()
mean_reversion = MeanReversionStrategy()

# Create MCP server instance
server = Server("trading-strategies-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available tools for the MCP server.
    """
    return [
        Tool(
            name="get_moving_average_crossover",
            description=(
                "Get moving average crossover signal for a stock ticker. "
                "Returns BUY signal when fast MA crosses above slow MA (Golden Cross), "
                "SELL signal when fast MA crosses below slow MA (Death Cross), "
                "or HOLD signal otherwise."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
                    },
                    "fast_period": {
                        "type": "integer",
                        "description": "Period for fast moving average (default: 50)",
                        "default": 50
                    },
                    "slow_period": {
                        "type": "integer",
                        "description": "Period for slow moving average (default: 200)",
                        "default": 200
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 2y)",
                        "default": "2y"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo (default: 1d)",
                        "default": "1d"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_sp500_crossover_signals",
            description=(
                "Get moving average crossover signals for all S&P 500 stocks. "
                "Processes all tickers in parallel and returns summary statistics. "
                "Results are also saved to CSV file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fast_period": {
                        "type": "integer",
                        "description": "Period for fast moving average (default: 50)",
                        "default": 50
                    },
                    "slow_period": {
                        "type": "integer",
                        "description": "Period for slow moving average (default: 200)",
                        "default": 200
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical data period (default: 2y)",
                        "default": "2y"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval (default: 1d)",
                        "default": "1d"
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Number of concurrent workers for parallel processing (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_nasdaq100_crossover_signals",
            description=(
                "Get moving average crossover signals for all NASDAQ-100 stocks. "
                "Processes all tickers in parallel and returns summary statistics. "
                "Results are also saved to CSV file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "fast_period": {
                        "type": "integer",
                        "description": "Period for fast moving average (default: 50)",
                        "default": 50
                    },
                    "slow_period": {
                        "type": "integer",
                        "description": "Period for slow moving average (default: 200)",
                        "default": 200
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical data period (default: 2y)",
                        "default": "2y"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval (default: 1d)",
                        "default": "1d"
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Number of concurrent workers for parallel processing (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_mean_reversion_signal",
            description=(
                "Get mean reversion signal for a stock ticker using Bollinger Bands, RSI, Z-Score, or Combined method. "
                "Returns BUY signal when price is oversold (below mean), "
                "SELL signal when price is overbought (above mean), "
                "or HOLD signal otherwise. Window parameter allows short (20), medium (50), or long (100) term analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
                    },
                    "method": {
                        "type": "string",
                        "description": "Method: bollinger, rsi, zscore, or combined (default: bollinger)",
                        "default": "bollinger",
                        "enum": ["bollinger", "rsi", "zscore", "combined"]
                    },
                    "window": {
                        "type": "integer",
                        "description": "Window size: 20=short term, 50=medium term, 100=long term (default: 20)",
                        "default": 20
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)",
                        "default": "1y"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo (default: 1d)",
                        "default": "1d"
                    },
                    "upper_threshold": {
                        "type": "number",
                        "description": "Upper threshold for Z-score/Bollinger (default: 2.0)",
                        "default": 2.0
                    },
                    "lower_threshold": {
                        "type": "number",
                        "description": "Lower threshold for Z-score/Bollinger (default: -2.0)",
                        "default": -2.0
                    },
                    "rsi_overbought": {
                        "type": "number",
                        "description": "RSI overbought threshold (default: 70.0)",
                        "default": 70.0
                    },
                    "rsi_oversold": {
                        "type": "number",
                        "description": "RSI oversold threshold (default: 30.0)",
                        "default": 30.0
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_sp500_mean_reversion_signals",
            description=(
                "Get mean reversion signals for all S&P 500 stocks. "
                "Processes all tickers in parallel and returns summary statistics. "
                "Results are also saved to CSV file. Window parameter allows short (20), medium (50), or long (100) term analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Method: bollinger, rsi, zscore, or combined (default: bollinger)",
                        "default": "bollinger",
                        "enum": ["bollinger", "rsi", "zscore", "combined"]
                    },
                    "window": {
                        "type": "integer",
                        "description": "Window size: 20=short term, 50=medium term, 100=long term (default: 20)",
                        "default": 20
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical data period (default: 1y)",
                        "default": "1y"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval (default: 1d)",
                        "default": "1d"
                    },
                    "upper_threshold": {
                        "type": "number",
                        "description": "Upper threshold for Z-score/Bollinger (default: 2.0)",
                        "default": 2.0
                    },
                    "lower_threshold": {
                        "type": "number",
                        "description": "Lower threshold for Z-score/Bollinger (default: -2.0)",
                        "default": -2.0
                    },
                    "rsi_overbought": {
                        "type": "number",
                        "description": "RSI overbought threshold (default: 70.0)",
                        "default": 70.0
                    },
                    "rsi_oversold": {
                        "type": "number",
                        "description": "RSI oversold threshold (default: 30.0)",
                        "default": 30.0
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Number of concurrent workers for parallel processing (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_nasdaq100_mean_reversion_signals",
            description=(
                "Get mean reversion signals for all NASDAQ-100 stocks. "
                "Processes all tickers in parallel and returns summary statistics. "
                "Results are also saved to CSV file. Window parameter allows short (20), medium (50), or long (100) term analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Method: bollinger, rsi, zscore, or combined (default: bollinger)",
                        "default": "bollinger",
                        "enum": ["bollinger", "rsi", "zscore", "combined"]
                    },
                    "window": {
                        "type": "integer",
                        "description": "Window size: 20=short term, 50=medium term, 100=long term (default: 20)",
                        "default": 20
                    },
                    "period": {
                        "type": "string",
                        "description": "Historical data period (default: 1y)",
                        "default": "1y"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval (default: 1d)",
                        "default": "1d"
                    },
                    "upper_threshold": {
                        "type": "number",
                        "description": "Upper threshold for Z-score/Bollinger (default: 2.0)",
                        "default": 2.0
                    },
                    "lower_threshold": {
                        "type": "number",
                        "description": "Lower threshold for Z-score/Bollinger (default: -2.0)",
                        "default": -2.0
                    },
                    "rsi_overbought": {
                        "type": "number",
                        "description": "RSI overbought threshold (default: 70.0)",
                        "default": 70.0
                    },
                    "rsi_oversold": {
                        "type": "number",
                        "description": "RSI oversold threshold (default: 30.0)",
                        "default": 30.0
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Number of concurrent workers for parallel processing (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """
    Handle tool calls from MCP clients.
    """
    try:
        if name == "get_moving_average_crossover":
            symbol = arguments.get("symbol", "").upper()
            fast_period = arguments.get("fast_period", 50)
            slow_period = arguments.get("slow_period", 200)
            period = arguments.get("period", "2y")
            interval = arguments.get("interval", "1d")
            
            if not symbol:
                return [TextContent(
                    type="text",
                    text="Error: Symbol is required"
                )]
            
            # Get crossover signal
            result = strategies.moving_average_crossover(
                symbol=symbol,
                fast_period=fast_period,
                slow_period=slow_period,
                period=period,
                interval=interval
            )
            
            # Format response
            fast_ma_str = f"${result.get('fast_ma'):.2f}" if result.get('fast_ma') is not None else "N/A"
            slow_ma_str = f"${result.get('slow_ma'):.2f}" if result.get('slow_ma') is not None else "N/A"
            
            response = f"""Moving Average Crossover Signal for {result['symbol']}

Current Price: ${result['current_price']:.2f}
Fast MA ({result['fast_period']}-day): {fast_ma_str}
Slow MA ({result['slow_period']}-day): {slow_ma_str}
Signal: {result['current_signal']}

Statistics:
- Total Crossover Signals: {result['total_signals']}
- Buy Signals: {result['buy_signals']}
- Sell Signals: {result['sell_signals']}
"""
            
            return [TextContent(type="text", text=response)]
        
        elif name == "get_sp500_crossover_signals":
            fast_period = arguments.get("fast_period", 50)
            slow_period = arguments.get("slow_period", 200)
            period = arguments.get("period", "2y")
            interval = arguments.get("interval", "1d")
            max_workers = arguments.get("max_workers", 20)
            
            try:
                # Use absolute path for CSV file
                import os
                project_root = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(project_root, "sp500_moving_average_signals.csv")
                
                # Get S&P 500 signals (CSV saving is optional - will continue even if it fails)
                df = strategies.moving_average_crossover_sp500(
                    fast_period=fast_period,
                    slow_period=slow_period,
                    period=period,
                    interval=interval,
                    csv_path=csv_path,
                    max_workers=max_workers
                )
                
                # Get signal distribution
                signal_counts = df['Signal'].value_counts().to_dict()
                
                # Format response with data
                response = f"""S&P 500 Moving Average Crossover Signals

Total Tickers Processed: {len(df)}
Signal Distribution:
"""
                for signal, count in signal_counts.items():
                    response += f"  - {signal}: {count}\n"
                
                # Add top stocks by signal type
                response += "\n\nSample Results (first 20 stocks):\n"
                response += "=" * 80 + "\n"
                for idx, row in df.head(20).iterrows():
                    response += f"{row['Ticker']:6s} | Price: ${row['Current Price']:8.2f} | Signal: {row['Signal']}\n"
                
                if len(df) > 20:
                    response += f"\n... and {len(df) - 20} more stocks\n"
                
                response += "\n" + "=" * 80 + "\n"
                response += f"\nFull results available in DataFrame. CSV save attempted to: {csv_path}"
                
                return [TextContent(type="text", text=response)]
            except Exception as e:
                error_msg = str(e)
                import traceback
                error_details = traceback.format_exc()
                print(f"MCP Server Error: {error_msg}\n{error_details}", file=sys.stderr)
                return [TextContent(
                    type="text",
                    text=f"Error processing S&P 500 signals: {error_msg}\n\nThis might take a few minutes. Try again or use the single-stock tool for specific tickers."
                )]
        
        elif name == "get_nasdaq100_crossover_signals":
            fast_period = arguments.get("fast_period", 50)
            slow_period = arguments.get("slow_period", 200)
            period = arguments.get("period", "2y")
            interval = arguments.get("interval", "1d")
            max_workers = arguments.get("max_workers", 20)
            
            try:
                # Use absolute path for CSV file
                import os
                project_root = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(project_root, "nasdaq100_moving_average_signals.csv")
                
                # Get NASDAQ-100 signals (CSV saving is optional - will continue even if it fails)
                df = strategies.moving_average_crossover_nasdaq100(
                    fast_period=fast_period,
                    slow_period=slow_period,
                    period=period,
                    interval=interval,
                    csv_path=csv_path,
                    max_workers=max_workers
                )
                
                # Get signal distribution
                signal_counts = df['Signal'].value_counts().to_dict()
                
                # Format response with data
                response = f"""NASDAQ-100 Moving Average Crossover Signals

Total Tickers Processed: {len(df)}
Signal Distribution:
"""
                for signal, count in signal_counts.items():
                    response += f"  - {signal}: {count}\n"
                
                # Add top stocks by signal type
                response += "\n\nSample Results (first 20 stocks):\n"
                response += "=" * 80 + "\n"
                for idx, row in df.head(20).iterrows():
                    response += f"{row['Ticker']:6s} | Price: ${row['Current Price']:8.2f} | Signal: {row['Signal']}\n"
                
                if len(df) > 20:
                    response += f"\n... and {len(df) - 20} more stocks\n"
                
                response += "\n" + "=" * 80 + "\n"
                response += f"\nFull results available in DataFrame. CSV save attempted to: {csv_path}"
                
                return [TextContent(type="text", text=response)]
            except Exception as e:
                error_msg = str(e)
                import traceback
                error_details = traceback.format_exc()
                print(f"MCP Server Error: {error_msg}\n{error_details}", file=sys.stderr)
                return [TextContent(
                    type="text",
                    text=f"Error processing NASDAQ-100 signals: {error_msg}\n\nThis might take a few minutes. Try again or use the single-stock tool for specific tickers."
                )]
        
        elif name == "get_mean_reversion_signal":
            symbol = arguments.get("symbol", "").upper()
            method = arguments.get("method", "bollinger")
            window = arguments.get("window", 20)
            period = arguments.get("period", "1y")
            interval = arguments.get("interval", "1d")
            upper_threshold = arguments.get("upper_threshold", 2.0)
            lower_threshold = arguments.get("lower_threshold", -2.0)
            rsi_overbought = arguments.get("rsi_overbought", 70.0)
            rsi_oversold = arguments.get("rsi_oversold", 30.0)
            
            if not symbol:
                return [TextContent(
                    type="text",
                    text="Error: Symbol is required"
                )]
            
            if method not in ["bollinger", "rsi", "zscore", "combined"]:
                return [TextContent(
                    type="text",
                    text=f"Error: Invalid method '{method}'. Must be one of: bollinger, rsi, zscore, combined"
                )]
            
            try:
                # Get mean reversion signal
                result = mean_reversion.mean_reversion_signal(
                    symbol=symbol,
                    method=method,
                    window=window,
                    period=period,
                    interval=interval,
                    upper_threshold=upper_threshold,
                    lower_threshold=lower_threshold,
                    rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold
                )
                
                # Format response based on method
                response = f"""Mean Reversion Signal for {result['symbol']} ({method.upper()} method, {window}-day window)

Current Price: ${result['current_price']:.2f}
Signal: {result['current_signal']}

Statistics:
- Total Signals: {result['total_signals']}
- Buy Signals: {result['buy_signals']}
- Sell Signals: {result['sell_signals']}
"""
                
                # Add method-specific indicators
                if method == "bollinger" or method == "combined":
                    upper_str = f"${result.get('upper_band'):.2f}" if result.get('upper_band') is not None else "N/A"
                    middle_str = f"${result.get('middle_band'):.2f}" if result.get('middle_band') is not None else "N/A"
                    lower_str = f"${result.get('lower_band'):.2f}" if result.get('lower_band') is not None else "N/A"
                    response += f"\nBollinger Bands:\n"
                    response += f"  Upper Band: {upper_str}\n"
                    response += f"  Middle Band: {middle_str}\n"
                    response += f"  Lower Band: {lower_str}\n"
                
                if method == "rsi" or method == "combined":
                    rsi_str = f"{result.get('rsi'):.2f}" if result.get('rsi') is not None else "N/A"
                    response += f"\nRSI: {rsi_str}\n"
                
                if method == "zscore" or method == "combined":
                    zscore_str = f"{result.get('z_score'):.2f}" if result.get('z_score') is not None else "N/A"
                    response += f"\nZ-Score: {zscore_str}\n"
                
                return [TextContent(type="text", text=response)]
            except Exception as e:
                error_msg = str(e)
                import traceback
                error_details = traceback.format_exc()
                print(f"MCP Server Error: {error_msg}\n{error_details}", file=sys.stderr)
                return [TextContent(
                    type="text",
                    text=f"Error processing mean reversion signal: {error_msg}"
                )]
        
        elif name == "get_sp500_mean_reversion_signals":
            method = arguments.get("method", "bollinger")
            window = arguments.get("window", 20)
            period = arguments.get("period", "1y")
            interval = arguments.get("interval", "1d")
            upper_threshold = arguments.get("upper_threshold", 2.0)
            lower_threshold = arguments.get("lower_threshold", -2.0)
            rsi_overbought = arguments.get("rsi_overbought", 70.0)
            rsi_oversold = arguments.get("rsi_oversold", 30.0)
            max_workers = arguments.get("max_workers", 20)
            
            if method not in ["bollinger", "rsi", "zscore", "combined"]:
                return [TextContent(
                    type="text",
                    text=f"Error: Invalid method '{method}'. Must be one of: bollinger, rsi, zscore, combined"
                )]
            
            try:
                # Use absolute path for CSV file
                import os
                project_root = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(project_root, f"sp500_mean_reversion_signals_{method}_w{window}.csv")
                
                # Get S&P 500 mean reversion signals
                df = mean_reversion.mean_reversion_sp500(
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
                
                # Get signal distribution
                signal_counts = df['Signal'].value_counts().to_dict()
                
                # Format response with data
                response = f"""S&P 500 Mean Reversion Signals ({method.upper()} method, {window}-day window)

Total Tickers Processed: {len(df)}
Signal Distribution:
"""
                for signal, count in signal_counts.items():
                    response += f"  - {signal}: {count}\n"
                
                # Add top stocks by signal type
                response += "\n\nSample Results (first 20 stocks):\n"
                response += "=" * 80 + "\n"
                for idx, row in df.head(20).iterrows():
                    response += f"{row['Ticker']:6s} | Price: ${row['Current Price']:8.2f} | Signal: {row['Signal']}\n"
                
                if len(df) > 20:
                    response += f"\n... and {len(df) - 20} more stocks\n"
                
                response += "\n" + "=" * 80 + "\n"
                response += f"\nFull results available in DataFrame. CSV save attempted to: {csv_path}"
                
                return [TextContent(type="text", text=response)]
            except Exception as e:
                error_msg = str(e)
                import traceback
                error_details = traceback.format_exc()
                print(f"MCP Server Error: {error_msg}\n{error_details}", file=sys.stderr)
                return [TextContent(
                    type="text",
                    text=f"Error processing S&P 500 mean reversion signals: {error_msg}\n\nThis might take a few minutes. Try again or use the single-stock tool for specific tickers."
                )]
        
        elif name == "get_nasdaq100_mean_reversion_signals":
            method = arguments.get("method", "bollinger")
            window = arguments.get("window", 20)
            period = arguments.get("period", "1y")
            interval = arguments.get("interval", "1d")
            upper_threshold = arguments.get("upper_threshold", 2.0)
            lower_threshold = arguments.get("lower_threshold", -2.0)
            rsi_overbought = arguments.get("rsi_overbought", 70.0)
            rsi_oversold = arguments.get("rsi_oversold", 30.0)
            max_workers = arguments.get("max_workers", 20)
            
            if method not in ["bollinger", "rsi", "zscore", "combined"]:
                return [TextContent(
                    type="text",
                    text=f"Error: Invalid method '{method}'. Must be one of: bollinger, rsi, zscore, combined"
                )]
            
            try:
                # Use absolute path for CSV file
                import os
                project_root = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(project_root, f"nasdaq100_mean_reversion_signals_{method}_w{window}.csv")
                
                # Get NASDAQ-100 mean reversion signals
                df = mean_reversion.mean_reversion_nasdaq100(
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
                
                # Get signal distribution
                signal_counts = df['Signal'].value_counts().to_dict()
                
                # Format response with data
                response = f"""NASDAQ-100 Mean Reversion Signals ({method.upper()} method, {window}-day window)

Total Tickers Processed: {len(df)}
Signal Distribution:
"""
                for signal, count in signal_counts.items():
                    response += f"  - {signal}: {count}\n"
                
                # Add top stocks by signal type
                response += "\n\nSample Results (first 20 stocks):\n"
                response += "=" * 80 + "\n"
                for idx, row in df.head(20).iterrows():
                    response += f"{row['Ticker']:6s} | Price: ${row['Current Price']:8.2f} | Signal: {row['Signal']}\n"
                
                if len(df) > 20:
                    response += f"\n... and {len(df) - 20} more stocks\n"
                
                response += "\n" + "=" * 80 + "\n"
                response += f"\nFull results available in DataFrame. CSV save attempted to: {csv_path}"
                
                return [TextContent(type="text", text=response)]
            except Exception as e:
                error_msg = str(e)
                import traceback
                error_details = traceback.format_exc()
                print(f"MCP Server Error: {error_msg}\n{error_details}", file=sys.stderr)
                return [TextContent(
                    type="text",
                    text=f"Error processing NASDAQ-100 mean reversion signals: {error_msg}\n\nThis might take a few minutes. Try again or use the single-stock tool for specific tickers."
                )]
        
        else:
            return [TextContent(
                type="text",
                text=f"Error: Unknown tool '{name}'"
            )]
    
    except ValueError as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error processing request: {str(e)}"
        )]


async def main():
    """
    Main entry point for the MCP server.
    """
    # Print startup message to stderr (stdout is used for MCP protocol)
    print("MCP Server: Trading Strategies - Starting...", file=sys.stderr)
    print("MCP Server: Waiting for MCP client connection...", file=sys.stderr)
    print("MCP Server: Server is ready and listening on stdio", file=sys.stderr)
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        print("MCP Server: Connected to MCP client", file=sys.stderr)
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("=" * 60, file=sys.stderr)
        print("MCP Server Test Mode", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("", file=sys.stderr)
        print("✓ Server file syntax: OK", file=sys.stderr)
        
        # Test imports
        try:
            from mcp.server import Server
            from mcp.server.stdio import stdio_server
            from mcp.types import Tool, TextContent
            print("✓ MCP SDK imports: OK", file=sys.stderr)
        except ImportError as e:
            print(f"✗ MCP SDK imports: FAILED - {e}", file=sys.stderr)
            sys.exit(1)
        
        try:
            from moving_average_crossover_strategy import TradingStrategies
            print("✓ Moving average crossover strategy import: OK", file=sys.stderr)
        except ImportError as e:
            print(f"✗ Moving average crossover strategy import: FAILED - {e}", file=sys.stderr)
            sys.exit(1)
        
        try:
            from mean_reversion_strategy import MeanReversionStrategy
            print("✓ Mean reversion strategy import: OK", file=sys.stderr)
        except ImportError as e:
            print(f"✗ Mean reversion strategy import: FAILED - {e}", file=sys.stderr)
            sys.exit(1)
        
        # Test initialization
        try:
            strategies = TradingStrategies()
            print("✓ Moving average crossover strategy initialization: OK", file=sys.stderr)
        except Exception as e:
            print(f"✗ Moving average crossover strategy initialization: FAILED - {e}", file=sys.stderr)
            sys.exit(1)
        
        try:
            mean_reversion = MeanReversionStrategy()
            print("✓ Mean reversion strategy initialization: OK", file=sys.stderr)
        except Exception as e:
            print(f"✗ Mean reversion strategy initialization: FAILED - {e}", file=sys.stderr)
            sys.exit(1)
        
        print("", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print("All checks passed! Server is ready to use.", file=sys.stderr)
        print("", file=sys.stderr)
        print("To use with Claude Desktop:", file=sys.stderr)
        print("1. Make sure config file is set up", file=sys.stderr)
        print("2. Restart Claude Desktop", file=sys.stderr)
        print("3. Ask Claude: 'What MCP tools do you have available?'", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(0)
    
    # Normal mode - run the server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCP Server: Shutting down...", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"MCP Server: Error - {e}", file=sys.stderr)
        sys.exit(1)

