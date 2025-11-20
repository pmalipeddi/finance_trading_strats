from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import sys

from mcp.server import Server
from mcp.types import Tool, TextContent

# Import strategy classes directly
from mean_reversion_strategy import MeanReversionStrategy
from moving_average_crossover_strategy import TradingStrategies
from momentum_price_vol_strategy import MomentumBreakoutStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP Server
app = Server("trading-strategies-mcp")

# Tool definitions
TOOLS: List[Tool] = [
    # Mean Reversion Strategy Tools
    Tool(
        name="backtest_mean_reversion",
        description="Backtest the mean reversion trading strategy on historical data. Returns performance metrics including total return, Sharpe ratio, max drawdown, and trade statistics.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "lookback_period": {"type": "integer", "description": "Lookback period for mean calculation (default: 20)"},
                "entry_threshold": {"type": "number", "description": "Standard deviations for entry (default: 2.0)"},
                "exit_threshold": {"type": "number", "description": "Standard deviations for exit (default: 0.5)"}
            },
            "required": ["symbol", "start_date", "end_date"]
        }
    ),
    Tool(
        name="optimize_mean_reversion",
        description="Optimize mean reversion strategy parameters to find the best combination for maximum returns or Sharpe ratio.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "optimization_metric": {"type": "string", "description": "Metric to optimize (return or sharpe, default: sharpe)"}
            },
            "required": ["symbol", "start_date", "end_date"]
        }
    ),
    Tool(
        name="get_mean_reversion_signals",
        description="Get current trading signals for the mean reversion strategy based on recent price data.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "lookback_period": {"type": "integer", "description": "Lookback period for mean calculation (default: 20)"},
                "entry_threshold": {"type": "number", "description": "Standard deviations for entry (default: 2.0)"},
                "exit_threshold": {"type": "number", "description": "Standard deviations for exit (default: 0.5)"}
            },
            "required": ["symbol"]
        }
    ),
    # Moving Average Crossover Strategy Tools
    Tool(
        name="backtest_moving_average_crossover",
        description="Backtest the moving average crossover trading strategy on historical data. Returns performance metrics including total return, Sharpe ratio, max drawdown, and trade statistics.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "short_window": {"type": "integer", "description": "Short moving average period (default: 50)"},
                "long_window": {"type": "integer", "description": "Long moving average period (default: 200)"}
            },
            "required": ["symbol", "start_date", "end_date"]
        }
    ),
    Tool(
        name="optimize_moving_average_crossover",
        description="Optimize moving average crossover strategy parameters to find the best window combination for maximum returns or Sharpe ratio.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "optimization_metric": {"type": "string", "description": "Metric to optimize (return or sharpe, default: sharpe)"}
            },
            "required": ["symbol", "start_date", "end_date"]
        }
    ),
    Tool(
        name="get_moving_average_crossover_signals",
        description="Get current trading signals for the moving average crossover strategy based on recent price data.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "short_window": {"type": "integer", "description": "Short moving average period (default: 50)"},
                "long_window": {"type": "integer", "description": "Long moving average period (default: 200)"}
            },
            "required": ["symbol"]
        }
    ),
    # Momentum Price & Volume Strategy Tools
    Tool(
        name="backtest_momentum_price_vol",
        description="Backtest the momentum breakout trading strategy based on price and volume on historical data. Returns performance metrics including total return, Sharpe ratio, max drawdown, and trade statistics.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "price_lookback": {"type": "integer", "description": "Lookback period for price momentum (default: 20)"},
                "volume_lookback": {"type": "integer", "description": "Lookback period for volume analysis (default: 20)"},
                "price_threshold": {"type": "number", "description": "Price momentum threshold percentage (default: 5.0)"},
                "volume_threshold": {"type": "number", "description": "Volume surge threshold multiplier (default: 1.5)"}
            },
            "required": ["symbol", "start_date", "end_date"]
        }
    ),
    Tool(
        name="optimize_momentum_price_vol",
        description="Optimize momentum price & volume strategy parameters to find the best combination for maximum returns or Sharpe ratio.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "optimization_metric": {"type": "string", "description": "Metric to optimize (return or sharpe, default: sharpe)"}
            },
            "required": ["symbol", "start_date", "end_date"]
        }
    ),
    Tool(
        name="get_momentum_price_vol_signals",
        description="Get current trading signals for the momentum price & volume strategy based on recent price and volume data.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                "price_lookback": {"type": "integer", "description": "Lookback period for price momentum (default: 20)"},
                "volume_lookback": {"type": "integer", "description": "Lookback period for volume analysis (default: 20)"},
                "price_threshold": {"type": "number", "description": "Price momentum threshold percentage (default: 5.0)"},
                "volume_threshold": {"type": "number", "description": "Volume surge threshold multiplier (default: 1.5)"}
            },
            "required": ["symbol"]
        }
    )
]

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available trading strategy tools."""
    return TOOLS

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a trading strategy tool."""
    try:
        logger.info(f"Calling tool: {name} with arguments: {arguments}")
        
        # Mean Reversion Strategy Tools
        if name == "get_mean_reversion_signals":
            strategy = MeanReversionStrategy()
            result = strategy.mean_reversion_signal(
                symbol=arguments["symbol"],
                method="bollinger",  # Default method
                window=arguments.get("lookback_period", 20),
                upper_threshold=arguments.get("entry_threshold", 2.0),
                lower_threshold=-arguments.get("entry_threshold", 2.0),
                period="1y"  # Default period
            )
        elif name == "backtest_mean_reversion":
            # Use signal method with date range - simplified backtest
            strategy = MeanReversionStrategy()
            result = strategy.mean_reversion_signal(
                symbol=arguments["symbol"],
                method="bollinger",
                window=arguments.get("lookback_period", 20),
                upper_threshold=arguments.get("entry_threshold", 2.0),
                lower_threshold=-arguments.get("entry_threshold", 2.0),
                period="1y"
            )
            result["note"] = "This is a simplified implementation. For full backtesting, use the API endpoints."
        elif name == "optimize_mean_reversion":
            result = {
                "error": "Optimization not yet implemented in MCP server",
                "suggestion": "Use the strategy classes directly or the FastAPI endpoints for optimization"
            }
        
        # Moving Average Crossover Strategy Tools
        elif name == "get_moving_average_crossover_signals":
            strategies = TradingStrategies()
            result = strategies.moving_average_crossover(
                symbol=arguments["symbol"],
                fast_period=arguments.get("short_window", 50),
                slow_period=arguments.get("long_window", 200),
                period="2y"
            )
        elif name == "backtest_moving_average_crossover":
            strategies = TradingStrategies()
            result = strategies.backtest_strategy(
                symbol=arguments["symbol"],
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date"),
                short_window=arguments.get("short_window", 50),
                long_window=arguments.get("long_window", 200)
            )
        elif name == "optimize_moving_average_crossover":
            result = {
                "error": "Optimization not yet implemented in MCP server",
                "suggestion": "Use the strategy classes directly or the FastAPI endpoints for optimization"
            }
        
        # Momentum Price & Volume Strategy Tools
        elif name == "get_momentum_price_vol_signals":
            strategy = MomentumBreakoutStrategy(
                lookback_period=arguments.get("price_lookback", 20),
                volume_threshold=arguments.get("volume_threshold", 1.5),
                breakout_threshold=arguments.get("price_threshold", 5.0) / 100.0  # Convert percentage to decimal
            )
            result = strategy.analyze_stock(arguments["symbol"])
        elif name == "backtest_momentum_price_vol":
            result = {
                "error": "Backtesting not yet implemented for momentum strategy in MCP server",
                "suggestion": "Use the FastAPI endpoints or implement backtesting in the strategy class"
            }
        elif name == "optimize_momentum_price_vol":
            result = {
                "error": "Optimization not yet implemented in MCP server",
                "suggestion": "Use the strategy classes directly or the FastAPI endpoints for optimization"
            }
        
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        # Format the result as JSON
        result_json = json.dumps(result, indent=2, default=str)
        
        return [TextContent(
            type="text",
            text=result_json
        )]
    
    except Exception as e:
        logger.error(f"Error executing tool {name}: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "tool": name,
                "arguments": arguments
            }, indent=2)
        )]

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    
    asyncio.run(main())