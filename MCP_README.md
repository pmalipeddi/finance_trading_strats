# MCP Server for Trading Strategies

This MCP (Model Context Protocol) server exposes moving average crossover trading signals as tools that can be used by MCP-compatible AI assistants.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The MCP server requires the `mcp` package. If the import fails, you may need to install it from the official source:
```bash
pip install mcp
```

## Running the MCP Server

### Method 1: Direct Execution
```bash
python mcp_server.py
```

The server will communicate via stdio (standard input/output), which is the standard way MCP servers operate.

### Method 2: With MCP Client

The server is designed to work with MCP-compatible clients. You'll need to configure your MCP client to use this server.

## Available Tools

The MCP server exposes the following tools:

### 1. `get_moving_average_crossover`
Get moving average crossover signal for a single stock ticker.

**Parameters:**
- `symbol` (required): Stock ticker symbol (e.g., "AAPL", "MSFT")
- `fast_period` (optional, default: 50): Period for fast moving average
- `slow_period` (optional, default: 200): Period for slow moving average
- `period` (optional, default: "2y"): Historical data period
- `interval` (optional, default: "1d"): Data interval

**Returns:**
- Current price
- Fast and slow moving averages
- Current signal (BUY/SELL/HOLD)
- Statistics (total signals, buy signals, sell signals)

### 2. `get_sp500_crossover_signals`
Get moving average crossover signals for all S&P 500 stocks.

**Parameters:**
- `fast_period` (optional, default: 50)
- `slow_period` (optional, default: 200)
- `period` (optional, default: "2y")
- `interval` (optional, default: "1d")
- `max_workers` (optional, default: 20): Number of concurrent workers

**Returns:**
- Total tickers processed
- Signal distribution
- Results saved to CSV file

### 3. `get_nasdaq100_crossover_signals`
Get moving average crossover signals for all NASDAQ-100 stocks.

**Parameters:** Same as S&P 500 tool

**Returns:** Same format as S&P 500 tool

## Example Usage

When connected to an MCP client, you can call the tools like this:

```
Tool: get_moving_average_crossover
Arguments: {
  "symbol": "AAPL",
  "fast_period": 50,
  "slow_period": 200
}
```

## Configuration

If you're using a specific MCP client (like Claude Desktop), you may need to add this server to your configuration file. The exact location and format depends on your client.

Example configuration (location varies by client):
```json
{
  "mcpServers": {
    "trading-strategies": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "cwd": "/path/to/project"
    }
  }
}
```

## Troubleshooting

1. **Import errors**: Make sure all dependencies are installed, especially the `mcp` package
2. **Connection issues**: Verify the MCP client is properly configured
3. **Data errors**: Ensure you have internet connectivity for fetching stock data

## Integration with FastAPI

This MCP server is separate from the FastAPI server (`api.py`). You can run both:
- FastAPI server: For HTTP/REST API access
- MCP server: For MCP protocol access (used by AI assistants)

Both use the same underlying `TradingStrategies` class, so they provide consistent results.

