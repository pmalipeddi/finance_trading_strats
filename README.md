# Trading Strategies - Moving Average Crossover & Mean Reversion

A comprehensive Python toolkit for implementing and analyzing trading strategies including Moving Average Crossover and Mean Reversion strategies. Includes FastAPI endpoints and MCP (Model Context Protocol) server integration for use with Claude Desktop.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Claude Desktop MCP Setup](#claude-desktop-mcp-setup)
- [API Endpoints](#api-endpoints)
- [MCP Server Tools](#mcp-server-tools)
- [Troubleshooting](#troubleshooting)

## Features

### Trading Strategies

1. **Moving Average Crossover Strategy**
   - Golden Cross (BUY) and Death Cross (SELL) signals
   - Configurable fast and slow moving average periods
   - Bulk processing for S&P 500 and NASDAQ-100

2. **Mean Reversion Strategy**
   - Multiple methods: Bollinger Bands, RSI, Z-Score, and Combined
   - Configurable window sizes (short: 20, medium: 50, long: 100 days)
   - Bulk processing for S&P 500 and NASDAQ-100

### Integration Options

- **FastAPI REST API**: HTTP endpoints for web applications
- **MCP Server**: Integration with Claude Desktop and other MCP clients
- **Direct Python Usage**: Use strategies programmatically

## Requirements

### Python Version

**Python 3.9 - 3.13** (Recommended: Python 3.11, 3.12, or 3.13)

**Important Notes:**
- **Python 3.13** (including 3.13.9) is fully supported but requires special NumPy setup (see installation steps)
- Python 3.9-3.12 work with standard package versions from PyPI
- Python 3.8 and below are not supported

### System Requirements

- macOS, Linux, or Windows
- Internet connection (for fetching stock data)
- At least 2GB RAM (for bulk processing)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/pmalipeddi/finance_trading_strats.git
cd finance_trading_strats
```

### Step 2: Check Python Version

```bash
python3 --version
```

**Expected output:** `Python 3.9.x` to `Python 3.13.x` (e.g., `Python 3.13.9`)

If you don't have Python 3.9+, install it:
- **macOS**: `brew install python@3.13` or download from [python.org](https://www.python.org/downloads/)
- **Linux**: `sudo apt-get install python3.13` (Ubuntu/Debian) or download from [python.org](https://www.python.org/downloads/)
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 4: Install Dependencies

#### For Python 3.13 (Including 3.13.9) - Current Setup

The main `requirements.txt` is configured for Python 3.13.9:

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies (includes pre-release NumPy for Python 3.13)
pip install -r requirements.txt
```

**Note:** If you encounter issues with NumPy installation, you may need to install it separately first:

```bash
# Install NumPy pre-release for Python 3.13
pip install --force-reinstall --pre numpy --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple

# Then install other dependencies
pip install -r requirements.txt
```

#### For Python 3.9 - 3.12 (Alternative Setup)

If you're using Python 3.9-3.12, you'll need to install standard NumPy versions instead:

```bash
# Upgrade pip
pip install --upgrade pip

# Install NumPy compatible with Python 3.9-3.12
pip install "numpy>=1.21.0,<2.0"

# Install other dependencies
pip install yfinance>=0.2.28 lxml>=4.9.0 requests>=2.31.0 fastapi>=0.104.0 uvicorn[standard]>=0.24.0 pydantic>=2.0.0 mcp>=0.9.0
```

### Step 5: Verify Installation

```bash
# Test imports
python3 -c "from moving_average_crossover_strategy import TradingStrategies; from mean_reversion_strategy import MeanReversionStrategy; print('✓ All imports successful!')"
```

## Quick Start

### Example 1: Moving Average Crossover for a Single Stock

```python
from moving_average_crossover_strategy import TradingStrategies

# Create instance
strategy = TradingStrategies(symbol="AAPL")

# Get crossover signal
result = strategy.moving_average_crossover(fast_period=50, slow_period=200)

print(f"Signal: {result['current_signal']}")
print(f"Price: ${result['current_price']:.2f}")
```

### Example 2: Mean Reversion for a Single Stock

```python
from mean_reversion_strategy import MeanReversionStrategy

# Create instance
strategy = MeanReversionStrategy(symbol="AAPL")

# Get mean reversion signal (Bollinger Bands, 20-day window)
result = strategy.mean_reversion_signal(method="bollinger", window=20)

print(f"Signal: {result['current_signal']}")
print(f"Price: ${result['current_price']:.2f}")
```

### Example 3: Generate S&P 500 Signals

```python
from moving_average_crossover_strategy import TradingStrategies

strategy = TradingStrategies()

# Generate signals for all S&P 500 stocks
df = strategy.moving_average_crossover_sp500(
    fast_period=50,
    slow_period=200,
    csv_path="sp500_signals.csv"
)

print(f"Processed {len(df)} stocks")
```

## Usage

### Running Strategies Directly

```bash
# Moving Average Crossover examples
python3 moving_average_crossover_strategy.py

# Mean Reversion examples
python3 mean_reversion_strategy.py
```

### Running FastAPI Servers

#### Moving Average Crossover API (Port 8000)

```bash
python3 api.py
# Or
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Access at: `http://localhost:8000/docs`

#### Mean Reversion API (Port 8001)

```bash
python3 mean_reversion.py
# Or
uvicorn mean_reversion:app --reload --host 0.0.0.0 --port 8001
```

Access at: `http://localhost:8001/docs`

## Claude Desktop MCP Setup

### Prerequisites

1. **Claude Desktop Installed**
   - Download from [claude.ai/download](https://claude.ai/download)
   - Install and open Claude Desktop at least once

2. **Virtual Environment Activated**
   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate  # Windows
   ```

### Step-by-Step Configuration

#### Step 1: Locate Claude Desktop Config File

The config file location depends on your operating system:

**macOS:**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

#### Step 2: Get Your Project Paths

Run these commands to get the paths you'll need:

```bash
# Get project root path
pwd

# Get Python executable path (from your venv)
which python
# or on Windows:
where python
```

**Example output:**
- Project path: `/Users/username/finance_trading_strats`
- Python path: `/Users/username/finance_trading_strats/venv/bin/python`

#### Step 3: Create/Edit Config File

**Option A: Automated Setup (Recommended)**

```bash
python3 generate_mcp_config.py
```

This script will:
- Detect your Python executable
- Generate the correct configuration
- Optionally write it to Claude Desktop config

**Option B: Manual Setup**

1. Open the config file in a text editor (create it if it doesn't exist)

2. Add or update the `mcpServers` section:

**macOS/Linux Example:**
```json
{
  "mcpServers": {
    "trading-strategies": {
      "command": "/full/path/to/finance_trading_strats/venv/bin/python",
      "args": [
        "/full/path/to/finance_trading_strats/mcp_server.py"
      ],
      "cwd": "/full/path/to/finance_trading_strats",
      "env": {}
    }
  }
}
```

**Windows Example:**
```json
{
  "mcpServers": {
    "trading-strategies": {
      "command": "C:\\full\\path\\to\\finance_trading_strats\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\full\\path\\to\\finance_trading_strats\\mcp_server.py"
      ],
      "cwd": "C:\\full\\path\\to\\finance_trading_strats",
      "env": {}
    }
  }
}
```

**Important:**
- Replace `/full/path/to/` with your actual project path
- Use **absolute paths** (not relative)
- Use forward slashes `/` on macOS/Linux
- Use backslashes `\\` on Windows (or forward slashes work too)

#### Step 4: Verify Config File

Check that your config file is valid JSON:

```bash
# macOS/Linux
python3 -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
python -m json.tool "%APPDATA%\Claude\claude_desktop_config.json"
```

If there are no errors, the JSON is valid.

#### Step 5: Test MCP Server

Before connecting to Claude Desktop, test the server:

```bash
# Activate virtual environment
source venv/bin/activate

# Test the server
python3 mcp_server.py --test
```

**Expected output:**
```
============================================================
MCP Server Test Mode
============================================================

✓ Server file syntax: OK
✓ MCP SDK imports: OK
✓ Moving average crossover strategy import: OK
✓ Mean reversion strategy import: OK
✓ Moving average crossover strategy initialization: OK
✓ Mean reversion strategy initialization: OK

============================================================
All checks passed! Server is ready to use.
============================================================
```

#### Step 6: Restart Claude Desktop

**Critical:** You must restart Claude Desktop for changes to take effect:

1. **Quit Claude Desktop completely**
   - macOS: Press ⌘Q or use Quit from menu
   - Windows: Close all windows and check Task Manager
   - Linux: Close all windows

2. **Wait 5-10 seconds**

3. **Reopen Claude Desktop**

#### Step 7: Verify Connection

1. **Open a new conversation** in Claude Desktop

2. **Ask Claude:**
   ```
   What MCP tools do you have available?
   ```

3. **Expected Response:**
   Claude should list 6 tools:
   - `get_moving_average_crossover`
   - `get_sp500_crossover_signals`
   - `get_nasdaq100_crossover_signals`
   - `get_mean_reversion_signal`
   - `get_sp500_mean_reversion_signals`
   - `get_nasdaq100_mean_reversion_signals`

#### Step 8: Test the Tools

Try these example queries:

```
Get the moving average crossover signal for AAPL
```

```
Get the mean reversion signal for TSLA with a 20-day window using Bollinger Bands
```

```
Get mean reversion signals for all S&P 500 stocks with a 50-day medium-term window
```

## API Endpoints

### Moving Average Crossover API (Port 8000)

**Base URL:** `http://localhost:8000`

- `GET /signal/{symbol}` - Get crossover signal for a single ticker
- `GET /sp500` - Get signals for all S&P 500 stocks
- `GET /nasdaq100` - Get signals for all NASDAQ-100 stocks
- `GET /docs` - Interactive API documentation

**Example:**
```bash
curl "http://localhost:8000/signal/AAPL?fast_period=50&slow_period=200"
```

### Mean Reversion API (Port 8001)

**Base URL:** `http://localhost:8001`

- `GET /signal/{symbol}` - Get mean reversion signal for a single ticker
- `GET /sp500` - Get signals for all S&P 500 stocks
- `GET /nasdaq100` - Get signals for all NASDAQ-100 stocks
- `GET /docs` - Interactive API documentation

**Example:**
```bash
curl "http://localhost:8001/signal/AAPL?method=bollinger&window=20"
```

## MCP Server Tools

When connected to Claude Desktop, you have access to 6 tools:

### Moving Average Crossover Tools

1. **`get_moving_average_crossover`**
   - Get signal for a single stock
   - Parameters: `symbol`, `fast_period`, `slow_period`, `period`, `interval`

2. **`get_sp500_crossover_signals`**
   - Get signals for all S&P 500 stocks
   - Parameters: `fast_period`, `slow_period`, `period`, `interval`, `max_workers`

3. **`get_nasdaq100_crossover_signals`**
   - Get signals for all NASDAQ-100 stocks
   - Same parameters as S&P 500

### Mean Reversion Tools

4. **`get_mean_reversion_signal`**
   - Get signal for a single stock
   - Parameters: `symbol`, `method` (bollinger/rsi/zscore/combined), `window` (20/50/100), `period`, `interval`

5. **`get_sp500_mean_reversion_signals`**
   - Get signals for all S&P 500 stocks
   - Parameters: `method`, `window`, `period`, `interval`, `max_workers`

6. **`get_nasdaq100_mean_reversion_signals`**
   - Get signals for all NASDAQ-100 stocks
   - Same parameters as S&P 500

## Troubleshooting

### Issue: Python Version Not Compatible

**Error:** `No matching distribution found for numpy==1.26.4` or similar NumPy errors

**Solution:**
- **For Python 3.13.9**: The `requirements.txt` is already configured. If you get errors, install NumPy separately first:
  ```bash
  pip install --force-reinstall --pre numpy --extra-index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple
  pip install -r requirements.txt
  ```
- **For Python 3.9-3.12**: Install standard NumPy first:
  ```bash
  pip install "numpy>=1.21.0,<2.0"
  pip install yfinance>=0.2.28 lxml>=4.9.0 requests>=2.31.0 fastapi>=0.104.0 uvicorn[standard]>=0.24.0 pydantic>=2.0.0 mcp>=0.9.0
  ```

### Issue: MCP Server Not Connecting

**Symptoms:** Claude Desktop doesn't show the tools

**Solutions:**

1. **Check config file exists:**
   ```bash
   # macOS
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. **Verify paths are absolute:**
   - Use full paths, not relative
   - Check Python path points to your venv

3. **Test server manually:**
   ```bash
   source venv/bin/activate
   python3 mcp_server.py --test
   ```

4. **Check Claude Desktop logs:**
   ```bash
   # macOS
   tail -f ~/Library/Logs/Claude/claude_desktop.log
   ```

5. **Restart Claude Desktop completely** (not just close window)

### Issue: Import Errors

**Error:** `ModuleNotFoundError: No module named 'mcp'`

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: CSV File Permission Errors

**Error:** `Permission denied` or `Could not save CSV`

**Solution:**
- The code handles this gracefully - data is still returned
- Check directory permissions: `ls -la .`
- CSV saving is optional - results are still available

### Issue: Network/API Errors

**Error:** `Failed to retrieve S&P 500 constituents`

**Solutions:**
- Check internet connection
- Verify `lxml` is installed: `pip install lxml`
- Try again - sometimes Wikipedia blocks requests temporarily

### Issue: Server Hangs When Run Directly

**Symptom:** `python3 mcp_server.py` appears to hang

**Explanation:** This is **normal behavior**! The server waits for MCP protocol messages. It's working correctly.

**To test:** Use `python3 mcp_server.py --test` instead

## Project Structure

```
finance_trading_strats/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── moving_average_crossover_strategy.py  # MA Crossover strategy
├── mean_reversion_strategy.py         # Mean Reversion strategy
├── api.py                             # FastAPI for MA Crossover
├── mean_reversion.py                  # FastAPI for Mean Reversion
├── mcp_server.py                      # MCP server (for Claude Desktop)
├── mcp_client.py                      # Example MCP client
├── generate_mcp_config.py             # Config generator helper
└── test_mean_reversion.py             # Test script
```

## Examples

### Example 1: Get Moving Average Crossover Signal

**In Claude Desktop:**
```
Get the moving average crossover signal for Apple (AAPL)
```

**Via API:**
```bash
curl "http://localhost:8000/signal/AAPL"
```

**In Python:**
```python
from moving_average_crossover_strategy import TradingStrategies
strategy = TradingStrategies(symbol="AAPL")
result = strategy.moving_average_crossover()
print(result['current_signal'])
```

### Example 2: Get Mean Reversion Signal with Different Windows

**Short-term (20 days) in Claude Desktop:**
```
Get the mean reversion signal for Tesla with a 20-day short-term window using Bollinger Bands
```

**Medium-term (50 days) in Claude Desktop:**
```
Get mean reversion signals for all S&P 500 stocks with a 50-day medium-term window
```

**Long-term (100 days) in Claude Desktop:**
```
Get mean reversion signals for NASDAQ-100 with a 100-day long-term window using the combined method
```

### Example 3: Generate CSV Files

**In Python:**
```python
from moving_average_crossover_strategy import TradingStrategies

strategy = TradingStrategies()

# Generate S&P 500 signals
df = strategy.moving_average_crossover_sp500(
    fast_period=50,
    slow_period=200,
    csv_path="sp500_signals.csv"
)

# Generate NASDAQ-100 signals
df = strategy.moving_average_crossover_nasdaq100(
    fast_period=50,
    slow_period=200,
    csv_path="nasdaq100_signals.csv"
)
```

## Testing

### Test Mean Reversion Strategy

```bash
python3 test_mean_reversion.py
```

This runs comprehensive tests on all mean reversion methods.

### Test MCP Server

```bash
python3 mcp_server.py --test
```

This verifies all imports and initialization.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the installation and setup sections in this README
3. Check Claude Desktop logs for error messages

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]

---

**Happy Trading! 📈**

