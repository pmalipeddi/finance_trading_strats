# How to Use Your MCP Server

## Quick Answer: You Don't "Open" It Manually!

The MCP server **runs automatically** when Claude Desktop connects to it. You don't need to start it yourself.

## How It Works

1. **Claude Desktop reads** your config file: `~/Library/Application Support/Claude/claude_desktop_config.json`
2. **Claude Desktop starts** your Python script automatically
3. **Claude Desktop communicates** with it via stdio
4. **You use it** by asking Claude to use the tools

## Steps to Use It

### 1. Make Sure Everything is Set Up

✅ Config file exists: `~/Library/Application Support/Claude/claude_desktop_config.json`  
✅ MCP server file exists: `mcp_server.py`  
✅ Dependencies installed: `pip install -r requirements.txt`

### 2. Restart Claude Desktop

**Important:** You must restart Claude Desktop after adding/changing the config:
- Quit Claude Desktop completely (⌘Q on Mac)
- Reopen Claude Desktop

### 3. Test the Connection

Open a new conversation in Claude Desktop and ask:

```
What MCP tools do you have available?
```

**If it works**, Claude will list:
- `get_moving_average_crossover`
- `get_sp500_crossover_signals`
- `get_nasdaq100_crossover_signals`

### 4. Use the Tools

Now you can ask Claude to use the tools:

**Example 1: Get signal for a single stock**
```
Get the moving average crossover signal for AAPL
```

**Example 2: Get signal with custom parameters**
```
What's the moving average crossover for Tesla with a 20-day fast MA and 50-day slow MA?
```

**Example 3: Get all S&P 500 signals**
```
Get moving average crossover signals for all S&P 500 stocks
```

**Example 4: Get NASDAQ-100 signals**
```
Show me the moving average crossover signals for NASDAQ-100 stocks
```

## Troubleshooting

### "I don't see the tools"

1. **Check Claude Desktop logs:**
   ```bash
   tail -f ~/Library/Logs/Claude/claude_desktop.log
   ```
   Look for errors about the MCP server

2. **Verify config file:**
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

3. **Test the server manually:**
   ```bash
   cd /Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats
   source venv/bin/activate
   python3 -c "from mcp.server import Server; print('MCP installed')"
   ```

4. **Restart Claude Desktop again**

### "Server not connecting"

- Make sure paths in config are **absolute** (not relative)
- Verify Python path points to your virtual environment
- Check that `mcp_server.py` exists at the specified path

### "Import errors"

Install dependencies:
```bash
cd /Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats
source venv/bin/activate
pip install -r requirements.txt
```

## Testing the Server (Optional)

If you want to verify the server works before using it in Claude Desktop:

### Test 1: Syntax Check
```bash
python3 -m py_compile mcp_server.py
```

### Test 2: Import Check
```bash
source venv/bin/activate
python3 -c "from mcp.server import Server; from moving_average_crossover_strategy import TradingStrategies; print('OK')"
```

### Test 3: Test Trading Strategies
```bash
source venv/bin/activate
python3 -c "from moving_average_crossover_strategy import TradingStrategies; s = TradingStrategies(); result = s.moving_average_crossover('AAPL'); print('Success!')"
```

## What Happens When You Run It Directly?

If you try:
```bash
python3 mcp_server.py
```

The server will:
- Start successfully
- **Appear to "hang"** (waiting for input)
- This is **normal behavior** - it's waiting for MCP protocol messages

**This is expected!** The server is designed to be started by Claude Desktop, not run manually.

## Summary

- ✅ **Config file is created** at `~/Library/Application Support/Claude/claude_desktop_config.json`
- ✅ **Server is ready** - it will start automatically when Claude Desktop connects
- ✅ **Restart Claude Desktop** to connect
- ✅ **Ask Claude** to use the tools - no manual server startup needed!

The MCP server is a background service that Claude Desktop manages. You just use it through Claude's interface!

