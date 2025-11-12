# Connecting MCP Server to Claude Desktop

This guide will help you connect your Trading Strategies MCP server to Claude Desktop.

## Prerequisites

1. **Claude Desktop installed** - Download from [Anthropic's website](https://claude.ai/download)
2. **Python environment set up** - Your virtual environment with all dependencies installed
3. **MCP server file** - `mcp_server.py` in your project directory

## Step 1: Locate Claude Desktop Configuration File

The configuration file location depends on your operating system:

### macOS
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

### Windows
```
%APPDATA%\Claude\claude_desktop_config.json
```

### Linux
```
~/.config/Claude/claude_desktop_config.json
```

## Step 2: Edit the Configuration File

1. **Open the configuration file** in a text editor
   - If the file doesn't exist, create it
   - If it exists, you'll see existing MCP server configurations

2. **Add your trading strategies server** to the `mcpServers` section

### Example Configuration

Here's what your `claude_desktop_config.json` should look like:

```json
{
  "mcpServers": {
    "trading-strategies": {
      "command": "python",
      "args": [
        "/Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats/mcp_server.py"
      ],
      "cwd": "/Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats",
      "env": {}
    }
  }
}
```

### Important Notes:

- **`command`**: Use `python` or `python3` depending on your system
  - If using a virtual environment, use the full path to the Python executable:
    ```json
    "command": "/Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats/venv/bin/python"
    ```

- **`args`**: Full path to your `mcp_server.py` file

- **`cwd`**: Working directory (your project root)

- **`env`**: Environment variables (optional)
  - If you need to set environment variables, add them here:
    ```json
    "env": {
      "PYTHONPATH": "/Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats"
    }
    ```

## Step 3: Using Virtual Environment (Recommended)

If you're using a virtual environment (which you should be), use the full path to the Python executable:

```json
{
  "mcpServers": {
    "trading-strategies": {
      "command": "/Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats/venv/bin/python",
      "args": [
        "/Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats/mcp_server.py"
      ],
      "cwd": "/Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats",
      "env": {}
    }
  }
}
```

### Finding Your Virtual Environment Python Path

On macOS/Linux:
```bash
which python
# or
which python3
```

Or if you're in your virtual environment:
```bash
which python
```

On Windows:
```powershell
where python
```

## Step 4: Restart Claude Desktop

1. **Quit Claude Desktop completely** (not just close the window)
2. **Reopen Claude Desktop**
3. The MCP server should automatically connect

## Step 5: Verify Connection

1. **Open Claude Desktop**
2. **Start a new conversation**
3. **Ask Claude to list available tools:**
   ```
   What MCP tools do you have available?
   ```
   or
   ```
   Can you show me the trading strategy tools?
   ```

4. **Test the connection** by asking:
   ```
   Get the moving average crossover signal for AAPL
   ```

## Step 6: Using the Tools

Once connected, you can ask Claude to use the tools:

### Example Queries:

1. **Get signal for a single stock:**
   ```
   What's the moving average crossover signal for Tesla (TSLA)?
   ```

2. **Get signal with custom parameters:**
   ```
   Get the moving average crossover for MSFT with a 20-day fast MA and 50-day slow MA
   ```

3. **Get S&P 500 signals:**
   ```
   Get moving average crossover signals for all S&P 500 stocks
   ```

4. **Get NASDAQ-100 signals:**
   ```
   Show me the moving average crossover signals for NASDAQ-100 stocks
   ```

## Troubleshooting

### Issue: MCP server not connecting

**Solution:**
1. Check that the Python path is correct in the config file
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Test the server manually:
   ```bash
   python mcp_server.py
   ```
   (It should start and wait for input - this is normal)

### Issue: "Command not found" or "Python not found"

**Solution:**
- Use the full path to Python executable
- Make sure you're using the Python from your virtual environment
- On macOS/Linux, you can find it with: `which python`
- On Windows: `where python`

### Issue: Import errors

**Solution:**
1. Make sure you're using the Python from your virtual environment
2. Verify all packages are installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Check that `moving_average_crossover_strategy.py` is in the same directory

### Issue: Claude doesn't see the tools

**Solution:**
1. Restart Claude Desktop completely
2. Check the Claude Desktop logs for errors
3. Verify the JSON syntax in the config file is valid
4. Make sure the file paths are absolute (not relative)

### Issue: Permission denied

**Solution:**
- Make sure the `mcp_server.py` file is executable:
  ```bash
  chmod +x mcp_server.py
  ```

## Viewing Logs

To see what's happening with your MCP server:

### macOS
Check Claude Desktop logs:
```
~/Library/Logs/Claude/claude_desktop.log
```

### Windows
```
%APPDATA%\Claude\logs\claude_desktop.log
```

### Linux
```
~/.config/Claude/logs/claude_desktop.log
```

## Multiple MCP Servers

If you have multiple MCP servers, your config file might look like:

```json
{
  "mcpServers": {
    "trading-strategies": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/mcp_server.py"],
      "cwd": "/path/to/project"
    },
    "another-server": {
      "command": "python",
      "args": ["/path/to/another_server.py"],
      "cwd": "/path/to/another/project"
    }
  }
}
```

## Testing the Server Manually

Before connecting to Claude Desktop, you can test the server:

```bash
cd /Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats
python mcp_server.py
```

The server will start and wait for MCP protocol messages via stdio. This is normal behavior - it means the server is working correctly.

## Next Steps

Once connected, you can:
- Ask Claude to analyze stocks using the crossover signals
- Get signals for multiple stocks
- Compare signals across different time periods
- Export results to CSV files

Enjoy using your Trading Strategies MCP server with Claude Desktop!

