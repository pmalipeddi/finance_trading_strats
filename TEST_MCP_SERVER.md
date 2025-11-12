# Testing Your MCP Server

## How MCP Servers Work

MCP servers don't run like regular applications. They communicate via **stdio** (standard input/output) and are **started automatically by the MCP client** (like Claude Desktop).

When Claude Desktop connects, it:
1. Starts your Python script
2. Communicates with it via stdin/stdout
3. Sends JSON-RPC messages
4. Receives responses

## Testing the Server Manually

### Option 1: Quick Syntax Check

Verify the server file has no syntax errors:

```bash
python -m py_compile mcp_server.py
```

If there are no errors, it will complete silently.

### Option 2: Test Imports

Check if all imports work:

```bash
python -c "from mcp_server import *; print('All imports successful!')"
```

### Option 3: Verify Dependencies

Make sure all required packages are installed:

```bash
python -c "import mcp; import moving_average_crossover_strategy; print('Dependencies OK!')"
```

### Option 4: Test Trading Strategies Directly

Test the underlying functionality:

```bash
python -c "from moving_average_crossover_strategy import TradingStrategies; s = TradingStrategies(); result = s.moving_average_crossover('AAPL'); print('Success!')"
```

## What Happens When You "Run" the Server

If you try to run it directly:

```bash
python mcp_server.py
```

The server will:
- Start and wait for input
- Appear to "hang" (this is normal!)
- Wait for MCP protocol messages via stdin

This is **expected behavior** - the server is waiting for Claude Desktop to connect.

## Verifying It's Working with Claude Desktop

### Step 1: Check Claude Desktop Logs

**macOS:**
```bash
tail -f ~/Library/Logs/Claude/claude_desktop.log
```

Look for:
- Connection messages
- Any error messages
- MCP server startup logs

### Step 2: Test in Claude Desktop

1. **Restart Claude Desktop** (if you just added the config)
2. **Open a new conversation**
3. **Ask Claude:**
   ```
   What MCP tools do you have available?
   ```
4. **If it works, you'll see:**
   - `get_moving_average_crossover`
   - `get_sp500_crossover_signals`
   - `get_nasdaq100_crossover_signals`

### Step 3: Try a Tool Call

Ask Claude:
```
Get the moving average crossover signal for AAPL
```

If it works, Claude will call the tool and return the results.

## Troubleshooting

### Issue: Server not starting

**Check the config file:**
```bash
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Verify paths are correct:**
- Python path should be absolute
- MCP server path should be absolute
- Working directory should be absolute

### Issue: Import errors

**Test imports:**
```bash
cd /Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats
python -c "from moving_average_crossover_strategy import TradingStrategies; print('OK')"
```

**Install missing packages:**
```bash
pip install -r requirements.txt
```

### Issue: Permission errors

**Make sure the file is readable:**
```bash
chmod +r mcp_server.py
```

### Issue: Python path wrong

**Find your Python:**
```bash
which python
# or
which python3
```

**Update the config file** with the correct path.

## Debug Mode (Advanced)

If you want to see what's happening, you can modify `mcp_server.py` temporarily to add logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
```

This will send debug messages to stderr, which you can see in Claude Desktop logs.

## Summary

- **You don't manually "open" the MCP server** - Claude Desktop starts it automatically
- **To test**: Restart Claude Desktop and ask it about available tools
- **To debug**: Check Claude Desktop logs and verify your config file
- **The server "hanging" when run directly is normal** - it's waiting for MCP messages

