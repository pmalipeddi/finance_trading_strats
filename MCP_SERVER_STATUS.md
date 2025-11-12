# MCP Server Status Guide

## Testing the Server

### Option 1: Test Mode (Recommended)

Run the server in test mode to verify everything is working:

```bash
cd /Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats
source venv/bin/activate
python3 mcp_server.py --test
```

**Expected Output:**
```
============================================================
MCP Server Test Mode
============================================================

✓ Server file syntax: OK
✓ MCP SDK imports: OK
✓ Trading strategies import: OK
✓ Trading strategies initialization: OK

============================================================
All checks passed! Server is ready to use.
============================================================
```

### Option 2: Normal Mode (Shows Startup Messages)

When you run the server normally:

```bash
python3 mcp_server.py
```

**Expected Output:**
```
MCP Server: Trading Strategies - Starting...
MCP Server: Waiting for MCP client connection...
MCP Server: Server is ready and listening on stdio
```

**Note:** The server will then appear to "hang" - this is **normal**! It's waiting for MCP protocol messages from Claude Desktop.

## Understanding the Behavior

### Why It "Hangs"

When you run `python3 mcp_server.py`:
1. ✅ Server starts successfully
2. ✅ Shows startup messages
3. ⏳ **Waits for input** (this is expected!)
4. ⏳ Appears to "hang" (actually waiting for MCP client)

**This is correct behavior!** The server communicates via stdio and waits for Claude Desktop to send commands.

### What Happens with Claude Desktop

When Claude Desktop connects:
1. Claude Desktop starts your `mcp_server.py` script
2. Server prints: `MCP Server: Connected to MCP client`
3. Claude Desktop sends MCP protocol messages
4. Server responds with tool results
5. Everything happens automatically in the background

## Verifying It's Working

### Method 1: Check Claude Desktop Logs

**macOS:**
```bash
tail -f ~/Library/Logs/Claude/claude_desktop.log
```

Look for:
- MCP server startup messages
- Connection confirmations
- Any error messages

### Method 2: Test in Claude Desktop

1. **Restart Claude Desktop**
2. **Open a new conversation**
3. **Ask:** `What MCP tools do you have available?`

**If working, you'll see:**
- `get_moving_average_crossover`
- `get_sp500_crossover_signals`
- `get_nasdaq100_crossover_signals`

### Method 3: Use a Tool

Ask Claude:
```
Get the moving average crossover signal for AAPL
```

If it works, Claude will call the tool and return results.

## Troubleshooting

### Issue: No startup messages appear

**Solution:**
- Make sure you're running: `python3 mcp_server.py` (not `python`)
- Check that you're in the project directory
- Verify virtual environment is activated

### Issue: "Import errors" in test mode

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Server starts but Claude Desktop doesn't see it

**Check:**
1. Config file exists: `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Paths in config are **absolute** (not relative)
3. Python path points to your virtual environment
4. Restart Claude Desktop completely

### Issue: "Command not found" in Claude Desktop logs

**Solution:**
- Update config file with full path to Python:
  ```json
  "command": "/Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats/venv/bin/python"
  ```

## Quick Status Check

Run this to verify everything:

```bash
cd /Users/prashanthmalipeddi/PycharmProjects/finance_trading_strats
source venv/bin/activate
python3 mcp_server.py --test
```

If all checks pass ✅, your server is ready!

## Summary

- ✅ **Test mode** (`--test`): Shows status and exits
- ✅ **Normal mode**: Shows startup messages, then waits (this is normal!)
- ✅ **With Claude Desktop**: Runs automatically in background
- ✅ **You don't need to manually start it** - Claude Desktop does that

The "hanging" behavior when running manually is **expected** - it means the server is working correctly and waiting for MCP protocol messages!

