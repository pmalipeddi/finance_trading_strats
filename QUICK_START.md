# Quick Start Guide - MCP Server Setup

## Option 1: Automated Setup (Recommended)

Run the helper script to automatically configure Claude Desktop:

```bash
python generate_mcp_config.py
```

This script will:
- Find your Python executable (including virtual environment)
- Generate the correct configuration
- Optionally write it to Claude Desktop config file
- Show you exactly what was configured

## Option 2: Manual Setup

### Step 1: Find Claude Desktop Config File

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

### Step 2: Add This Configuration

Open the config file and add/update the `mcpServers` section:

```json
{
  "mcpServers": {
    "trading-strategies": {
      "command": "/full/path/to/venv/bin/python",
      "args": [
        "/full/path/to/finance_trading_strats/mcp_server.py"
      ],
      "cwd": "/full/path/to/finance_trading_strats",
      "env": {}
    }
  }
}
```

**Important:** Replace `/full/path/to/` with your actual project path!

### Step 3: Restart Claude Desktop

1. Quit Claude Desktop completely
2. Reopen it
3. The MCP server should connect automatically

### Step 4: Test It

In Claude Desktop, ask:
```
What MCP tools do you have available?
```

Then try:
```
Get the moving average crossover signal for AAPL
```

## Finding Your Paths

### Get Project Path:
```bash
pwd
# Copy the output
```

### Get Python Path (if using venv):
```bash
which python
# or
which python3
```

### Get Python Path (Windows):
```powershell
where python
```

## Common Issues

**"Command not found"**
- Use full path to Python executable
- Make sure you're using Python from your virtual environment

**"Import errors"**
- Run: `pip install -r requirements.txt`
- Make sure you're using the correct Python executable

**"Server not connecting"**
- Check that paths in config are absolute (not relative)
- Verify JSON syntax is valid
- Restart Claude Desktop completely

## Need More Help?

See `CLAUDE_DESKTOP_SETUP.md` for detailed instructions and troubleshooting.

