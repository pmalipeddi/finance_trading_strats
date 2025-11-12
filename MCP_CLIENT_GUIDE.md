# MCP Client Guide

## Understanding MCP Client vs Server

- **MCP Server** (`mcp_server.py`): The service that provides tools (what we built)
- **MCP Client**: The application that connects to and uses the server
  - **Claude Desktop** is an MCP client (already built)
  - You can also build your own custom client

## Option 1: Use Claude Desktop (Recommended)

Claude Desktop is already a fully-featured MCP client. You just need to:
1. Configure it to connect to your server (already done)
2. Restart Claude Desktop
3. Use the tools through Claude's interface

**No need to build a client - Claude Desktop is your client!**

## Option 2: Build a Custom MCP Client

If you want to build your own client application, here's how:

### Simple Python Client

I've created `mcp_client.py` as an example. It demonstrates:

1. **Connecting to the MCP server**
2. **Listing available tools**
3. **Calling tools programmatically**

### Running the Example Client

```bash
# Run the example client (shows available tools and makes sample calls)
python3 mcp_client.py

# Run interactive client (allows you to call tools interactively)
python3 mcp_client.py --interactive
```

### Example Client Usage

The client will:
- Connect to your MCP server
- List all available tools
- Make example tool calls
- Show you how to use the tools programmatically

### Building Your Own Client

To build a custom client, you need:

1. **MCP SDK** (already installed):
   ```python
   from mcp import ClientSession, StdioServerParameters
   from mcp.client.stdio import stdio_client
   ```

2. **Connect to Server**:
   ```python
   server_params = StdioServerParameters(
       command="python",
       args=["mcp_server.py"]
   )
   
   async with stdio_client(server_params) as (read, write):
       async with ClientSession(read, write) as session:
           await session.initialize()
   ```

3. **List Tools**:
   ```python
   tools = await session.list_tools()
   ```

4. **Call Tools**:
   ```python
   result = await session.call_tool(
       "get_mean_reversion_signal",
       arguments={"symbol": "AAPL", "method": "bollinger", "window": 20}
   )
   ```

## Option 3: Use the FastAPI Endpoints

Instead of building an MCP client, you can use the REST API:

```bash
# Start the API server
python3 mean_reversion.py

# Or
uvicorn mean_reversion:app --host 0.0.0.0 --port 8001
```

Then make HTTP requests:
```bash
curl "http://localhost:8001/signal/AAPL?method=bollinger&window=20"
```

## Which Should You Use?

- **Claude Desktop**: Best for interactive use, natural language queries
- **Custom MCP Client**: Best if you want to integrate into your own application
- **FastAPI**: Best for web applications, HTTP-based integrations

## Testing the Client

Run the example client to see it in action:

```bash
python3 mcp_client.py
```

This will:
1. Connect to your MCP server
2. List all 6 available tools
3. Make example calls to demonstrate usage

## Next Steps

1. **For Claude Desktop**: Just restart it - it's already configured!
2. **For Custom Client**: Use `mcp_client.py` as a starting point
3. **For API**: Use the FastAPI endpoints in `mean_reversion.py` or `api.py`

The MCP server is ready - you just need to connect a client to it!

