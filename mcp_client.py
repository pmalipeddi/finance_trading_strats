#!/usr/bin/env python3
"""
MCP Client Example
Demonstrates how to connect to and use the MCP server
"""
import asyncio
import json
import sys
from typing import Any, Dict

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("Error: MCP SDK not installed. Please install it with: pip install mcp", file=sys.stderr)
    sys.exit(1)


async def run_client():
    """
    Example MCP client that connects to the trading strategies MCP server.
    """
    # Configure the server to connect to
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )
    
    print("Connecting to MCP server...")
    
    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            print("✓ Connected to MCP server")
            print("\n" + "="*70)
            print("Available Tools:")
            print("="*70)
            
            # List available tools
            tools = await session.list_tools()
            
            for tool in tools.tools:
                print(f"\n{tool.name}")
                print(f"  Description: {tool.description}")
                if tool.inputSchema:
                    props = tool.inputSchema.get("properties", {})
                    if props:
                        print("  Parameters:")
                        for param_name, param_info in props.items():
                            param_desc = param_info.get("description", "")
                            param_default = param_info.get("default", "")
                            default_str = f" (default: {param_default})" if param_default else ""
                            print(f"    - {param_name}: {param_desc}{default_str}")
            
            print("\n" + "="*70)
            print("Example Tool Calls:")
            print("="*70)
            
            # Example 1: Get moving average crossover for a single stock
            print("\n1. Getting moving average crossover for AAPL...")
            try:
                result = await session.call_tool(
                    "get_moving_average_crossover",
                    arguments={
                        "symbol": "AAPL",
                        "fast_period": 50,
                        "slow_period": 200
                    }
                )
                if result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
            except Exception as e:
                print(f"Error: {e}")
            
            # Example 2: Get mean reversion signal for a single stock
            print("\n2. Getting mean reversion signal for TSLA (Bollinger Bands, 20-day)...")
            try:
                result = await session.call_tool(
                    "get_mean_reversion_signal",
                    arguments={
                        "symbol": "TSLA",
                        "method": "bollinger",
                        "window": 20
                    }
                )
                if result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
            except Exception as e:
                print(f"Error: {e}")
            
            # Example 3: Get mean reversion with different window
            print("\n3. Getting mean reversion signal for MSFT (RSI, 14-day)...")
            try:
                result = await session.call_tool(
                    "get_mean_reversion_signal",
                    arguments={
                        "symbol": "MSFT",
                        "method": "rsi",
                        "window": 14
                    }
                )
                if result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
            except Exception as e:
                print(f"Error: {e}")
            
            print("\n" + "="*70)
            print("Client session complete!")
            print("="*70)


async def interactive_client():
    """
    Interactive MCP client that allows you to call tools interactively.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )
    
    print("="*70)
    print("Interactive MCP Client")
    print("="*70)
    print("\nConnecting to MCP server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("✓ Connected!")
            
            # List tools
            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            
            print(f"\nAvailable tools: {', '.join(tool_names)}")
            print("\nExample usage:")
            print("  Tool: get_moving_average_crossover")
            print("  Arguments: {\"symbol\": \"AAPL\", \"fast_period\": 50, \"slow_period\": 200}")
            print("\nType 'exit' to quit, 'list' to see tools, or call a tool directly.")
            print("="*70)
            
            while True:
                try:
                    user_input = input("\n> ").strip()
                    
                    if user_input.lower() == 'exit':
                        break
                    
                    if user_input.lower() == 'list':
                        print("\nAvailable Tools:")
                        for tool in tools.tools:
                            print(f"  - {tool.name}: {tool.description}")
                        continue
                    
                    # Try to parse as JSON (tool call)
                    try:
                        # Format: tool_name {arguments}
                        if '{' in user_input:
                            parts = user_input.split('{', 1)
                            tool_name = parts[0].strip()
                            args_json = '{' + parts[1]
                            arguments = json.loads(args_json)
                        else:
                            # Simple format: tool_name symbol=value
                            parts = user_input.split()
                            tool_name = parts[0]
                            arguments = {}
                            for part in parts[1:]:
                                if '=' in part:
                                    key, value = part.split('=', 1)
                                    # Try to convert to appropriate type
                                    try:
                                        if '.' in value:
                                            arguments[key] = float(value)
                                        else:
                                            arguments[key] = int(value)
                                    except ValueError:
                                        arguments[key] = value
                        
                        if tool_name not in tool_names:
                            print(f"Error: Unknown tool '{tool_name}'")
                            print(f"Available tools: {', '.join(tool_names)}")
                            continue
                        
                        print(f"\nCalling {tool_name}...")
                        result = await session.call_tool(tool_name, arguments)
                        
                        if result.content:
                            for content in result.content:
                                if hasattr(content, 'text'):
                                    print(content.text)
                        else:
                            print("No content returned")
                    
                    except json.JSONDecodeError:
                        print("Error: Invalid JSON format. Use: tool_name {\"arg\": \"value\"}")
                    except Exception as e:
                        print(f"Error: {e}")
                
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except EOFError:
                    break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_client())
    else:
        asyncio.run(run_client())

