#!/usr/bin/env python3
"""
Helper script to generate Claude Desktop MCP configuration
"""
import os
import json
import sys
from pathlib import Path

def find_python_executable():
    """Find the Python executable to use"""
    # First, try to use the current Python interpreter
    python_exe = sys.executable
    
    # Check if we're in a virtual environment
    venv_python = Path(__file__).parent / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python.absolute())
    
    # Check for venv on Windows
    venv_python_win = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    if venv_python_win.exists():
        return str(venv_python_win.absolute())
    
    # Fall back to current Python
    return python_exe

def get_claude_config_path():
    """Get the path to Claude Desktop config file based on OS"""
    system = sys.platform
    
    if system == "darwin":  # macOS
        config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "win32":  # Windows
        config_path = Path(os.getenv("APPDATA")) / "Claude" / "claude_desktop_config.json"
    else:  # Linux
        config_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    
    return config_path

def generate_config():
    """Generate MCP server configuration"""
    project_root = Path(__file__).parent.absolute()
    mcp_server_path = project_root / "mcp_server.py"
    python_exe = find_python_executable()
    
    config = {
        "mcpServers": {
            "trading-strategies": {
                "command": python_exe,
                "args": [str(mcp_server_path)],
                "cwd": str(project_root),
                "env": {}
            }
        }
    }
    
    return config

def main():
    """Main function"""
    print("Trading Strategies MCP Server - Configuration Generator")
    print("=" * 60)
    
    project_root = Path(__file__).parent.absolute()
    mcp_server_path = project_root / "mcp_server.py"
    
    # Check if mcp_server.py exists
    if not mcp_server_path.exists():
        print(f"Error: mcp_server.py not found at {mcp_server_path}")
        sys.exit(1)
    
    # Generate configuration
    config = generate_config()
    python_exe = find_python_executable()
    
    print(f"\nProject Root: {project_root}")
    print(f"MCP Server: {mcp_server_path}")
    print(f"Python Executable: {python_exe}")
    print("\nGenerated Configuration:")
    print(json.dumps(config, indent=2))
    
    # Get Claude config path
    claude_config_path = get_claude_config_path()
    print(f"\nClaude Desktop Config Location: {claude_config_path}")
    
    # Check if config file exists
    if claude_config_path.exists():
        print(f"\n⚠️  Warning: Config file already exists at {claude_config_path}")
        print("You have two options:")
        print("1. Manually merge this configuration into your existing file")
        print("2. Backup your existing file and replace it")
        
        response = input("\nDo you want to see the existing config? (y/n): ").lower()
        if response == 'y':
            with open(claude_config_path, 'r') as f:
                existing_config = json.load(f)
                print("\nExisting Configuration:")
                print(json.dumps(existing_config, indent=2))
    else:
        print(f"\nConfig file doesn't exist yet. It will be created when you add this configuration.")
    
    # Create config directory if it doesn't exist
    claude_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ask if user wants to write the config
    print("\n" + "=" * 60)
    response = input("Do you want to write this configuration to Claude Desktop config? (y/n): ").lower()
    
    if response == 'y':
        # Read existing config if it exists
        existing_config = {}
        if claude_config_path.exists():
            try:
                with open(claude_config_path, 'r') as f:
                    existing_config = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Existing config file is not valid JSON. It will be backed up.")
                backup_path = claude_config_path.with_suffix('.json.backup')
                claude_config_path.rename(backup_path)
                print(f"Backup saved to: {backup_path}")
        
        # Merge configurations
        if "mcpServers" not in existing_config:
            existing_config["mcpServers"] = {}
        
        existing_config["mcpServers"]["trading-strategies"] = config["mcpServers"]["trading-strategies"]
        
        # Write configuration
        with open(claude_config_path, 'w') as f:
            json.dump(existing_config, f, indent=2)
        
        print(f"\n✅ Configuration written to: {claude_config_path}")
        print("\nNext steps:")
        print("1. Restart Claude Desktop completely")
        print("2. Open a new conversation")
        print("3. Ask Claude: 'What MCP tools do you have available?'")
        print("4. Test with: 'Get the moving average crossover signal for AAPL'")
    else:
        print("\nConfiguration not written. Copy the JSON above and add it to your Claude Desktop config manually.")
        print(f"\nConfig file location: {claude_config_path}")
        print("\nTo add manually:")
        print("1. Open the config file in a text editor")
        print("2. Add the 'trading-strategies' entry to the 'mcpServers' section")
        print("3. Save and restart Claude Desktop")

if __name__ == "__main__":
    main()

