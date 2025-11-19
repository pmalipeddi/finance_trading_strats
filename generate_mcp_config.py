#!/usr/bin/env python3
"""
Generate MCP configuration file for trading strategies server.

This script generates the mcp_config.json file with all available
trading strategies and their configurations.
"""

import json
import os
from typing import Dict, List, Any


def generate_strategy_config(strategy_name: str, description: str, 
                            tools: List[str], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Generate configuration for a trading strategy."""
    return {
        "name": strategy_name,
        "description": description,
        "tools": tools,
        "parameters": parameters
    }


def generate_mcp_config() -> Dict[str, Any]:
    """Generate complete MCP configuration."""
    
    # Mean Reversion Strategy
    mean_reversion_config = generate_strategy_config(
        strategy_name="mean_reversion",
        description="Mean reversion strategy that trades when prices deviate from their historical mean",
        tools=[
            "backtest_mean_reversion",
            "optimize_mean_reversion",
            "get_mean_reversion_signals"
        ],
        parameters={
            "lookback_period": {
                "default": 20,
                "description": "Number of periods for calculating mean and standard deviation"
            },
            "entry_threshold": {
                "default": 2.0,
                "description": "Number of standard deviations from mean to trigger entry"
            },
            "exit_threshold": {
                "default": 0.5,
                "description": "Number of standard deviations from mean to trigger exit"
            }
        }
    )
    
    # Moving Average Crossover Strategy
    ma_crossover_config = generate_strategy_config(
        strategy_name="moving_average_crossover",
        description="Moving average crossover strategy that generates signals when short MA crosses long MA",
        tools=[
            "backtest_moving_average_crossover",
            "optimize_moving_average_crossover",
            "get_moving_average_crossover_signals"
        ],
        parameters={
            "short_window": {
                "default": 50,
                "description": "Period for short-term moving average"
            },
            "long_window": {
                "default": 200,
                "description": "Period for long-term moving average"
            }
        }
    )
    
    # Momentum Price & Volume Strategy
    momentum_price_vol_config = generate_strategy_config(
        strategy_name="momentum_price_vol",
        description="Momentum breakout strategy based on price momentum and volume surge analysis",
        tools=[
            "backtest_momentum_price_vol",
            "optimize_momentum_price_vol",
            "get_momentum_price_vol_signals"
        ],
        parameters={
            "price_lookback": {
                "default": 20,
                "description": "Lookback period for calculating price momentum"
            },
            "volume_lookback": {
                "default": 20,
                "description": "Lookback period for calculating average volume"
            },
            "price_threshold": {
                "default": 5.0,
                "description": "Minimum percentage price change to trigger momentum signal"
            },
            "volume_threshold": {
                "default": 1.5,
                "description": "Volume multiplier to identify volume surge (e.g., 1.5 = 150% of average)"
            }
        }
    )
    
    # Complete MCP configuration
    config = {
        "mcpServers": {
            "trading-strategies": {
                "command": "python",
                "args": ["mcp_server.py"],
                "description": "Trading strategies MCP server with multiple strategy implementations",
                "strategies": [
                    mean_reversion_config,
                    ma_crossover_config,
                    momentum_price_vol_config
                ],
                "version": "1.3.0",
                "capabilities": [
                    "backtesting",
                    "optimization",
                    "signal_generation"
                ]
            }
        }
    }
    
    return config


def main():
    """Generate and save MCP configuration file."""
    config = generate_mcp_config()
    
    # Write to file
    output_file = "mcp_config.json"
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ MCP configuration generated successfully: {output_file}")
    print(f"\n📊 Strategies configured:")
    for strategy in config["mcpServers"]["trading-strategies"]["strategies"]:
        print(f"  - {strategy['name']}: {len(strategy['tools'])} tools")
    
    print(f"\n🔧 Total capabilities: {len(config['mcpServers']['trading-strategies']['capabilities'])}")
    print(f"📦 Version: {config['mcpServers']['trading-strategies']['version']}")


if __name__ == "__main__":
    main()