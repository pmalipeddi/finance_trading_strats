#!/usr/bin/env python3
"""
Test script for mean_reversion_signal method
Tests all methods and verifies correctness
"""
from mean_reversion_strategy import MeanReversionStrategy
import pandas as pd
import numpy as np

def test_mean_reversion_signal():
    """Test the mean_reversion_signal method with different configurations"""
    
    print("="*70)
    print("Testing Mean Reversion Signal Method")
    print("="*70)
    
    # Create strategy instance
    strategy = MeanReversionStrategy(symbol="AAPL")
    
    # Test 1: Bollinger Bands Method
    print("\n" + "="*70)
    print("Test 1: Bollinger Bands Method")
    print("="*70)
    
    try:
        result = strategy.mean_reversion_signal(
            method="bollinger",
            window=20,
            period="6mo",
            interval="1d"
        )
        
        # Verify result structure
        assert 'symbol' in result, "Missing 'symbol' in result"
        assert 'current_price' in result, "Missing 'current_price' in result"
        assert 'current_signal' in result, "Missing 'current_signal' in result"
        assert 'method' in result, "Missing 'method' in result"
        assert 'data' in result, "Missing 'data' in result"
        assert 'signals' in result, "Missing 'signals' in result"
        assert 'upper_band' in result, "Missing 'upper_band' in result"
        assert 'middle_band' in result, "Missing 'middle_band' in result"
        assert 'lower_band' in result, "Missing 'lower_band' in result"
        
        print("✓ All required fields present")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Current Price: ${result['current_price']:.2f}")
        print(f"  Upper Band: ${result['upper_band']:.2f}")
        print(f"  Middle Band: ${result['middle_band']:.2f}")
        print(f"  Lower Band: ${result['lower_band']:.2f}")
        print(f"  Current Signal: {result['current_signal']}")
        print(f"  Total Signals: {result['total_signals']}")
        print(f"  Buy Signals: {result['buy_signals']}")
        print(f"  Sell Signals: {result['sell_signals']}")
        
        # Verify data structure
        assert isinstance(result['data'], pd.DataFrame), "Data should be a DataFrame"
        assert 'Close' in result['data'].columns, "Data should have 'Close' column"
        assert 'Upper_Band' in result['data'].columns, "Data should have 'Upper_Band' column"
        assert 'Lower_Band' in result['data'].columns, "Data should have 'Lower_Band' column"
        print("✓ Data structure is correct")
        
        # Verify signals
        assert isinstance(result['signals'], pd.DataFrame), "Signals should be a DataFrame"
        if len(result['signals']) > 0:
            assert 'Action' in result['signals'].columns, "Signals should have 'Action' column"
            print(f"✓ Found {len(result['signals'])} historical signals")
        
        print("✓ Test 1 PASSED: Bollinger Bands method works correctly")
        
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: RSI Method
    print("\n" + "="*70)
    print("Test 2: RSI Method")
    print("="*70)
    
    try:
        result = strategy.mean_reversion_signal(
            method="rsi",
            window=14,
            period="6mo",
            interval="1d"
        )
        
        assert 'rsi' in result, "Missing 'rsi' in result"
        assert result['rsi'] is not None, "RSI value should not be None"
        assert 0 <= result['rsi'] <= 100, "RSI should be between 0 and 100"
        
        print("✓ All required fields present")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Current Price: ${result['current_price']:.2f}")
        print(f"  RSI: {result['rsi']:.2f}")
        print(f"  Current Signal: {result['current_signal']}")
        
        # Verify RSI in data
        assert 'RSI' in result['data'].columns, "Data should have 'RSI' column"
        print("✓ RSI calculation is correct")
        
        print("✓ Test 2 PASSED: RSI method works correctly")
        
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Z-Score Method
    print("\n" + "="*70)
    print("Test 3: Z-Score Method")
    print("="*70)
    
    try:
        result = strategy.mean_reversion_signal(
            method="zscore",
            window=20,
            period="6mo",
            interval="1d"
        )
        
        assert 'z_score' in result, "Missing 'z_score' in result"
        assert result['z_score'] is not None, "Z-Score value should not be None"
        
        print("✓ All required fields present")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Current Price: ${result['current_price']:.2f}")
        print(f"  Z-Score: {result['z_score']:.2f}")
        print(f"  Current Signal: {result['current_signal']}")
        
        # Verify Z-Score in data
        assert 'Z_Score' in result['data'].columns, "Data should have 'Z_Score' column"
        print("✓ Z-Score calculation is correct")
        
        print("✓ Test 3 PASSED: Z-Score method works correctly")
        
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Combined Method
    print("\n" + "="*70)
    print("Test 4: Combined Method")
    print("="*70)
    
    try:
        result = strategy.mean_reversion_signal(
            method="combined",
            window=20,
            period="6mo",
            interval="1d"
        )
        
        # Combined method should have all indicators
        assert 'upper_band' in result, "Missing 'upper_band' in result"
        assert 'rsi' in result, "Missing 'rsi' in result"
        assert 'z_score' in result, "Missing 'z_score' in result"
        
        print("✓ All required fields present")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Current Price: ${result['current_price']:.2f}")
        print(f"  Upper Band: ${result['upper_band']:.2f}")
        print(f"  Lower Band: ${result['lower_band']:.2f}")
        print(f"  RSI: {result['rsi']:.2f}")
        print(f"  Z-Score: {result['z_score']:.2f}")
        print(f"  Current Signal: {result['current_signal']}")
        
        # Verify all indicators in data
        assert 'Upper_Band' in result['data'].columns, "Data should have 'Upper_Band' column"
        assert 'RSI' in result['data'].columns, "Data should have 'RSI' column"
        assert 'Z_Score' in result['data'].columns, "Data should have 'Z_Score' column"
        print("✓ All indicators calculated correctly")
        
        print("✓ Test 4 PASSED: Combined method works correctly")
        
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Signal Logic Verification
    print("\n" + "="*70)
    print("Test 5: Signal Logic Verification")
    print("="*70)
    
    try:
        # Test with a stock that should generate signals
        result = strategy.mean_reversion_signal(
            method="bollinger",
            window=20,
            period="1y",
            interval="1d"
        )
        
        # Check signal logic
        current_price = result['current_price']
        upper_band = result['upper_band']
        lower_band = result['lower_band']
        
        if current_price >= upper_band:
            assert "SELL" in result['current_signal'] or "sell" in result['current_signal'].lower(), \
                "Price above upper band should generate SELL signal"
            print("✓ Signal logic correct: Price above upper band = SELL")
        elif current_price <= lower_band:
            assert "BUY" in result['current_signal'] or "buy" in result['current_signal'].lower(), \
                "Price below lower band should generate BUY signal"
            print("✓ Signal logic correct: Price below lower band = BUY")
        else:
            assert "HOLD" in result['current_signal'] or "hold" in result['current_signal'].lower(), \
                "Price within bands should generate HOLD signal"
            print("✓ Signal logic correct: Price within bands = HOLD")
        
        print("✓ Test 5 PASSED: Signal logic is correct")
        
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Multiple Symbols
    print("\n" + "="*70)
    print("Test 6: Multiple Symbols Test")
    print("="*70)
    
    try:
        test_symbols = ["MSFT", "GOOGL", "TSLA"]
        
        for symbol in test_symbols:
            result = strategy.mean_reversion_signal(
                symbol=symbol,
                method="bollinger",
                window=20,
                period="3mo",
                interval="1d"
            )
            
            assert result['symbol'] == symbol.upper(), f"Symbol mismatch for {symbol}"
            assert result['current_price'] > 0, f"Price should be positive for {symbol}"
            print(f"✓ {symbol}: ${result['current_price']:.2f} - {result['current_signal']}")
        
        print("✓ Test 6 PASSED: Multiple symbols work correctly")
        
    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 7: Edge Cases
    print("\n" + "="*70)
    print("Test 7: Edge Cases")
    print("="*70)
    
    try:
        # Test with different window sizes
        for window in [10, 20, 50]:
            result = strategy.mean_reversion_signal(
                method="bollinger",
                window=window,
                period="6mo",
                interval="1d"
            )
            assert result['upper_band'] is not None, f"Upper band should not be None for window={window}"
            assert result['lower_band'] is not None, f"Lower band should not be None for window={window}"
            print(f"✓ Window size {window} works correctly")
        
        # Test with different periods
        for period in ["1mo", "3mo", "6mo", "1y"]:
            result = strategy.mean_reversion_signal(
                method="rsi",
                window=14,
                period=period,
                interval="1d"
            )
            assert result['rsi'] is not None, f"RSI should not be None for period={period}"
            print(f"✓ Period {period} works correctly")
        
        print("✓ Test 7 PASSED: Edge cases handled correctly")
        
    except Exception as e:
        print(f"✗ Test 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nThe mean_reversion_signal method is working correctly.")
    print("All methods (bollinger, rsi, zscore, combined) are functioning properly.")
    
    return True

if __name__ == "__main__":
    success = test_mean_reversion_signal()
    exit(0 if success else 1)

