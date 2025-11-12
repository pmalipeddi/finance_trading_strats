from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List
from pydantic import BaseModel
import pandas as pd
from moving_average_crossover_strategy import TradingStrategies

app = FastAPI(
    title="Trading Strategies API",
    description="API for moving average crossover trading signals",
    version="1.0.0"
)

# Initialize the trading strategies instance
strategies = TradingStrategies()


class CrossoverResponse(BaseModel):
    """Response model for single ticker crossover signal"""
    symbol: str
    current_price: float
    fast_ma: Optional[float]
    slow_ma: Optional[float]
    current_signal: str
    fast_period: int
    slow_period: int
    total_signals: int
    buy_signals: int
    sell_signals: int


class CrossoverRequest(BaseModel):
    """Request model for crossover signal"""
    symbol: str
    fast_period: int = 50
    slow_period: int = 200
    period: str = "2y"
    interval: str = "1d"


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Trading Strategies API",
        "version": "1.0.0",
        "endpoints": {
            "/signal/{symbol}": "Get crossover signal for a single ticker",
            "/signal": "Get crossover signal (POST with body)",
            "/sp500": "Get crossover signals for all S&P 500 stocks",
            "/nasdaq100": "Get crossover signals for all NASDAQ-100 stocks"
        }
    }


@app.get("/signal/{symbol}", response_model=CrossoverResponse)
async def get_crossover_signal(
    symbol: str,
    fast_period: int = Query(50, ge=1, le=500, description="Fast moving average period"),
    slow_period: int = Query(200, ge=1, le=500, description="Slow moving average period"),
    period: str = Query("2y", description="Historical data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)")
):
    """
    Get moving average crossover signal for a single ticker symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
        fast_period: Period for fast moving average (default: 50)
        slow_period: Period for slow moving average (default: 200)
        period: Historical data period (default: "2y")
        interval: Data interval (default: "1d")
    
    Returns:
        CrossoverResponse with signal information
    """
    try:
        result = strategies.moving_average_crossover(
            symbol=symbol.upper(),
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
            interval=interval
        )
        
        return CrossoverResponse(
            symbol=result['symbol'],
            current_price=result['current_price'],
            fast_ma=result.get('fast_ma'),
            slow_ma=result.get('slow_ma'),
            current_signal=result['current_signal'],
            fast_period=result['fast_period'],
            slow_period=result['slow_period'],
            total_signals=result['total_signals'],
            buy_signals=result['buy_signals'],
            sell_signals=result['sell_signals']
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/signal", response_model=CrossoverResponse)
async def post_crossover_signal(request: CrossoverRequest):
    """
    Get moving average crossover signal for a single ticker symbol (POST method).
    
    Args:
        request: CrossoverRequest with symbol and parameters
    
    Returns:
        CrossoverResponse with signal information
    """
    try:
        result = strategies.moving_average_crossover(
            symbol=request.symbol.upper(),
            fast_period=request.fast_period,
            slow_period=request.slow_period,
            period=request.period,
            interval=request.interval
        )
        
        return CrossoverResponse(
            symbol=result['symbol'],
            current_price=result['current_price'],
            fast_ma=result.get('fast_ma'),
            slow_ma=result.get('slow_ma'),
            current_signal=result['current_signal'],
            fast_period=result['fast_period'],
            slow_period=result['slow_period'],
            total_signals=result['total_signals'],
            buy_signals=result['buy_signals'],
            sell_signals=result['sell_signals']
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/sp500")
async def get_sp500_signals(
    fast_period: int = Query(50, ge=1, le=500, description="Fast moving average period"),
    slow_period: int = Query(200, ge=1, le=500, description="Slow moving average period"),
    period: str = Query("2y", description="Historical data period"),
    interval: str = Query("1d", description="Data interval"),
    max_workers: int = Query(20, ge=1, le=50, description="Number of concurrent workers"),
    return_dataframe: bool = Query(False, description="Return full DataFrame as JSON")
):
    """
    Get moving average crossover signals for all S&P 500 stocks.
    
    This endpoint processes all S&P 500 tickers in parallel and returns the results.
    Processing may take several minutes.
    
    Args:
        fast_period: Period for fast moving average (default: 50)
        slow_period: Period for slow moving average (default: 200)
        period: Historical data period (default: "2y")
        interval: Data interval (default: "1d")
        max_workers: Number of concurrent workers for parallel processing (default: 20)
        return_dataframe: If True, returns full DataFrame as JSON (default: False)
    
    Returns:
        JSON response with DataFrame data or summary statistics
    """
    try:
        df = strategies.moving_average_crossover_sp500(
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
            interval=interval,
            csv_path="sp500_moving_average_signals.csv",
            max_workers=max_workers
        )
        
        if return_dataframe:
            # Return full DataFrame as JSON
            return JSONResponse(
                content={
                    "status": "success",
                    "total_tickers": len(df),
                    "data": df.to_dict(orient="records")
                }
            )
        else:
            # Return summary statistics
            signal_counts = df['Signal'].value_counts().to_dict()
            return {
                "status": "success",
                "total_tickers": len(df),
                "signal_distribution": signal_counts,
                "csv_file": "sp500_moving_average_signals.csv",
                "message": "Full data saved to CSV file. Use return_dataframe=true to get JSON data."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing S&P 500 signals: {str(e)}")


@app.get("/nasdaq100")
async def get_nasdaq100_signals(
    fast_period: int = Query(50, ge=1, le=500, description="Fast moving average period"),
    slow_period: int = Query(200, ge=1, le=500, description="Slow moving average period"),
    period: str = Query("2y", description="Historical data period"),
    interval: str = Query("1d", description="Data interval"),
    max_workers: int = Query(20, ge=1, le=50, description="Number of concurrent workers"),
    return_dataframe: bool = Query(False, description="Return full DataFrame as JSON")
):
    """
    Get moving average crossover signals for all NASDAQ-100 stocks.
    
    This endpoint processes all NASDAQ-100 tickers in parallel and returns the results.
    Processing may take several minutes.
    
    Args:
        fast_period: Period for fast moving average (default: 50)
        slow_period: Period for slow moving average (default: 200)
        period: Historical data period (default: "2y")
        interval: Data interval (default: "1d")
        max_workers: Number of concurrent workers for parallel processing (default: 20)
        return_dataframe: If True, returns full DataFrame as JSON (default: False)
    
    Returns:
        JSON response with DataFrame data or summary statistics
    """
    try:
        df = strategies.moving_average_crossover_nasdaq100(
            fast_period=fast_period,
            slow_period=slow_period,
            period=period,
            interval=interval,
            csv_path="nasdaq100_moving_average_signals.csv",
            max_workers=max_workers
        )
        
        if return_dataframe:
            # Return full DataFrame as JSON
            return JSONResponse(
                content={
                    "status": "success",
                    "total_tickers": len(df),
                    "data": df.to_dict(orient="records")
                }
            )
        else:
            # Return summary statistics
            signal_counts = df['Signal'].value_counts().to_dict()
            return {
                "status": "success",
                "total_tickers": len(df),
                "signal_distribution": signal_counts,
                "csv_file": "nasdaq100_moving_average_signals.csv",
                "message": "Full data saved to CSV file. Use return_dataframe=true to get JSON data."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NASDAQ-100 signals: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Trading Strategies API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

