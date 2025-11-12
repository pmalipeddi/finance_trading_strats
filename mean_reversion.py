from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List
from pydantic import BaseModel
import pandas as pd
from mean_reversion_strategy import MeanReversionStrategy

app = FastAPI(
    title="Mean Reversion Trading Strategy API",
    description="API for mean reversion trading signals using Bollinger Bands, RSI, Z-Score, and Combined methods",
    version="1.0.0"
)

# Initialize the mean reversion strategy instance
strategy = MeanReversionStrategy()


class MeanReversionResponse(BaseModel):
    """Response model for single ticker mean reversion signal"""
    symbol: str
    current_price: float
    current_signal: str
    method: str
    window: int
    total_signals: int
    buy_signals: int
    sell_signals: int
    upper_band: Optional[float] = None
    middle_band: Optional[float] = None
    lower_band: Optional[float] = None
    rsi: Optional[float] = None
    z_score: Optional[float] = None


class MeanReversionRequest(BaseModel):
    """Request model for mean reversion signal"""
    symbol: str
    method: str = "bollinger"
    window: int = 20
    period: str = "1y"
    interval: str = "1d"
    upper_threshold: float = 2.0
    lower_threshold: float = -2.0
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Mean Reversion Trading Strategy API",
        "version": "1.0.0",
        "endpoints": {
            "/signal/{symbol}": "Get mean reversion signal for a single ticker",
            "/signal": "Get mean reversion signal (POST with body)",
            "/sp500": "Get mean reversion signals for all S&P 500 stocks",
            "/nasdaq100": "Get mean reversion signals for all NASDAQ-100 stocks"
        },
        "methods": {
            "bollinger": "Bollinger Bands mean reversion",
            "rsi": "RSI (Relative Strength Index) mean reversion",
            "zscore": "Z-Score mean reversion",
            "combined": "Combined method using all three indicators"
        },
        "window_suggestions": {
            "short_term": 20,
            "medium_term": 50,
            "long_term": 100
        }
    }


@app.get("/signal/{symbol}", response_model=MeanReversionResponse)
async def get_mean_reversion_signal(
    symbol: str,
    method: str = Query("bollinger", description="Method: bollinger, rsi, zscore, or combined"),
    window: int = Query(20, ge=5, le=200, description="Window size (e.g., 20=short, 50=medium, 100=long term)"),
    period: str = Query("1y", description="Historical data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    upper_threshold: float = Query(2.0, description="Upper threshold for Z-score/Bollinger"),
    lower_threshold: float = Query(-2.0, description="Lower threshold for Z-score/Bollinger"),
    rsi_overbought: float = Query(70.0, ge=50, le=90, description="RSI overbought threshold"),
    rsi_oversold: float = Query(30.0, ge=10, le=50, description="RSI oversold threshold")
):
    """
    Get mean reversion signal for a single ticker symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
        method: Method to use - bollinger, rsi, zscore, or combined (default: bollinger)
        window: Window size for calculations (default: 20)
               - Short term: 20 days
               - Medium term: 50 days
               - Long term: 100 days
        period: Historical data period (default: "1y")
        interval: Data interval (default: "1d")
        upper_threshold: Upper threshold for Z-score/Bollinger (default: 2.0)
        lower_threshold: Lower threshold for Z-score/Bollinger (default: -2.0)
        rsi_overbought: RSI overbought threshold (default: 70.0)
        rsi_oversold: RSI oversold threshold (default: 30.0)
    
    Returns:
        MeanReversionResponse with signal information
    """
    if method not in ["bollinger", "rsi", "zscore", "combined"]:
        raise HTTPException(status_code=400, detail=f"Invalid method: {method}. Must be one of: bollinger, rsi, zscore, combined")
    
    try:
        result = strategy.mean_reversion_signal(
            symbol=symbol.upper(),
            method=method,
            window=window,
            period=period,
            interval=interval,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold
        )
        
        return MeanReversionResponse(
            symbol=result['symbol'],
            current_price=result['current_price'],
            current_signal=result['current_signal'],
            method=result['method'],
            window=window,
            total_signals=result['total_signals'],
            buy_signals=result['buy_signals'],
            sell_signals=result['sell_signals'],
            upper_band=result.get('upper_band'),
            middle_band=result.get('middle_band'),
            lower_band=result.get('lower_band'),
            rsi=result.get('rsi'),
            z_score=result.get('z_score')
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/signal", response_model=MeanReversionResponse)
async def post_mean_reversion_signal(request: MeanReversionRequest):
    """
    Get mean reversion signal for a single ticker symbol (POST method).
    
    Args:
        request: MeanReversionRequest with symbol and parameters
    
    Returns:
        MeanReversionResponse with signal information
    """
    if request.method not in ["bollinger", "rsi", "zscore", "combined"]:
        raise HTTPException(status_code=400, detail=f"Invalid method: {request.method}. Must be one of: bollinger, rsi, zscore, combined")
    
    try:
        result = strategy.mean_reversion_signal(
            symbol=request.symbol.upper(),
            method=request.method,
            window=request.window,
            period=request.period,
            interval=request.interval,
            upper_threshold=request.upper_threshold,
            lower_threshold=request.lower_threshold,
            rsi_overbought=request.rsi_overbought,
            rsi_oversold=request.rsi_oversold
        )
        
        return MeanReversionResponse(
            symbol=result['symbol'],
            current_price=result['current_price'],
            current_signal=result['current_signal'],
            method=result['method'],
            window=request.window,
            total_signals=result['total_signals'],
            buy_signals=result['buy_signals'],
            sell_signals=result['sell_signals'],
            upper_band=result.get('upper_band'),
            middle_band=result.get('middle_band'),
            lower_band=result.get('lower_band'),
            rsi=result.get('rsi'),
            z_score=result.get('z_score')
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/sp500")
async def get_sp500_mean_reversion_signals(
    method: str = Query("bollinger", description="Method: bollinger, rsi, zscore, or combined"),
    window: int = Query(20, ge=5, le=200, description="Window size (e.g., 20=short, 50=medium, 100=long term)"),
    period: str = Query("1y", description="Historical data period"),
    interval: str = Query("1d", description="Data interval"),
    upper_threshold: float = Query(2.0, description="Upper threshold for Z-score/Bollinger"),
    lower_threshold: float = Query(-2.0, description="Lower threshold for Z-score/Bollinger"),
    rsi_overbought: float = Query(70.0, description="RSI overbought threshold"),
    rsi_oversold: float = Query(30.0, description="RSI oversold threshold"),
    max_workers: int = Query(20, ge=1, le=50, description="Number of concurrent workers"),
    return_dataframe: bool = Query(False, description="Return full DataFrame as JSON")
):
    """
    Get mean reversion signals for all S&P 500 stocks.
    
    This endpoint processes all S&P 500 tickers in parallel and returns the results.
    Processing may take several minutes.
    
    Args:
        method: Method to use - bollinger, rsi, zscore, or combined (default: bollinger)
        window: Window size for calculations (default: 20)
               - Short term: 20 days
               - Medium term: 50 days
               - Long term: 100 days
        period: Historical data period (default: "1y")
        interval: Data interval (default: "1d")
        upper_threshold: Upper threshold for Z-score/Bollinger (default: 2.0)
        lower_threshold: Lower threshold for Z-score/Bollinger (default: -2.0)
        rsi_overbought: RSI overbought threshold (default: 70.0)
        rsi_oversold: RSI oversold threshold (default: 30.0)
        max_workers: Number of concurrent workers for parallel processing (default: 20)
        return_dataframe: If True, returns full DataFrame as JSON (default: False)
    
    Returns:
        JSON response with DataFrame data or summary statistics
    """
    if method not in ["bollinger", "rsi", "zscore", "combined"]:
        raise HTTPException(status_code=400, detail=f"Invalid method: {method}. Must be one of: bollinger, rsi, zscore, combined")
    
    try:
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(project_root, f"sp500_mean_reversion_signals_{method}_w{window}.csv")
        
        df = strategy.mean_reversion_sp500(
            method=method,
            window=window,
            period=period,
            interval=interval,
            csv_path=csv_path,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            max_workers=max_workers
        )
        
        if return_dataframe:
            # Return full DataFrame as JSON
            return JSONResponse(
                content={
                    "status": "success",
                    "total_tickers": len(df),
                    "method": method,
                    "window": window,
                    "data": df.to_dict(orient="records")
                }
            )
        else:
            # Return summary statistics
            signal_counts = df['Signal'].value_counts().to_dict()
            return {
                "status": "success",
                "total_tickers": len(df),
                "method": method,
                "window": window,
                "signal_distribution": signal_counts,
                "csv_file": csv_path,
                "message": "Full data saved to CSV file. Use return_dataframe=true to get JSON data."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing S&P 500 signals: {str(e)}")


@app.get("/nasdaq100")
async def get_nasdaq100_mean_reversion_signals(
    method: str = Query("bollinger", description="Method: bollinger, rsi, zscore, or combined"),
    window: int = Query(20, ge=5, le=200, description="Window size (e.g., 20=short, 50=medium, 100=long term)"),
    period: str = Query("1y", description="Historical data period"),
    interval: str = Query("1d", description="Data interval"),
    upper_threshold: float = Query(2.0, description="Upper threshold for Z-score/Bollinger"),
    lower_threshold: float = Query(-2.0, description="Lower threshold for Z-score/Bollinger"),
    rsi_overbought: float = Query(70.0, description="RSI overbought threshold"),
    rsi_oversold: float = Query(30.0, description="RSI oversold threshold"),
    max_workers: int = Query(20, ge=1, le=50, description="Number of concurrent workers"),
    return_dataframe: bool = Query(False, description="Return full DataFrame as JSON")
):
    """
    Get mean reversion signals for all NASDAQ-100 stocks.
    
    This endpoint processes all NASDAQ-100 tickers in parallel and returns the results.
    Processing may take several minutes.
    
    Args:
        method: Method to use - bollinger, rsi, zscore, or combined (default: bollinger)
        window: Window size for calculations (default: 20)
               - Short term: 20 days
               - Medium term: 50 days
               - Long term: 100 days
        period: Historical data period (default: "1y")
        interval: Data interval (default: "1d")
        upper_threshold: Upper threshold for Z-score/Bollinger (default: 2.0)
        lower_threshold: Lower threshold for Z-score/Bollinger (default: -2.0)
        rsi_overbought: RSI overbought threshold (default: 70.0)
        rsi_oversold: RSI oversold threshold (default: 30.0)
        max_workers: Number of concurrent workers for parallel processing (default: 20)
        return_dataframe: If True, returns full DataFrame as JSON (default: False)
    
    Returns:
        JSON response with DataFrame data or summary statistics
    """
    if method not in ["bollinger", "rsi", "zscore", "combined"]:
        raise HTTPException(status_code=400, detail=f"Invalid method: {method}. Must be one of: bollinger, rsi, zscore, combined")
    
    try:
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(project_root, f"nasdaq100_mean_reversion_signals_{method}_w{window}.csv")
        
        df = strategy.mean_reversion_nasdaq100(
            method=method,
            window=window,
            period=period,
            interval=interval,
            csv_path=csv_path,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            max_workers=max_workers
        )
        
        if return_dataframe:
            # Return full DataFrame as JSON
            return JSONResponse(
                content={
                    "status": "success",
                    "total_tickers": len(df),
                    "method": method,
                    "window": window,
                    "data": df.to_dict(orient="records")
                }
            )
        else:
            # Return summary statistics
            signal_counts = df['Signal'].value_counts().to_dict()
            return {
                "status": "success",
                "total_tickers": len(df),
                "method": method,
                "window": window,
                "signal_distribution": signal_counts,
                "csv_file": csv_path,
                "message": "Full data saved to CSV file. Use return_dataframe=true to get JSON data."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NASDAQ-100 signals: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Mean Reversion Trading Strategy API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

