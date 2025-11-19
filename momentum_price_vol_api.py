from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import os
from momentum_price_vol_strategy import MomentumBreakoutStrategy
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Momentum Breakout Strategy API",
    description="API for momentum breakout trading strategy based on price and volume",
    version="1.0.0"
)


class StrategyParameters(BaseModel):
    """Model for strategy parameters."""
    lookback_period: int = Field(default=20, ge=5, le=200, description="Lookback period for high/low")
    volume_threshold: float = Field(default=1.5, ge=1.0, le=5.0, description="Volume threshold multiplier")
    breakout_threshold: float = Field(default=0.02, ge=0.0, le=0.1, description="Breakout threshold percentage")
    stop_loss_pct: float = Field(default=0.05, ge=0.0, le=0.2, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.10, ge=0.0, le=0.5, description="Take profit percentage")


class StockAnalysisRequest(BaseModel):
    """Model for stock analysis request."""
    symbol: str = Field(..., description="Stock ticker symbol")
    parameters: Optional[StrategyParameters] = None


class BulkAnalysisRequest(BaseModel):
    """Model for bulk analysis request."""
    symbols: List[str] = Field(..., description="List of stock ticker symbols")
    parameters: Optional[StrategyParameters] = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Momentum Breakout Strategy API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze/{symbol}",
            "/analyze/bulk",
            "/analyze/sp500",
            "/analyze/nasdaq100",
            "/health"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "momentum-breakout-strategy"}


@app.get("/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    lookback_period: int = Query(default=20, ge=5, le=200),
    volume_threshold: float = Query(default=1.5, ge=1.0, le=5.0),
    breakout_threshold: float = Query(default=0.02, ge=0.0, le=0.1),
    stop_loss_pct: float = Query(default=0.05, ge=0.0, le=0.2),
    take_profit_pct: float = Query(default=0.10, ge=0.0, le=0.5)
):
    """
    Analyze a single stock and return momentum breakout signal.

    Args:
        symbol: Stock ticker symbol
        lookback_period: Lookback period for high/low calculation
        volume_threshold: Volume threshold multiplier
        breakout_threshold: Breakout threshold percentage
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage

    Returns:
        Analysis result with signal and metrics
    """
    try:
        strategy = MomentumBreakoutStrategy(
            lookback_period=lookback_period,
            volume_threshold=volume_threshold,
            breakout_threshold=breakout_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

        result = strategy.analyze_stock(symbol.upper())

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/bulk")
async def analyze_bulk(request: BulkAnalysisRequest):
    """
    Analyze multiple stocks and return momentum breakout signals.

    Args:
        request: Bulk analysis request with symbols and parameters

    Returns:
        List of analysis results
    """
    try:
        params = request.parameters or StrategyParameters()
        strategy = MomentumBreakoutStrategy(
            lookback_period=params.lookback_period,
            volume_threshold=params.volume_threshold,
            breakout_threshold=params.breakout_threshold,
            stop_loss_pct=params.stop_loss_pct,
            take_profit_pct=params.take_profit_pct
        )

        results = []
        for symbol in request.symbols:
            result = strategy.analyze_stock(symbol.upper())
            results.append(result)

        return JSONResponse(content={"results": results, "count": len(results)})

    except Exception as e:
        logger.error(f"Error in bulk analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/sp500")
async def analyze_sp500(
    lookback_period: int = Query(default=20, ge=5, le=200),
    volume_threshold: float = Query(default=1.5, ge=1.0, le=5.0),
    breakout_threshold: float = Query(default=0.02, ge=0.0, le=0.1),
    stop_loss_pct: float = Query(default=0.05, ge=0.0, le=0.2),
    take_profit_pct: float = Query(default=0.10, ge=0.0, le=0.5),
    return_file: bool = Query(default=False, description="Return CSV file instead of JSON")
):
    """
    Analyze all S&P 500 stocks and return momentum breakout signals.

    Args:
        lookback_period: Lookback period for high/low calculation
        volume_threshold: Volume threshold multiplier
        breakout_threshold: Breakout threshold percentage
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        return_file: If True, return CSV file for download

    Returns:
        Analysis results for all S&P 500 stocks
    """
    try:
        strategy = MomentumBreakoutStrategy(
            lookback_period=lookback_period,
            volume_threshold=volume_threshold,
            breakout_threshold=breakout_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

        output_file = 'sp500_momentum_signals.csv'
        df_results = strategy.analyze_sp500_stocks(output_file)

        if return_file:
            if os.path.exists(output_file):
                return FileResponse(
                    output_file,
                    media_type='text/csv',
                    filename=output_file
                )
            else:
                raise HTTPException(status_code=404, detail="Results file not found")

        results = df_results.to_dict('records')
        buy_signals = len(df_results[df_results['signal'] == 'BUY'])
        sell_signals = len(df_results[df_results['signal'] == 'SELL'])
        hold_signals = len(df_results[df_results['signal'] == 'HOLD'])

        return JSONResponse(content={
            "results": results,
            "summary": {
                "total_stocks": len(results),
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals
            },
            "file": output_file
        })

    except Exception as e:
        logger.error(f"Error analyzing S&P 500: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/nasdaq100")
async def analyze_nasdaq100(
    lookback_period: int = Query(default=20, ge=5, le=200),
    volume_threshold: float = Query(default=1.5, ge=1.0, le=5.0),
    breakout_threshold: float = Query(default=0.02, ge=0.0, le=0.1),
    stop_loss_pct: float = Query(default=0.05, ge=0.0, le=0.2),
    take_profit_pct: float = Query(default=0.10, ge=0.0, le=0.5),
    return_file: bool = Query(default=False, description="Return CSV file instead of JSON")
):
    """
    Analyze all NASDAQ 100 stocks and return momentum breakout signals.

    Args:
        lookback_period: Lookback period for high/low calculation
        volume_threshold: Volume threshold multiplier
        breakout_threshold: Breakout threshold percentage
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        return_file: If True, return CSV file for download

    Returns:
        Analysis results for all NASDAQ 100 stocks
    """
    try:
        strategy = MomentumBreakoutStrategy(
            lookback_period=lookback_period,
            volume_threshold=volume_threshold,
            breakout_threshold=breakout_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )

        output_file = 'nasdaq100_momentum_signals.csv'
        df_results = strategy.analyze_nasdaq100_stocks(output_file)

        if return_file:
            if os.path.exists(output_file):
                return FileResponse(
                    output_file,
                    media_type='text/csv',
                    filename=output_file
                )
            else:
                raise HTTPException(status_code=404, detail="Results file not found")

        results = df_results.to_dict('records')
        buy_signals = len(df_results[df_results['signal'] == 'BUY'])
        sell_signals = len(df_results[df_results['signal'] == 'SELL'])
        hold_signals = len(df_results[df_results['signal'] == 'HOLD'])

        return JSONResponse(content={
            "results": results,
            "summary": {
                "total_stocks": len(results),
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals
            },
            "file": output_file
        })

    except Exception as e:
        logger.error(f"Error analyzing NASDAQ 100: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)