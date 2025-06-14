from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import quantstats as qs
import yfinance as yf
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import matplotlib
matplotlib.use('Agg')
from fastapi_mcp import FastApiMCP
import httpx
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables for URL handling
IS_LOCAL_DEVELOPMENT = os.getenv('IS_LOCAL_DEVELOPMENT', '0') == '1'
BASE_URL_FOR_REPORTS = os.getenv('BASE_URL_FOR_REPORTS', 'https://quantstat-nextjs.vercel.app/')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Create a separate debug logger for detailed diagnostics
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler('debug.log')
debug_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
debug_logger.addHandler(debug_handler)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Monkey patch numpy.product
if not hasattr(np, 'product'):
    np.product = np.prod

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("=" * 40)
    logger.info("FastAPI server with MCP integration is starting!")
    logger.info("MCP endpoint is available at: http://localhost:8000/mcp")
    logger.info("Using custom httpx client with 3-minute (180 second) timeout")
    
    # Log all available routes and their operation IDs
    logger.info("Available routes and operation IDs in FastAPI app:")
    fastapi_operations = []
    for route in app.routes:
        if hasattr(route, "operation_id"):
            logger.info(f"Route: {route.path}, Operation ID: {route.operation_id}")
            fastapi_operations.append(route.operation_id)
    
    yield  # This is where the FastAPI app runs
    
    # Shutdown code
    logger.info("=" * 40)
    logger.info("FastAPI server is shutting down")
    logger.info("=" * 40)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="QuantStats Analysis API",
    description="API for generating portfolio analysis reports using QuantStats and yfinance data",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Ensure reports directory exists
REPORTS_DIR = os.path.join('static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Extend pandas
qs.extend_pandas()

# Pydantic models for API
class AnalysisRequest(BaseModel):
    """Request model for portfolio analysis."""
    stock_code: str = Field(
        description="Stock symbol to analyze (e.g., 'AAPL', 'GOOG', '^NSEI'). Must be a valid Yahoo Finance ticker symbol.",
        example="AAPL"
    )
    benchmark_code: str = Field(
        default="^GSPC",
        description="Benchmark symbol (default: S&P 500 Index). Must be a valid Yahoo Finance ticker symbol.",
        example="^GSPC"
    )
    start_date: str = Field(
        description="Start date for analysis in YYYY-MM-DD format. Should be at least 6 months before end_date for meaningful analysis.",
        example="2023-01-01"
    )
    end_date: str = Field(
        description="End date for analysis in YYYY-MM-DD format. Must be after start_date and not in the future.",
        example="2024-01-01"
    )
    risk_free_rate: float = Field(
        default=5.0,
        description="Risk-free rate as percentage (e.g., 5.0 for 5%). Used in Sharpe ratio calculations.",
        example=5.0,
        ge=0,
        le=100
    )

class AnalysisResponse(BaseModel):
    """Response model for portfolio analysis."""
    html_url: str = Field(
        description="URL to access the HTML report with visualizations",
        example="https://quantstat-nextjs.vercel.app/static/reports/AAPL_vs_^GSPC_20240315_123456.html"
    )

def process_stock_data(data, symbol):
    """Process stock data to get returns series with proper format"""
    try:
        debug_logger.debug(f"\nProcessing {symbol} data:")
        debug_logger.debug(f"Input data shape: {data.shape}")
        
        # Get the Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[('Close', symbol)]
        else:
            prices = data['Close']
        
        debug_logger.debug(f"Price series shape: {prices.shape}")
        debug_logger.debug(f"Price range: {prices.min():.2f} to {prices.max():.2f}")
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Convert to Series if it's not already
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
            
        # Name the series
        returns.name = symbol
        
        debug_logger.debug(f"Returns series shape: {returns.shape}")
        debug_logger.debug(f"Returns range: {returns.min():.4f} to {returns.max():.4f}")
        
        return returns
        
    except Exception as e:
        debug_logger.error(f"Error processing {symbol} data: {str(e)}", exc_info=True)
        raise

def generate_quantstats_report(stock_returns, benchmark_returns, output_file, title, risk_free_rate=5.0):
    """Generate QuantStats report with error handling"""
    try:
        debug_logger.debug("\nGenerating QuantStats report:")
        debug_logger.debug(f"Stock returns: {len(stock_returns)} periods")
        debug_logger.debug(f"Benchmark returns: {len(benchmark_returns)} periods")
        debug_logger.debug(f"Risk-free rate: {risk_free_rate:.2f}%")
        
        # Ensure the returns are properly aligned
        start_date = max(stock_returns.index.min(), benchmark_returns.index.min())
        end_date = min(stock_returns.index.max(), benchmark_returns.index.max())
        
        stock_returns = stock_returns[start_date:end_date]
        benchmark_returns = benchmark_returns[start_date:end_date]
        
        debug_logger.debug(f"Aligned period: {start_date} to {end_date}")
        debug_logger.debug(f"Final series lengths - Stock: {len(stock_returns)}, Benchmark: {len(benchmark_returns)}")
        
        # Convert annual rate to decimal
        rf_rate = risk_free_rate / 100.0
        
        # Generate the report
        qs.reports.html(
            stock_returns, 
            benchmark_returns,
            rf=rf_rate,
            output=output_file,
            title=title
        )
        
        return True
    except Exception as e:
        debug_logger.error("Error generating report:", exc_info=True)
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request,
    error: str = None,
    success: str = None,
    report_path: str = None,
    symbols: str = None
):
    """Serve the web interface"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error": error,
            "success": success,
            "report_generated": bool(report_path),
            "report_path": report_path,
            "symbols": symbols
        }
    )

def construct_report_url(filename: str) -> str:
    """Construct the appropriate URL for report files based on environment."""
    if IS_LOCAL_DEVELOPMENT:
        return f"/static/reports/{filename}"
    else:
        # Ensure BASE_URL_FOR_REPORTS ends with a slash
        base_url = BASE_URL_FOR_REPORTS.rstrip('/') + '/'
        return f"{base_url}static/reports/{filename}"

@app.get("/analyze", operation_id="create_portfolio_analysis", response_model=AnalysisResponse)
def analyze(
    symbols: str = Query(..., description="Stock symbol to analyze (e.g., 'AAPL'). Currently only supports analyzing one symbol at a time."),
    benchmark: str = Query(default="^GSPC", description="Benchmark symbol for comparison (default: ^GSPC for S&P 500 Index)"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    risk_free_rate: float = Query(default=5.0, description="Risk-free rate as percentage"),
    request: Request = None
):
    """
    Generate a comprehensive portfolio analysis report using QuantStats.
    
    This endpoint performs detailed portfolio analysis including:
    - Performance metrics (Sharpe ratio, Sortino ratio, etc.)
    - Risk metrics (Value at Risk, Maximum Drawdown, etc.)
    - Return analysis and distribution
    - Rolling statistics and correlations
    - Comparison against benchmark
    
    The analysis is returned as:
    - Interactive HTML report with visualizations
    - Links to view and download the report
    
    Parameters:
    - symbols: Stock symbol to analyze (e.g., 'AAPL'). Currently only supports analyzing one symbol at a time.
    - benchmark: Benchmark symbol for comparison (default: ^GSPC for S&P 500)
    - start_date: Analysis start date (YYYY-MM-DD)
    - end_date: Analysis end date (YYYY-MM-DD)
    - risk_free_rate: Risk-free rate percentage for calculations
    
    Returns:
    - AnalysisResponse with URL to access the generated HTML report
    
    Note: The analysis may take up to 2-3 minutes depending on the date range and data availability.
    """
    try:
        # Use only the first symbol for now (to match the working version's behavior)
        stock_symbols = symbols.split(',')
        if not stock_symbols:
            raise HTTPException(status_code=400, detail="Stock symbols are required")
        
        stock_code = stock_symbols[0].strip().upper()
        benchmark_code = benchmark.strip().upper()
        
        if not stock_code:
            raise HTTPException(status_code=400, detail="Stock symbol is required")
            
        logger.info(f"Processing request for stock: {stock_code} against benchmark: {benchmark_code}")
        debug_logger.debug(f"\nNew analysis request for {stock_code} vs {benchmark_code}")
        debug_logger.debug(f"Date range: {start_date} to {end_date}")
        debug_logger.debug(f"Risk-free rate: {risk_free_rate}%")
        
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f'{stock_code}_vs_{benchmark_code}_{timestamp}.html'
        report_path = os.path.join(REPORTS_DIR, report_filename)
        
        # Parse and validate dates
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # Add one day to end_date to include the last day in the analysis
            end = end + timedelta(days=1)
            
            if start >= end:
                raise HTTPException(status_code=400, detail="Start date must be before end date")
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
            
        debug_logger.debug(f"Downloading data for period: {start} to {end}")
        
        # Download stock data
        logger.info(f"Downloading data for {stock_code}")
        stock_data = yf.download(stock_code, start=start, end=end, progress=False)
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for stock {stock_code}")
            
        # Process stock returns
        stock_returns = process_stock_data(stock_data, stock_code)
        
        # Download benchmark data
        logger.info(f"Downloading data for benchmark {benchmark_code}")
        benchmark_data = yf.download(benchmark_code, start=start, end=end, progress=False)
        
        if benchmark_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {benchmark_code}")
            
        # Process benchmark returns
        benchmark_returns = process_stock_data(benchmark_data, benchmark_code)
        
        # Generate the report
        generate_quantstats_report(
            stock_returns,
            benchmark_returns,
            report_path,
            f'{stock_code} vs {benchmark_code} Analysis Report',
            risk_free_rate=risk_free_rate
        )
        
        logger.info(f"Successfully generated report for {stock_code} vs {benchmark_code}")
        
        # After successful report generation, construct the response
        report_url = construct_report_url(report_filename)
        
        # Return the standardized response with only html_url
        return AnalysisResponse(
            html_url=report_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        debug_logger.error(f"Detailed error for {symbols}:", exc_info=True)
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_api(analysis: AnalysisRequest):
    """API endpoint for analysis"""
    try:
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f'{analysis.stock_code}_vs_{analysis.benchmark_code}_{timestamp}.html'
        report_path = os.path.join(REPORTS_DIR, report_filename)

        # Convert dates
        start = datetime.strptime(analysis.start_date, '%Y-%m-%d')
        end = datetime.strptime(analysis.end_date, '%Y-%m-%d') + timedelta(days=1)

        if start >= end:
            raise HTTPException(status_code=400, detail="Start date must be before end date")

        # Download data
        stock_data = yf.download(analysis.stock_code, start=start, end=end, progress=False)
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for stock {analysis.stock_code}")

        benchmark_data = yf.download(analysis.benchmark_code, start=start, end=end, progress=False)
        if benchmark_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {analysis.benchmark_code}")

        # Process data
        stock_returns = process_stock_data(stock_data, analysis.stock_code)
        benchmark_returns = process_stock_data(benchmark_data, analysis.benchmark_code)

        # Generate report
        generate_quantstats_report(
            stock_returns,
            benchmark_returns,
            report_path,
            f'{analysis.stock_code} vs {analysis.benchmark_code} Analysis Report',
            risk_free_rate=analysis.risk_free_rate
        )

        # Return success response with report URL
        return {
            "success": True,
            "message": f"Successfully generated report for {analysis.stock_code} vs {analysis.benchmark_code}",
            "report_url": f"/static/reports/{report_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{filename:path}")
async def serve_report(filename: str):
    """Serve report files"""
    return FileResponse(f"static/reports/{filename}")

# Create MCP server and include relevant endpoints
mcp = FastApiMCP(
    app,
    name="QuantStats Analysis MCP API",
    description="MCP server for portfolio analysis endpoints. Note: Operations may take up to 3 minutes due to data fetching and analysis requirements.",
    base_url=os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000"),
    include_operations=[
        "create_portfolio_analysis"
    ],
    # Better schema descriptions
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Inject custom httpx client with 3-minute timeout
    http_client=httpx.AsyncClient(timeout=180.0)
)

# Mount the MCP server
mcp.mount()

# Log MCP operations
logger.info("Operations included in MCP server:")
for op in mcp._include_operations:
    logger.info(f"Operation '{op}' included in MCP")

logger.info("MCP server exposing portfolio analysis endpoints")
logger.info("=" * 40)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
