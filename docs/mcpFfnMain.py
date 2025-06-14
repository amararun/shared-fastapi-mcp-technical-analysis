from fastapi import FastAPI, Request, Query, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import ffn
import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, date, timedelta
import logging
import warnings
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout, asynccontextmanager
import requests
from fastapi_mcp import FastApiMCP
import traceback
import httpx
import uuid
import time
from starlette.middleware.base import BaseHTTPMiddleware
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns

# Load environment variables
load_dotenv()

# Get environment variables for URL handling with multiple fallback checks
IS_LOCAL_DEVELOPMENT_RAW = os.getenv('IS_LOCAL_DEVELOPMENT', '0')
# Check for various truthy values
IS_LOCAL_DEVELOPMENT = IS_LOCAL_DEVELOPMENT_RAW.lower() in ['1', 'true', 'yes', 'on']
BASE_URL_FOR_REPORTS = os.getenv('BASE_URL_FOR_REPORTS')

# If no .env file exists, default to local development for localhost
if not IS_LOCAL_DEVELOPMENT and not BASE_URL_FOR_REPORTS:
    logger.warning("No .env file found and no environment variables set. Defaulting to local development mode.")
    IS_LOCAL_DEVELOPMENT = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Debug: Log environment variable values
logger.info(f"Environment Variables:")
logger.info(f"IS_LOCAL_DEVELOPMENT (raw): '{IS_LOCAL_DEVELOPMENT_RAW}'")
logger.info(f"IS_LOCAL_DEVELOPMENT (processed): {IS_LOCAL_DEVELOPMENT}")
logger.info(f"BASE_URL_FOR_REPORTS: '{BASE_URL_FOR_REPORTS}'")

# Suppress warnings
warnings.filterwarnings('ignore')

# Patch numpy for older FFN versions
np.Inf = np.inf

# Simple file cleanup function for startup
def cleanup_old_reports(max_age_hours: int = 72) -> dict:
    """
    Clean up old report files from the reports directory on server startup.
    
    Args:
        max_age_hours: Maximum age of files in hours before deletion (default: 72 hours = 3 days)
        
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        logger.info(f"Starting file cleanup (max age: {max_age_hours} hours)")
        stats = {
            "total_removed": 0,
            "html_removed": 0,
            "csv_removed": 0,
            "png_removed": 0,
            "errors": 0
        }
        
        # Calculate cutoff time
        now = datetime.now()
        cutoff_time = now - timedelta(hours=max_age_hours)
        logger.info(f"Cutoff time for file cleanup: {cutoff_time}")
        
        # Check if reports directory exists
        if not os.path.exists(REPORTS_DIR):
            logger.info(f"Reports directory does not exist: {REPORTS_DIR}")
            return stats
        
        # Process all files in the reports directory
        for filename in os.listdir(REPORTS_DIR):
            file_path = os.path.join(REPORTS_DIR, filename)
            
            # Skip directories, only process files
            if os.path.isdir(file_path):
                continue
            
            try:
                # Get file modification time
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Check if file is older than cutoff
                if file_time < cutoff_time:
                    # Determine file type and update stats
                    if filename.endswith('.html'):
                        stats["html_removed"] += 1
                    elif filename.endswith('.csv'):
                        stats["csv_removed"] += 1
                    elif filename.endswith('.png'):
                        stats["png_removed"] += 1
                    
                    # Remove the file
                    os.remove(file_path)
                    logger.info(f"Removed old file: {filename}")
                    stats["total_removed"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                stats["errors"] += 1
        
        logger.info(f"File cleanup complete. Stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error during file cleanup: {str(e)}")
        return {"error": str(e), "total_removed": 0}

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("=" * 40)
    logger.info("FastAPI server with MCP integration is starting!")
    logger.info("MCP endpoint is available at: /mcp")
    logger.info("Using custom httpx client with 3-minute (180 second) timeout")
    
    # Run file cleanup on startup (keep files from last 3 days)
    logger.info("Running startup file cleanup...")
    cleanup_stats = cleanup_old_reports(max_age_hours=72)  # 3 days
    logger.info(f"Startup cleanup completed: {cleanup_stats}")
    
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

# Create a middleware for request logging
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log the request
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Request [{request_id}]: {request.method} {request.url.path} from {client_host}")
        
        # Try to log query parameters if any
        if request.query_params:
            logger.info(f"Request [{request_id}] params: {dict(request.query_params)}")
        
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log the response
            logger.info(f"Response [{request_id}]: {response.status_code} (took {process_time:.4f}s)")
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request [{request_id}] failed after {process_time:.4f}s: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Initialize FastAPI app - minimal configuration
app = FastAPI(
    title="FFN Portfolio Analysis API",
    description="API for generating portfolio analysis reports using FFN",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://addin.xlwings.org",  # Main xlwings add-in domain
        "https://xlwings.org",        # xlwings website resources
        "null",                       # For local debugging
        "*"                           # Allow all origins for MCP compatibility
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add the logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Simple request logging without custom middleware to avoid ASGI conflicts
import logging
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.disabled = False

# Mount static files
static_dir = os.path.join(os.getcwd(), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Ensure reports directory exists
REPORTS_DIR = os.path.join('static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Define the request model
class PortfolioAnalysisRequest(BaseModel):
    """Request model for portfolio analysis generation."""
    symbols: str = Field(
        description="Stock symbols to analyze, comma-separated (e.g., 'AAPL,GOOG,MSFT'). Must be valid Yahoo Finance ticker symbols. Supports multiple symbols for portfolio analysis.",
        example="AAPL,MSFT,GOOG"
    )
    start_date: date = Field(
        description="Start date for analysis. Should be at least 6 months before end_date for meaningful analysis. Format: YYYY-MM-DD",
        example="2023-01-01"
    )
    end_date: date = Field(
        description="End date for analysis. Must be after start_date and not in the future. Format: YYYY-MM-DD",
        example="2023-12-31"
    )
    risk_free_rate: float = Field(
        default=0.0,
        description="Risk-free rate as annual percentage (e.g., 5.0 for 5%). Used in Sharpe ratio and other risk-adjusted return calculations. Default is 0.0% if not provided.",
        example=5.0,
        ge=0.0,
        le=100.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbols": "AAPL,MSFT,GOOG",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "risk_free_rate": 5.0
            }
        }

# Define the response model
class PortfolioAnalysisResponse(BaseModel):
    """Response model for portfolio analysis."""
    html_report_ffn_url: str = Field(
        description="URL to access the HTML report with portfolio analysis and visualizations"
    )
    input_price_data_csv_url: str = Field(
        description="URL to download the raw price data CSV file"
    )
    cumulative_returns_csv_url: str = Field(
        description="URL to download the cumulative returns data CSV file"
    )
    success: str = Field(
        description="Success message indicating report generation status"
    )

def get_stock_data(symbols, start_date, end_date):
    """Fetch stock data for the given symbols."""
    try:
        # Convert symbols string to list
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Create empty DataFrame to store all data
        all_data = pd.DataFrame()
        
        # Debug: Print the symbols we're processing
        logger.info(f"Processing symbols: {symbol_list}")
        
        # Configure yfinance session with headers
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        
        # Fetch data for each symbol with retries
        for symbol in symbol_list:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Get data using yfinance with custom session
                    ticker = yf.Ticker(symbol)
                    ticker.session = session
                    df = ticker.history(start=start_date, end=end_date, interval='1d')
                    
                    if df.empty:
                        logger.error(f"No data available for {symbol}")
                        retry_count += 1
                        if retry_count == max_retries:
                            continue
                        continue
                        
                    # Use Close price and ensure data is properly formatted
                    prices = df['Close'].copy()
                    
                    # Convert to UTC first, then remove timezone info completely
                    if prices.index.tz is not None:
                        prices.index = prices.index.tz_convert('UTC').tz_localize(None)
                    
                    logger.info(f"\nDate info for {symbol}:")
                    logger.info(f"Index type: {type(prices.index)}")
                    logger.info(f"Sample date: {prices.index[0]}")
                    logger.info(f"Has timezone: {prices.index.tz is not None}")
                    
                    prices = prices.sort_index()  # Ensure data is sorted by date
                    
                    # Debug: Print first few dates for this symbol
                    logger.info(f"\nFirst 5 dates for {symbol}:")
                    logger.info(prices.head())
                    logger.info(f"\nLast 5 dates for {symbol}:")
                    logger.info(prices.tail())
                    logger.info(f"Total dates for {symbol}: {len(prices)}")
                    
                    # Remove any 0 or negative prices
                    prices = prices[prices > 0]
                    
                    # Remove any duplicated indices
                    prices = prices[~prices.index.duplicated(keep='first')]
                    
                    # Create DataFrame with symbol as column name
                    symbol_df = pd.DataFrame(prices)
                    symbol_df.columns = [symbol]  # Use the symbol as column name
                    
                    # Add to the main DataFrame
                    if all_data.empty:
                        all_data = symbol_df
                    else:
                        # Use merge instead of direct assignment to ensure proper date alignment
                        all_data = pd.merge(all_data, symbol_df,
                                          left_index=True, right_index=True,
                                          how='outer')
                    
                    # Debug: Print DataFrame info after adding this symbol
                    logger.info(f"\nDataFrame after adding {symbol}:")
                    logger.info(f"Shape: {all_data.shape}")
                    logger.info(f"Columns: {all_data.columns.tolist()}")
                    logger.info(f"Sample dates:")
                    logger.info(all_data.head())
                    
                    # If we got here, break the retry loop
                    break
                    
                except Exception as e:
                    logger.error(f"Failed to get ticker {symbol} reason: {str(e)}")
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Max retries reached for {symbol}")
                    continue
        
        if all_data.empty:
            raise ValueError(f"No data available for the provided symbols in the date range {start_date} to {end_date}")
        
        # Debug: Print DataFrame state before NaN handling
        logger.info("\nBefore NaN handling:")
        logger.info(f"Shape: {all_data.shape}")
        logger.info(f"NaN counts per column:")
        logger.info(all_data.isna().sum())
        logger.info("\nSample of data with NaNs:")
        logger.info(all_data.head())
        
        # Forward fill any missing values (up to 5 days)
        all_data = all_data.fillna(method='ffill', limit=5)
        
        # Debug: Print DataFrame state after forward fill
        logger.info("\nAfter forward fill:")
        logger.info(f"Shape: {all_data.shape}")
        logger.info(f"NaN counts per column:")
        logger.info(all_data.isna().sum())
        
        # Drop any remaining NaN values
        all_data = all_data.dropna()
        
        # Debug: Print final DataFrame state
        logger.info("\nFinal DataFrame state:")
        logger.info(f"Shape: {all_data.shape}")
        logger.info(f"First 5 rows:")
        logger.info(all_data.head())
        logger.info(f"Last 5 rows:")
        logger.info(all_data.tail())
        
        # Ensure we have enough data points
        if len(all_data) < 20:  # Minimum requirement for meaningful statistics
            logger.error(f"Not enough data points. Only found {len(all_data)} aligned points.")
            raise ValueError("Not enough data points for meaningful analysis")
            
        return all_data
        
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        raise ValueError(f"Error fetching stock data: {str(e)}")

def generate_cumulative_returns_chart(data, report_filename_base):
    """Generate a cumulative returns chart and save it as PNG."""
    try:
        # Set up the style for a professional look
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure and axis - keep original size for quality, control display size in CSS
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate cumulative returns (rebase to start at 0%)
        # Convert prices to returns, then to cumulative returns
        returns_data = data.pct_change().fillna(0)
        cumulative_returns = (1 + returns_data).cumprod() - 1
        
        # Plot each series
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, column in enumerate(cumulative_returns.columns):
            color = colors[i % len(colors)]
            line = ax.plot(cumulative_returns.index, cumulative_returns[column] * 100, 
                   label=column, linewidth=2, color=color)
        
        # Formatting - remove external title, will add inside chart
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add title inside chart area, centered at the top
        ax.text(0.5, 0.95, 'Cumulative Returns', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='center', va='top')
        
        # Tight layout
        plt.tight_layout()
        
        # Save the chart with high DPI for crisp quality
        chart_filename = f"{report_filename_base}_cumulative_returns.png"
        chart_path = os.path.join(REPORTS_DIR, chart_filename)
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()  # Important: close the figure to free memory
        
        return chart_filename
        
    except Exception as e:
        logger.error(f"Error generating cumulative returns chart: {str(e)}")
        plt.close()  # Ensure figure is closed even on error
        return None

def generate_perf_report(data, risk_free_rate=0.0):
    """Generate a performance report using ffn's GroupStats."""
    try:
        # Ensure the data is properly sorted
        data = data.sort_index()
        
        # Calculate performance statistics using calc_stats() first
        perf = data.calc_stats()
        
        # Then set the risk-free rate on the GroupStats object
        # Convert annual risk-free rate percentage to decimal
        rf_decimal = risk_free_rate / 100.0
        perf.set_riskfree_rate(rf_decimal)
        
        # Generate a timestamp for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a filename for the report
        symbols_text = '-'.join(data.columns).replace(' ', '_')
        filename = f"report_{symbols_text}_{timestamp}.html"
        report_filename_base = f"report_{symbols_text}_{timestamp}"
        
        # Generate the cumulative returns chart
        chart_filename = generate_cumulative_returns_chart(data, report_filename_base)
        chart_url = None
        if chart_filename:
            if IS_LOCAL_DEVELOPMENT:
                chart_url = f"/static/reports/{chart_filename}"
            else:
                base_url = BASE_URL_FOR_REPORTS.rstrip('/') + '/' if BASE_URL_FOR_REPORTS else ""
                chart_url = f"{base_url}static/reports/{chart_filename}"
        
        # Generate the CAGR bar chart
        cagr_chart_filename = generate_cagr_bar_chart(data, report_filename_base)
        cagr_chart_url = None
        if cagr_chart_filename:
            if IS_LOCAL_DEVELOPMENT:
                cagr_chart_url = f"/static/reports/{cagr_chart_filename}"
            else:
                base_url = BASE_URL_FOR_REPORTS.rstrip('/') + '/' if BASE_URL_FOR_REPORTS else ""
                cagr_chart_url = f"{base_url}static/reports/{cagr_chart_filename}"
        
        # Generate CSV exports
        csv_files = generate_csv_exports(data, perf, report_filename_base)
        csv_urls = {}
        for csv_type, csv_filename in csv_files.items():
            if IS_LOCAL_DEVELOPMENT:
                csv_urls[csv_type] = f"/static/reports/{csv_filename}"
            else:
                base_url = BASE_URL_FOR_REPORTS.rstrip('/') + '/' if BASE_URL_FOR_REPORTS else ""
                csv_urls[csv_type] = f"{base_url}static/reports/{csv_filename}"
        
        # Full path for the report
        report_path = os.path.join(REPORTS_DIR, filename)
        
        # Generate HTML report with tabular performance data
        with open(report_path, 'w') as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html lang="en">\n')
            f.write('<head>\n')
            f.write('    <meta charset="UTF-8">\n')
            f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
            f.write('    <title>Security Performance Report - FFN</title>\n')
            f.write('    <link href="https://cdn.tailwindcss.com" rel="stylesheet">\n')
            f.write('    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">\n')
            f.write('    <style>\n')
            f.write('        * {\n')
            f.write('            font-weight: normal !important;\n')  # Override all bold fonts
            f.write('        }\n')
            f.write('        .report-table {\n')
            f.write('            background-image: linear-gradient(#e5e7eb 1px, transparent 1px),\n')
            f.write('                              linear-gradient(90deg, #e5e7eb 1px, transparent 1px);\n')
            f.write('            background-size: 100% 1.5em, 25% 100%;\n')
            f.write('            background-color: white;\n')
            f.write('            font-size: 1.1em;\n')  # Increased from 1.0em for better readability
            f.write('            color: #374151;\n')  # Darker gray (gray-700) - darker than previous gray-600
            f.write('            font-weight: normal;\n')
            f.write('        }\n')
            f.write('        .main-header {\n')
            f.write('            color: #1e40af;\n')  # Darker blue (blue-800) instead of blue-600
            f.write('            font-weight: 800 !important;\n')  # Extra bold (800 instead of 700)
            f.write('        }\n')
            f.write('        .section-header {\n')
            f.write('            color: #3b82f6;\n')  # Darker blue (blue-500) instead of lighter blue
            f.write('            font-weight: 700 !important;\n')  # Bold (700 instead of 600)
            f.write('        }\n')
            f.write('        body {\n')
            f.write('            color: #374151;\n')  # Darker gray (gray-700) for better readability
            f.write('            font-weight: normal;\n')
            f.write('            margin: 0 !important;\n')
            f.write('            padding-top: 0 !important;\n')
            f.write('        }\n')
            f.write('        h1, h2, h3, h4, h5, h6 {\n')
            f.write('            font-weight: normal !important;\n')
            f.write('        }\n')
            f.write('        h1.main-header, h2.section-header, h3.section-header {\n')
            f.write('            font-weight: 700 !important;\n')  # Make headers even bolder
            f.write('        }\n')
            f.write('        strong {\n')
            f.write('            font-weight: normal !important;\n')
            f.write('        }\n')
            f.write('        .chart-container {\n')
            f.write('            position: sticky;\n')
            f.write('            top: 20px;\n')
            f.write('        }\n')
            f.write('        .chart-img {\n')
            f.write('            width: 60%;\n')  # Scale down to 60% for even better viewport fit
            f.write('            height: auto;\n')  # Maintain aspect ratio
            f.write('            border-radius: 8px;\n')
            f.write('            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);\n')
            f.write('            max-width: 100%;\n')  # Ensure responsiveness
            f.write('        }\n')
            f.write('    </style>\n')
            f.write('</head>\n')
            f.write('<body class="bg-gray-50">\n')
            
            # Add REX AI Header
            f.write('    <header style="\n')
            f.write('        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);\n')
            f.write('        color: white;\n')
            f.write('        padding: 8px 0;\n')
            f.write('        margin: 0;\n')
            f.write('        font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;\n')
            f.write('        box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n')
            f.write('        margin-bottom: 20px;\n')
            f.write('    ">\n')
            f.write('        <div style="max-width: 1200px; margin: 0 auto; padding: 0 24px; display: flex; justify-content: space-between; align-items: center;">\n')
            f.write('            <div style="display: flex; align-items: center; gap: 12px; font-size: 18px; font-weight: 900;">\n')
            f.write('                <!-- Icons first -->\n')
            f.write('                <div style="display: flex; align-items: center; gap: 3px; font-size: 20px;">\n')
            f.write('                    <span style="font-family: \'Times New Roman\', serif; font-style: italic; font-weight: 700;">f(x)</span>\n')
            f.write('                    <span style="font-family: monospace; font-weight: 700; letter-spacing: -1px;">&lt;/&gt;</span>\n')
            f.write('                </div>\n')
            f.write('                <!-- REX text with full hyperlink -->\n')
            f.write('                <a href="https://rex.tigzig.com" target="_blank" rel="noopener noreferrer" style="color: #fbbf24; text-decoration: none; font-weight: 900;">\n')
            f.write('                    REX <span style="color: #ffffff; font-weight: 900;">: AI Co-Analyst</span>\n')
            f.write('                </a>\n')
            f.write('                <span style="color: #ffffff; font-weight: 700;">- Portfolio Analytics</span>\n')
            f.write('            </div>\n')
            f.write('        </div>\n')
            f.write('    </header>\n')
            
            f.write('    <div style="max-width: 1200px; margin: 0 auto; padding: 0 24px 32px 24px;">\n')
            f.write('        <div class="mb-6">\n')
            f.write('            <h1 class="text-3xl mb-3 main-header">Security Performance Report - FFN</h1>\n')
            
            # Add simple line note below header (as requested) - made larger and closer
            f.write('            <p style="font-size: 16px; color: #6b7280; margin-bottom: 8px; line-height: 1.4;">\n')
            f.write('                All metrics in this report are generated using the open-source \n')
            f.write('                <a href="https://github.com/pmorissette/ffn" target="_blank" style="color: #1d4ed8; text-decoration: none; font-weight: 500;">FFN library</a>.\n')
            f.write('            </p>\n')
            f.write('        </div>\n')
            
            # Data Summary section - moved to top before chart
            f.write('        <div class="mb-12">\n')
            f.write('            <h2 class="text-2xl mb-4 section-header">Data Summary</h2>\n')
            f.write('            <div class="bg-white rounded-lg shadow overflow-x-auto">\n')
            f.write('                <pre class="p-4 font-mono text-sm report-table">\n')
            f.write(f'Start Date: {data.index[0].strftime("%Y-%m-%d")}\n')
            f.write(f'End Date: {data.index[-1].strftime("%Y-%m-%d")}\n')
            f.write(f'Trading Days: {len(data)}\n')
            f.write(f'Risk-Free Rate: {risk_free_rate:.2f}% (annual)\n')
            # Round first prices to 2 decimal places
            first_prices = {k: round(v, 2) for k, v in data.iloc[0].to_dict().items()}
            last_prices = {k: round(v, 2) for k, v in data.iloc[-1].to_dict().items()}
            f.write(f'First Prices: {first_prices}\n')
            f.write(f'Last Prices: {last_prices}\n')
            f.write('                </pre>\n')
            f.write('            </div>\n')
            f.write('        </div>\n')
            
            # Create two-column layout
            f.write('        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">\n')
            
            # Left column: Charts (1/3 width on large screens)
            f.write('            <div class="lg:col-span-1">\n')
            f.write('                <div class="chart-container">\n')
            
            # Cumulative Returns Chart
            if chart_url:
                f.write('                    <div class="bg-white rounded-lg shadow p-4 mb-6">\n')
                f.write(f'                        <img src="{chart_url}" alt="Cumulative Returns Chart" class="chart-img">\n')
                f.write('                    </div>\n')
            
            # CAGR Bar Chart
            if cagr_chart_url:
                f.write('                    <div class="bg-white rounded-lg shadow p-4 mb-6">\n')
                f.write(f'                        <img src="{cagr_chart_url}" alt="CAGR Bar Chart" class="chart-img">\n')
                f.write('                    </div>\n')
            
            # If no charts available
            if not chart_url and not cagr_chart_url:
                f.write('                    <div class="bg-white rounded-lg shadow p-4 mb-6">\n')
                f.write('                        <p class="text-gray-500">Chart generation failed</p>\n')
                f.write('                    </div>\n')
            
            f.write('                </div>\n')
            f.write('            </div>\n')
            
            # Right column: Tables (2/3 width on large screens)
            f.write('            <div class="lg:col-span-2">\n')
            
            # Performance Statistics section (moved Data Summary out of this column)
            f.write('                <div class="mb-12">\n')
            f.write('                    <h2 class="text-2xl mb-4 section-header">Performance Statistics</h2>\n')
            f.write('                    <div class="bg-white rounded-lg shadow overflow-x-auto">\n')
            f.write('                        <pre class="p-4 font-mono text-sm report-table">\n')
            
            stats_output = io.StringIO()
            with redirect_stdout(stats_output):
                perf.display()
            f.write(stats_output.getvalue())
            
            f.write('                        </pre>\n')
            f.write('                    </div>\n')
            f.write('                </div>\n')
            
            # Drawdown Analysis section
            f.write('                <div class="mb-12">\n')
            f.write('                    <h2 class="text-2xl mb-4 section-header">Drawdown Analysis</h2>\n')
            
            for column in data.columns:
                try:
                    # Calculate drawdown series
                    drawdown_series = ffn.to_drawdown_series(data[column])
                    logger.info(f"\nDrawdown series for {column}:")
                    logger.info(drawdown_series.head())
                    
                    # Get detailed drawdown info
                    drawdown_details = ffn.drawdown_details(drawdown_series)
                    logger.info(f"\nDrawdown details for {column}:")
                    logger.info(drawdown_details)
                    
                    if drawdown_details is not None and len(drawdown_details) > 0:
                        # Sort by drawdown magnitude (ascending order since drawdowns are negative)
                        drawdown_details = drawdown_details.sort_values('drawdown')
                        
                        f.write(f'                    <h3 class="text-xl mb-3 mt-6 section-header">{column}</h3>\n')
                        f.write('                    <div class="bg-white rounded-lg shadow overflow-x-auto mb-6">\n')
                        f.write('                        <pre class="p-4 font-mono text-sm report-table">\n')
                        
                        # Write header
                        f.write('Start Date      End Date        Duration (Days)    Drawdown %\n')
                        
                        # Show top 10 drawdowns
                        for _, row in drawdown_details.head(10).iterrows():
                            start_date = row["Start"].strftime("%Y-%m-%d")
                            end_date = row["End"].strftime("%Y-%m-%d")
                            duration = str(row["Length"]).rjust(8)  # Right align with more space
                            drawdown = f"{row['drawdown']*100:.2f}%".rjust(12)  # Right align with more space
                            
                            f.write(f'{start_date}    {end_date}    {duration}    {drawdown}\n')
                        
                        f.write('                        </pre>\n')
                        f.write('                    </div>\n')
                    else:
                        logger.warning(f"No valid drawdown details for {column}")
                        f.write(f'                    <h3 class="text-xl mb-3 mt-6 section-header">{column}</h3>\n')
                        f.write('                    <div class="bg-white rounded-lg shadow overflow-x-auto mb-6">\n')
                        f.write('                        <pre class="p-4 font-mono text-sm report-table">No significant drawdowns found.</pre>\n')
                        f.write('                    </div>\n')
                
                except Exception as e:
                    logger.error(f"Error processing drawdowns for {column}: {str(e)}")
                    f.write(f'                    <h3 class="text-xl mb-3 mt-6 section-header">{column}</h3>\n')
                    f.write('                    <div class="bg-white rounded-lg shadow overflow-x-auto mb-6">\n')
                    f.write('                        <pre class="p-4 font-mono text-sm report-table">Error calculating drawdowns.</pre>\n')
                    f.write('                    </div>\n')
                
                f.write('                </div>\n')
            
            # Monthly Returns section
            for symbol in data.columns:
                f.write('                <div class="mb-12">\n')
                f.write(f'                    <h2 class="text-2xl mb-4 section-header">Monthly Returns - {symbol}</h2>\n')
                f.write('                    <div class="bg-white rounded-lg shadow overflow-x-auto">\n')
                f.write('                        <pre class="p-4 font-mono text-sm report-table">\n')
                
                monthly_output = io.StringIO()
                with redirect_stdout(monthly_output):
                    perf[symbol].display_monthly_returns()
                f.write(monthly_output.getvalue())
                
                f.write('                        </pre>\n')
                f.write('                    </div>\n')
                f.write('                </div>\n')
            
            # Close right column
            f.write('            </div>\n')
            
            # Close grid layout
            f.write('        </div>\n')
            
            # Add FFN disclaimer box at bottom (moved from top as requested) - with proper margins
            f.write('        <div style="\n')
            f.write('            background: rgba(255,248,220,0.8);\n')
            f.write('            border: 1px solid #fbbf24;\n')
            f.write('            border-radius: 6px;\n')
            f.write('            padding: 12px;\n')
            f.write('            margin: 24px 0 16px 0;\n')
            f.write('            font-size: 13px;\n')
            f.write('            color: #92400e;\n')
            f.write('            font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;\n')
            f.write('            line-height: 1.5;\n')
            f.write('        ">\n')
            f.write('            <strong>Note:</strong> All metrics in this report are generated using the open-source \n')
            f.write('            <a href="https://github.com/pmorissette/ffn" target="_blank" style="color: #1d4ed8; text-decoration: none; font-weight: 500;">FFN library</a>. \n')
            f.write('            While widely used for performance and risk analytics, some calculations may rely on \n')
            f.write('            assumptions (e.g., trading days, compounding methods) or be subject to version-specific behavior. \n')
            f.write('            Key metrics such as total return and CAGR have been manually reviewed for consistency, but users \n')
            f.write('            are encouraged to refer to the \n')
            f.write('            <a href="https://github.com/pmorissette/ffn" target="_blank" style="color: #1d4ed8; text-decoration: none; font-weight: 500;">official documentation</a> \n')
            f.write('            for full methodology.\n')
            f.write('        </div>\n')
            
            # Custom single-line footer with email added
            f.write('        <footer style="\n')
            f.write('            background: rgba(255,255,255,0.5);\n')
            f.write('            border-top: 1px solid #e0e7ff;\n')
            f.write('            padding: 8px 0;\n')
            f.write('            margin-top: 20px;\n')
            f.write('            font-size: 12px;\n')
            f.write('            color: #1e1b4b;\n')
            f.write('            font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;\n')
            f.write('            font-weight: normal;\n')
            f.write('        ">\n')
            f.write('            <div style="max-width: 1200px; margin: 0 auto; padding: 0 24px;">\n')
            f.write('                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 4px;">\n')
            f.write('                    <div style="font-size: 12px; color: rgba(30, 27, 75, 0.7); font-weight: normal;">\n')
            f.write('                        Amar Harolikar <span style="margin: 0 6px; color: #c7d2fe;">•</span>\n')
            f.write('                        Specialist - Decision Sciences & Applied Generative AI <span style="margin: 0 6px; color: #c7d2fe;">•</span>\n')
            f.write('                        <a href="mailto:amar@harolikar.com" style="color: #4338ca; text-decoration: none; font-weight: normal;">amar@harolikar.com</a>\n')
            f.write('                    </div>\n')
            f.write('                    <div style="display: flex; align-items: center; gap: 16px; font-size: 12px;">\n')
            f.write('                        <a href="https://www.linkedin.com/in/amarharolikar" target="_blank" rel="noopener noreferrer"\n')
            f.write('                           style="color: #4338ca; text-decoration: none; font-weight: normal;">\n')
            f.write('                            LinkedIn\n')
            f.write('                        </a>\n')
            f.write('                        <a href="https://github.com/amararun" target="_blank" rel="noopener noreferrer"\n')
            f.write('                           style="color: #4338ca; text-decoration: none; font-weight: normal;">\n')
            f.write('                            GitHub\n')
            f.write('                        </a>\n')
            f.write('                        <a href="https://rex.tigzig.com" target="_blank" rel="noopener noreferrer"\n')
            f.write('                           style="color: #4338ca; text-decoration: none; font-weight: normal;">\n')
            f.write('                            rex.tigzig.com\n')
            f.write('                        </a>\n')
            f.write('                        <a href="https://tigzig.com" target="_blank" rel="noopener noreferrer"\n')
            f.write('                           style="color: #4338ca; text-decoration: none; font-weight: normal;">\n')
            f.write('                            tigzig.com\n')
            f.write('                        </a>\n')
            f.write('                    </div>\n')
            f.write('                </div>\n')
            f.write('            </div>\n')
            f.write('        </footer>\n')
            
            f.write('    </div>\n')
            f.write('</body>\n')
            f.write('</html>\n')
        
        html_url = construct_report_url(filename)
        return html_url, csv_urls
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise ValueError(f"Error generating report: {str(e)}")

def construct_report_url(filename: str) -> str:
    """Construct the appropriate URL for report files based on environment."""
    logger.info(f"construct_report_url called with filename: {filename}")
    logger.info(f"IS_LOCAL_DEVELOPMENT: {IS_LOCAL_DEVELOPMENT}")
    logger.info(f"BASE_URL_FOR_REPORTS: '{BASE_URL_FOR_REPORTS}'")
    
    if IS_LOCAL_DEVELOPMENT:
        url = f"/static/reports/{filename}"
        logger.info(f"Using local development URL: {url}")
        return url
    else:
        # For remote deployment, BASE_URL_FOR_REPORTS must be provided
        if not BASE_URL_FOR_REPORTS:
            logger.error("BASE_URL_FOR_REPORTS environment variable is required for remote deployment but not provided")
            raise ValueError(
                "Configuration Error: BASE_URL_FOR_REPORTS environment variable is required for remote deployment. "
                "Please set BASE_URL_FOR_REPORTS to your deployment URL (e.g., https://yourdomain.com/)"
            )
        
        # Ensure BASE_URL_FOR_REPORTS ends with a slash
        base_url = BASE_URL_FOR_REPORTS.rstrip('/') + '/'
        url = f"{base_url}static/reports/{filename}"
        logger.info(f"Using remote deployment URL: {url}")
        return url

@app.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request,
    error: str = Query(None, description="Error message to display"),
    success: str = Query(None, description="Success message to display"),
    report_path: str = Query(None, description="Path to generated report")
):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error": error,
            "success": success,
            "report_path": report_path,
            "report_generated": bool(report_path)
        }
    )

@app.post("/api/analyze")
async def analyze_api(request: Request):
    """API endpoint for JavaScript form submission - returns JSON response."""
    try:
        # Get form data from request
        form_data = await request.form()
        symbols = form_data.get("symbols")
        start_date = form_data.get("start_date") 
        end_date = form_data.get("end_date")
        risk_free_rate = float(form_data.get("risk_free_rate", 0.0))
        
        if not all([symbols, start_date, end_date]):
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required fields"}
            )
        
        # Get stock data
        data = get_stock_data(symbols, start_date, end_date)
        
        # Generate report (this now returns html_url and csv_urls)
        report_result = generate_perf_report(data, risk_free_rate)
        
        # Extract the URLs from the result
        if isinstance(report_result, tuple):
            html_url, csv_urls = report_result
        else:
            html_url = report_result
            csv_urls = {}
        
        # Return JSON response
        return JSONResponse(
            content={
                "success": True,
                "message": "Report generated successfully!",
                "html_report_ffn_url": html_url,
                "input_price_data_csv_url": csv_urls.get('price_data', ''),
                "cumulative_returns_csv_url": csv_urls.get('cumulative_returns', '')
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error in API: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"error": "Validation error occurred"}
        )
    except Exception as e:
        logger.error(f"Unexpected error in API: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "Error generating analysis"}
        )

@app.post("/analyze", operation_id="analyze_portfolio", response_model=PortfolioAnalysisResponse)
async def analyze(
    request: Request,
    analysis_request: PortfolioAnalysisRequest = Body(
        description="Portfolio analysis request parameters",
        example={
            "symbols": "AAPL,MSFT,GOOG",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "risk_free_rate": 5.0
        }
    )
):
    """
    Generates a comprehensive portfolio analysis report using FFN.
    
    This endpoint performs detailed portfolio analysis including:
    - Performance statistics (returns, volatility, Sharpe ratio, etc.)
    - Drawdown analysis (maximum drawdown, drawdown periods)
    - Monthly returns breakdown by security
    - Lookback returns analysis
    - Risk metrics and correlation analysis
    
    Supports multiple stock symbols for portfolio analysis. The risk-free rate is used in 
    Sharpe ratio and other risk-adjusted return calculations.
    
    The analysis is returned as an HTML report with:
    - Data summary with date ranges and price information
    - Comprehensive performance metrics table (including Sharpe ratios calculated with provided risk-free rate)
    - Detailed drawdown analysis for each security
    - Monthly return tables with color-coded performance
    - Lookback return analysis for different time periods
    
    Example request:
    {
        "symbols": "AAPL,MSFT,GOOG",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "risk_free_rate": 5.0
    }
    
    Parameters:
    - symbols: Comma-separated Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG'). Supports multiple symbols.
    - start_date: Analysis start date (YYYY-MM-DD format)
    - end_date: Analysis end date (YYYY-MM-DD format)
    - risk_free_rate: Annual risk-free rate percentage (default: 0.0%). Used for Sharpe ratio calculations.
    
    Returns:
    - PortfolioAnalysisResponse with HTML report URL and success message
    
    Raises:
    - HTTPException 400: Invalid parameters or insufficient data
    - HTTPException 500: Data fetching or processing errors
    
    Note: Analysis may take up to 2-3 minutes depending on date range and number of symbols.
    """
    try:
        # Extract parameters from the request model
        symbols = analysis_request.symbols
        start_date = analysis_request.start_date.isoformat()
        end_date = analysis_request.end_date.isoformat()
        risk_free_rate = analysis_request.risk_free_rate
        
        logger.info(f"Processing portfolio analysis request for symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Risk-free rate: {risk_free_rate}%")
        
        # Get stock data
        data = get_stock_data(symbols, start_date, end_date)
        
        # Generate report (this now returns html_url and csv_urls)
        report_result = generate_perf_report(data, risk_free_rate)
        
        # Extract the URLs from the result
        if isinstance(report_result, tuple):
            html_url, csv_urls = report_result
        else:
            html_url = report_result
            csv_urls = {}
        
        # Return structured response with specific CSV URLs
        return PortfolioAnalysisResponse(
            html_report_ffn_url=html_url,
            input_price_data_csv_url=csv_urls.get('price_data', ''),
            cumulative_returns_csv_url=csv_urls.get('cumulative_returns', ''),
            success="Portfolio analysis report generated successfully!"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")

def generate_csv_exports(data, perf, report_filename_base):
    """Generate multiple CSV files with different datasets."""
    csv_files = {}
    
    try:
        # 1. Raw Price Data
        price_data = data.copy()
        price_data.index = price_data.index.strftime('%Y-%m-%d')  # Format dates
        price_filename = f"{report_filename_base}_price_data.csv"
        price_path = os.path.join(REPORTS_DIR, price_filename)
        price_data.to_csv(price_path)
        csv_files['price_data'] = price_filename
        
        # 2. Daily Returns Data
        daily_returns = data.pct_change().dropna() * 100  # Convert to percentage
        daily_returns.index = daily_returns.index.strftime('%Y-%m-%d')
        returns_filename = f"{report_filename_base}_daily_returns.csv"
        returns_path = os.path.join(REPORTS_DIR, returns_filename)
        daily_returns.to_csv(returns_path)
        csv_files['daily_returns'] = returns_filename
        
        # 3. Cumulative Returns Data
        cumulative_returns = (1 + data.pct_change().fillna(0)).cumprod() - 1
        cumulative_returns = cumulative_returns * 100  # Convert to percentage
        cumulative_returns.index = cumulative_returns.index.strftime('%Y-%m-%d')
        cumulative_filename = f"{report_filename_base}_cumulative_returns.csv"
        cumulative_path = os.path.join(REPORTS_DIR, cumulative_filename)
        cumulative_returns.to_csv(cumulative_path)
        csv_files['cumulative_returns'] = cumulative_filename
        
        # 4. Summary Statistics Data
        summary_stats = {}
        for symbol in data.columns:
            stats = perf[symbol].stats
            summary_stats[symbol] = stats
        
        summary_df = pd.DataFrame(summary_stats)
        summary_filename = f"{report_filename_base}_summary_statistics.csv"
        summary_path = os.path.join(REPORTS_DIR, summary_filename)
        summary_df.to_csv(summary_path)
        csv_files['summary_statistics'] = summary_filename
        
        # 5. Correlation Matrix (Daily Returns)
        correlation_matrix = data.pct_change().corr()
        correlation_filename = f"{report_filename_base}_correlation_matrix.csv"
        correlation_path = os.path.join(REPORTS_DIR, correlation_filename)
        correlation_matrix.to_csv(correlation_path)
        csv_files['correlation_matrix'] = correlation_filename
        
        # 6. Monthly Returns Data (if available)
        try:
            monthly_data = {}
            for symbol in data.columns:
                if hasattr(perf[symbol], 'return_table') and len(perf[symbol].return_table) > 0:
                    monthly_data[symbol] = perf[symbol].return_table
            
            if monthly_data:
                # Convert to a more readable format
                monthly_df_list = []
                for symbol, monthly_table in monthly_data.items():
                    for year, months in monthly_table.items():
                        for month, return_val in months.items():
                            if month != 13:  # Skip YTD column
                                monthly_df_list.append({
                                    'Symbol': symbol,
                                    'Year': year,
                                    'Month': month,
                                    'Return_%': return_val * 100 if not pd.isna(return_val) else np.nan
                                })
                
                if monthly_df_list:
                    monthly_df = pd.DataFrame(monthly_df_list)
                    monthly_filename = f"{report_filename_base}_monthly_returns.csv"
                    monthly_path = os.path.join(REPORTS_DIR, monthly_filename)
                    monthly_df.to_csv(monthly_path, index=False)
                    csv_files['monthly_returns'] = monthly_filename
        except Exception as e:
            logger.warning(f"Could not generate monthly returns CSV: {str(e)}")
        
        return csv_files
        
    except Exception as e:
        logger.error(f"Error generating CSV exports: {str(e)}")
        return {}

def generate_cagr_bar_chart(data, report_filename_base):
    """Generate a CAGR bar chart for all securities."""
    try:
        # Set up the style for a professional look
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure and axis - same size as cumulative returns chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate CAGR for each security
        years = (data.index[-1] - data.index[0]).days / 365.25
        cagr_data = {}
        
        for column in data.columns:
            total_return = (data[column].iloc[-1] / data[column].iloc[0]) - 1
            cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            cagr_data[column] = cagr * 100  # Convert to percentage
        
        # Create bar chart
        securities = list(cagr_data.keys())
        cagr_values = list(cagr_data.values())
        
        # Use consistent colors with cumulative returns chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        bar_colors = [colors[i % len(colors)] for i in range(len(securities))]
        
        bars = ax.bar(securities, cagr_values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Formatting
        ax.set_ylabel('CAGR (%)', fontsize=12)
        ax.set_xlabel('Securities', fontsize=12)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
        
        # Add value labels on bars with better spacing
        max_value = max(cagr_values) if cagr_values else 0
        min_value = min(cagr_values) if cagr_values else 0
        value_range = max_value - min_value
        label_offset = value_range * 0.02 if value_range > 0 else 0.5  # Dynamic offset based on data range
        
        for bar, value in zip(bars, cagr_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (label_offset if height > 0 else -label_offset),
                   f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=12, fontweight='bold')
        
        # Add title inside chart area, but with more spacing from top
        ax.text(0.5, 0.92, 'Compound Annual Growth Rate (CAGR)', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='center', va='top')
        
        # Adjust y-axis limits to provide more space at the top for title and value labels
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        # Add 15% extra space at the top for title and labels
        ax.set_ylim(current_ylim[0], current_ylim[1] + y_range * 0.15)
        
        # Tight layout with padding
        plt.tight_layout(pad=2.0)
        
        # Save the chart with high DPI for crisp quality
        chart_filename = f"{report_filename_base}_cagr_chart.png"
        chart_path = os.path.join(REPORTS_DIR, chart_filename)
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()  # Important: close the figure to free memory
        
        return chart_filename
        
    except Exception as e:
        logger.error(f"Error generating CAGR bar chart: {str(e)}")
        plt.close()  # Ensure figure is closed even on error
        return None

# Create MCP server AFTER all endpoints are defined with proper configuration
base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
if not base_url.startswith(("http://", "https://")):
    base_url = f"http://{base_url}"

mcp = FastApiMCP(
    app,
    name="FFN Portfolio Analysis MCP API",
    description="MCP server for portfolio analysis endpoints. Note: Operations may take up to 3 minutes due to data fetching and analysis requirements.",
    include_operations=[
        "analyze_portfolio"
    ],
    # Better schema descriptions
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Inject custom httpx client with proper configuration
    http_client=httpx.AsyncClient(
        timeout=180.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        base_url=base_url
    )
)

# Mount the MCP server to the FastAPI app
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