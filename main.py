import os
import sys
import json
import logging
import traceback
import tempfile
import uuid
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Query, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from finta import TA
import httpx
import base64
import io
import re
import markdown
from fastapi_mcp import FastApiMCP
from dotenv import load_dotenv
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("technical-analysis-api")

# Load environment variables
load_dotenv()

# Get API keys and model names from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("=" * 40)
    logger.info("FastAPI server with MCP integration is starting!")
    logger.info("MCP endpoint is available at: http://localhost:8000/mcp")
    logger.info("Using custom httpx client with 5-minute (300 second) timeout")
    
    # Log all available routes and their operation IDs
    logger.info("Available routes and operation IDs in FastAPI app:")
    fastapi_operations = []
    for route in app.routes:
        if hasattr(route, "operation_id"):
            logger.info(f"Route: {route.path}, Operation ID: {route.operation_id}")
            fastapi_operations.append(route.operation_id)
    
    # Note: we don't log MCP operations here since the MCP instance hasn't been created yet
    
    yield  # This is where the FastAPI app runs
    
    # Shutdown code
    logger.info("=" * 40)
    logger.info("FastAPI server is shutting down")
    logger.info("=" * 40)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Technical Analysis API",
    description="API for generating technical analysis reports for stocks using data from Yahoo Finance and analysis from Google's Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files directory for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

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
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )

# Add the logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://addin.xlwings.org",  # Main xlwings add-in domain
        "https://xlwings.org",        # xlwings website resources
        "null",                       # For local debugging
        "*"                           # Temporarily allow all origins for testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model
class TechnicalAnalysisRequest(BaseModel):
    """Request model for technical analysis generation."""
    ticker: str = Field(
        description="The stock symbol to analyze (e.g., 'AAPL' for Apple, 'MSFT' for Microsoft). Must be a valid Yahoo Finance ticker symbol.",
        example="AAPL"
    )
    daily_start_date: date = Field(
        description="Start date for daily price data analysis. Should be at least 6 months before daily_end_date for meaningful analysis. Format: YYYY-MM-DD",
        example="2023-07-01"
    )
    daily_end_date: date = Field(
        description="End date for daily price data analysis. Must be after daily_start_date and not in the future. Format: YYYY-MM-DD",
        example="2023-12-31"
    )
    weekly_start_date: date = Field(
        description="Start date for weekly price data analysis. Should be at least 1 year before weekly_end_date for meaningful analysis. Format: YYYY-MM-DD",
        example="2022-01-01"
    )
    weekly_end_date: date = Field(
        description="End date for weekly price data analysis. Must be after weekly_start_date and not in the future. Format: YYYY-MM-DD",
        example="2023-12-31"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "daily_start_date": "2023-07-01",
                "daily_end_date": "2023-12-31",
                "weekly_start_date": "2022-01-01",
                "weekly_end_date": "2023-12-31"
            }
        }

# Define the response model
class TechnicalAnalysisResponse(BaseModel):
    pdf_url: str
    html_url: str

# Define the endpoint with operation_id and response model
@app.post("/api/technical-analysis", operation_id="create_technical_analysis", response_model=TechnicalAnalysisResponse)
async def create_technical_analysis(
    request: Request,
    analysis_request: TechnicalAnalysisRequest = Body(
        description="Technical analysis request parameters",
        example={
            "ticker": "AAPL",
            "daily_start_date": "2023-07-01",
            "daily_end_date": "2023-12-31",
            "weekly_start_date": "2022-01-01",
            "weekly_end_date": "2023-12-31"
        }
    )
):
    """
    Generates comprehensive technical analysis reports for a specified stock ticker.
    
    This endpoint performs detailed technical analysis including:
    - Price trend analysis on daily and weekly timeframes
    - Multiple technical indicators (EMAs, MACD, RSI, Bollinger Bands)
    - Support and resistance levels
    - Volume analysis
    - Pattern recognition
    - AI-powered market interpretation
    
    The analysis is returned in two formats:
    - PDF report with detailed analysis and charts
    - Interactive HTML report for dynamic viewing
    
    The analysis covers:
    - Daily timeframe analysis (short-term trends)
    - Weekly timeframe analysis (long-term trends)
    - Technical indicator signals
    - Volume profile analysis
    - Market structure assessment
    - Potential support/resistance zones
    
    Example request:
    {
        "ticker": "AAPL",
        "daily_start_date": "2023-07-01",
        "daily_end_date": "2023-12-31",
        "weekly_start_date": "2022-01-01",
        "weekly_end_date": "2023-12-31"
    }
    """
    try:
        # If analysis_request is provided directly (normal FastAPI route)
        if analysis_request:
            logger.info(f"Using provided analysis_request: {analysis_request}")
            ticker = analysis_request.ticker
            daily_start_date = analysis_request.daily_start_date.isoformat()
            daily_end_date = analysis_request.daily_end_date.isoformat()
            weekly_start_date = analysis_request.weekly_start_date.isoformat() 
            weekly_end_date = analysis_request.weekly_end_date.isoformat()
        else:
            # Get request body - handle potential empty body
            try:
                content_type = request.headers.get("content-type", "")
                logger.info(f"Request content-type: {content_type}")
                
                body_bytes = await request.body()
                logger.info(f"Raw request body: {body_bytes}")
                
                if not body_bytes:
                    raise HTTPException(status_code=400, detail="Empty request body")
                
                if "application/json" in content_type:
                    body = json.loads(body_bytes)
                else:
                    # Try to interpret as form data
                    form = await request.form()
                    body = dict(form)
                    if not body:
                        # Last attempt to parse JSON
                        try:
                            body = json.loads(body_bytes)
                        except json.JSONDecodeError:
                            raise HTTPException(status_code=400, detail="Invalid request format")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Request body was: {await request.body()}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
            
            logger.info(f"Parsed request body: {body}")
            
            ticker = body.get("ticker")
            daily_start_date = body.get("daily_start_date")
            daily_end_date = body.get("daily_end_date")
            weekly_start_date = body.get("weekly_start_date")
            weekly_end_date = body.get("weekly_end_date")
        
        # Validate required parameters
        if not all([ticker, daily_start_date, daily_end_date, weekly_start_date, weekly_end_date]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        logger.info(f"Processing technical analysis request for {ticker}")
        logger.info(f"Daily range: {daily_start_date} to {daily_end_date}")
        logger.info(f"Weekly range: {weekly_start_date} to {weekly_end_date}")
        
        # Process daily data
        try:
            # Call Yahoo Finance API for daily data
            daily_api_url = f"https://yfin.hosting.tigzig.com/get-all-prices/?tickers={ticker}&start_date={daily_start_date}&end_date={daily_end_date}"
            daily_response = requests.get(daily_api_url)
            
            if not daily_response.ok:
                raise HTTPException(status_code=daily_response.status_code, 
                                   detail=f"Failed to fetch daily data: {daily_response.text}")
            
            daily_data = daily_response.json()
            
            if isinstance(daily_data, dict) and "error" in daily_data:
                raise HTTPException(status_code=400, detail=f"Daily data error: {daily_data['error']}")
                
            # Process daily data
            daily_rows = []
            for date, ticker_data in daily_data.items():
                if ticker in ticker_data:
                    row = ticker_data[ticker]
                    row['Date'] = date
                    daily_rows.append(row)
            
            daily_df = pd.DataFrame(daily_rows)
            daily_df.columns = [col.lower() for col in daily_df.columns]
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_df = daily_df.sort_values('date')
            
            # Calculate daily technical indicators
            daily_display_df = calculate_technical_indicators(daily_df.copy())
            
            # Create daily chart
            daily_chart_path = create_chart(daily_display_df, ticker, "Technical Analysis Charts", "Daily")
        
        except Exception as e:
            logger.error(f"Error processing daily data: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing daily data: {str(e)}")
        
        # Process weekly data
        try:
            # Call Yahoo Finance API for weekly data
            weekly_api_url = f"https://yfin.hosting.tigzig.com/get-all-prices/?tickers={ticker}&start_date={weekly_start_date}&end_date={weekly_end_date}"
            weekly_response = requests.get(weekly_api_url)
            
            if not weekly_response.ok:
                raise HTTPException(status_code=weekly_response.status_code, 
                                   detail=f"Failed to fetch weekly data: {weekly_response.text}")
            
            weekly_data = weekly_response.json()
            
            if isinstance(weekly_data, dict) and "error" in weekly_data:
                raise HTTPException(status_code=400, detail=f"Weekly data error: {weekly_data['error']}")
                
            # Process weekly data
            weekly_rows = []
            for date, ticker_data in weekly_data.items():
                if ticker in ticker_data:
                    row = ticker_data[ticker]
                    row['Date'] = date
                    weekly_rows.append(row)
            
            weekly_df = pd.DataFrame(weekly_rows)
            weekly_df['Date'] = pd.to_datetime(weekly_df['Date'])
            weekly_df = weekly_df.sort_values('Date')
            
            # Resample to weekly data
            weekly_df = weekly_df.resample('W-FRI', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            weekly_df.reset_index(inplace=True)
            
            # Calculate weekly technical indicators
            weekly_display_df = calculate_technical_indicators(weekly_df.copy())
            
            # Create weekly chart
            weekly_chart_path = create_chart(weekly_display_df, ticker, "Technical Analysis Charts", "Weekly")
        
        except Exception as e:
            logger.error(f"Error processing weekly data: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing weekly data: {str(e)}")
        
        # Create combined chart for PDF
        try:
            combined_chart_path = combine_charts(
                daily_chart_path, 
                weekly_chart_path,
                datetime.strptime(daily_start_date, "%Y-%m-%d"),
                datetime.strptime(daily_end_date, "%Y-%m-%d"),
                datetime.strptime(weekly_start_date, "%Y-%m-%d"),
                datetime.strptime(weekly_end_date, "%Y-%m-%d")
            )
        except Exception as e:
            logger.error(f"Error creating combined chart: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error creating combined chart: {str(e)}")
        
        # Upload charts to server for Gemini
        try:
            # Upload daily chart
            daily_files = {
                'file': ('daily_chart.png', open(daily_chart_path, 'rb'), 'image/png')
            }
            daily_upload_response = requests.post(
                "https://mdtopdf.hosting.tigzig.com/api/upload-image",
                files=daily_files
            )
            
            if not daily_upload_response.ok:
                raise HTTPException(status_code=daily_upload_response.status_code,
                                   detail=f"Failed to upload daily image: {daily_upload_response.text}")
            
            daily_upload_data = daily_upload_response.json()
            daily_image_path = daily_upload_data['image_path']
            
            # Upload weekly chart
            weekly_files = {
                'file': ('weekly_chart.png', open(weekly_chart_path, 'rb'), 'image/png')
            }
            weekly_upload_response = requests.post(
                "https://mdtopdf.hosting.tigzig.com/api/upload-image",
                files=weekly_files
            )
            
            if not weekly_upload_response.ok:
                raise HTTPException(status_code=weekly_upload_response.status_code,
                                   detail=f"Failed to upload weekly image: {weekly_upload_response.text}")
            
            weekly_upload_data = weekly_upload_response.json()
            weekly_image_path = weekly_upload_data['image_path']
            
            # Upload combined chart
            combined_files = {
                'file': ('combined_chart.png', open(combined_chart_path, 'rb'), 'image/png')
            }
            combined_upload_response = requests.post(
                "https://mdtopdf.hosting.tigzig.com/api/upload-image",
                files=combined_files
            )
            
            if not combined_upload_response.ok:
                raise HTTPException(status_code=combined_upload_response.status_code,
                                   detail=f"Failed to upload combined image: {combined_upload_response.text}")
            
            combined_upload_data = combined_upload_response.json()
            combined_image_path = combined_upload_data['image_path']
        
        except Exception as e:
            logger.error(f"Error uploading images: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error uploading images: {str(e)}")
        
        # Generate analysis with Gemini API
        try:
            analysis_markdown = await generate_analysis_with_gemini(
                ticker,
                daily_display_df,
                weekly_display_df,
                daily_chart_path,
                weekly_chart_path,
                combined_image_path
            )
            
            # Convert to PDF and save URL
            pdf_api_url = "https://mdtopdf.hosting.tigzig.com/text-input"
            
            pdf_response = requests.post(
                pdf_api_url,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json={"text": analysis_markdown, "image_path": combined_image_path}
            )
            
            if not pdf_response.ok:
                raise HTTPException(
                    status_code=pdf_response.status_code,
                    detail=f"Failed to convert markdown to PDF: {pdf_response.text}"
                )
            
            response_data = pdf_response.json()
            
            # Return URLs for PDF and HTML
            return TechnicalAnalysisResponse(
                pdf_url=response_data["pdf_url"],
                html_url=response_data["html_url"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Create MCP server and include all relevant endpoints AFTER defining the endpoint
mcp = FastApiMCP(
    app,
    name="Technical Analysis MCP API",
    description="MCP server for technical analysis endpoints. Note: Some operations may take up to 3 minutes due to data fetching and analysis requirements.",
    base_url=os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000"),  # Use RENDER_EXTERNAL_URL when deployed, localhost for development
    include_operations=[
        "create_technical_analysis"
    ],
    # Better schema descriptions
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Inject custom httpx client with 5-minute (300 seconds) timeout
    http_client=httpx.AsyncClient(timeout=300.0)
)

# Mount the MCP server to the FastAPI app
mcp.mount()

# Log MCP operations
logger.info("Operations included in MCP server:")
for op in mcp._include_operations:
    # We can't check against fastapi_operations here since we're outside the lifespan function
    # Just log what's included in MCP
    logger.info(f"Operation '{op}' included in MCP")

logger.info("MCP server exposing technical analysis endpoints")
logger.info("=" * 40)

# Helper functions
def calculate_technical_indicators(df):
    """Calculate technical indicators for a DataFrame."""
    logger.info("Calculating technical indicators...")
    
    # Ensure column names are lowercase for finta
    df.columns = [col.lower() for col in df.columns]
    
    # Calculate various technical indicators
    df['EMA_12'] = TA.EMA(df, 12)
    df['EMA_26'] = TA.EMA(df, 26)
    df['RSI_14'] = TA.RSI(df)
    df['ROC_14'] = TA.ROC(df, 14)
    
    # MACD
    macd = TA.MACD(df)
    if isinstance(macd, pd.DataFrame):
        df['MACD_12_26'] = macd['MACD']
        df['MACD_SIGNAL_9'] = macd['SIGNAL']
    
    # Bollinger Bands
    bb = TA.BBANDS(df)
    if isinstance(bb, pd.DataFrame):
        df['BBANDS_UPPER_20_2'] = bb['BB_UPPER']
        df['BBANDS_MIDDLE_20_2'] = bb['BB_MIDDLE']
        df['BBANDS_LOWER_20_2'] = bb['BB_LOWER']
    
    # Rename columns back to uppercase for consistency
    column_mapping = {
        'date': 'DATE',
        'open': 'OPEN',
        'high': 'HIGH',
        'low': 'LOW',
        'close': 'CLOSE',
        'volume': 'VOLUME'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    return df

def create_chart(df, ticker, title, frequency):
    """Create a chart using matplotlib and return the path to the saved image."""
    logger.info(f"Creating {frequency} chart for {ticker}...")
    
    # Create matplotlib figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                       height_ratios=[2, 1, 1], 
                                       sharex=True, 
                                       gridspec_kw={'hspace': 0})
    
    # Create a twin axis for volume
    ax1v = ax1.twinx()
    
    # Plot on the first subplot (price chart)
    ax1.plot(df['DATE'], df['CLOSE'], label='Close Price', color='black', linewidth=1.5, alpha=0.7)
    ax1.plot(df['DATE'], df['BBANDS_UPPER_20_2'], label='BB Upper', color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.plot(df['DATE'], df['BBANDS_MIDDLE_20_2'], label='BB Middle', color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.plot(df['DATE'], df['BBANDS_LOWER_20_2'], label='BB Lower', color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.plot(df['DATE'], df['EMA_12'], label='EMA-12', color='blue', linewidth=2)
    ax1.plot(df['DATE'], df['EMA_26'], label='EMA-26', color='red', linewidth=2)
    
    # Add volume bars with improved scaling
    # Calculate colors for volume bars based on price movement
    df['price_change'] = df['CLOSE'].diff()
    volume_colors = ['#26A69A' if val >= 0 else '#EF5350' for val in df['price_change']]
    
    # Calculate bar width based on date range
    bar_width = (df['DATE'].iloc[-1] - df['DATE'].iloc[0]).days / len(df) * 0.8
    if bar_width <= 0:
        bar_width = 0.8  # Default width if calculation fails
        
    # Normalize volume to make it visible
    price_range = df['CLOSE'].max() - df['CLOSE'].min()
    volume_scale_factor = price_range * 0.2 / df['VOLUME'].max() if df['VOLUME'].max() > 0 else 0.2
    normalized_volume = df['VOLUME'] * volume_scale_factor
    
    # Plot volume bars with normalized height
    ax1v.bar(df['DATE'], normalized_volume, width=bar_width, color=volume_colors, alpha=0.3)
    
    # Set volume axis properties
    ax1v.set_ylabel('Volume', fontsize=10, color='gray')
    ax1v.set_yticklabels([])
    ax1v.tick_params(axis='y', length=0)
    ax1v.set_ylim(0, price_range * 0.3)
    
    ax1.set_title(f"{ticker} - Price with EMAs and Bollinger Bands ({frequency})", fontsize=14, fontweight='bold', pad=10, loc='center')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_xticklabels([])
    
    # Plot on the second subplot (MACD)
    macd_hist = df['MACD_12_26'] - df['MACD_SIGNAL_9']
    colors = ['#26A69A' if val >= 0 else '#EF5350' for val in macd_hist]
    ax2.bar(df['DATE'], macd_hist, color=colors, alpha=0.85, label='MACD Histogram', width=bar_width)
    ax2.plot(df['DATE'], df['MACD_12_26'], label='MACD', color='#2962FF', linewidth=1.5)
    ax2.plot(df['DATE'], df['MACD_SIGNAL_9'], label='Signal', color='#FF6D00', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    ax2.set_title(f'MACD (12,26,9) - {frequency}', fontsize=12, fontweight='bold', loc='center')
    ax2.set_ylabel('MACD', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.set_xticklabels([])
    
    # Plot on the third subplot (RSI and ROC)
    ax3.plot(df['DATE'], df['RSI_14'], label='RSI (14)', color='#2962FF', linewidth=1.5)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['DATE'], df['ROC_14'], label='ROC (14)', color='#FF6D00', linewidth=1.5)
    ax3.axhline(y=70, color='#EF5350', linestyle='--', linewidth=0.8, alpha=0.3)
    ax3.axhline(y=30, color='#26A69A', linestyle='--', linewidth=0.8, alpha=0.3)
    ax3.axhline(y=50, color='gray', linestyle='-', linewidth=0.8, alpha=0.2)
    ax3_twin.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    ax3.set_ylim(0, 100)
    ax3.set_title(f'RSI & ROC - {frequency}', fontsize=12, fontweight='bold', loc='center')
    ax3.set_ylabel('RSI', fontsize=12, color='#2962FF')
    ax3_twin.set_ylabel('ROC', fontsize=12, color='#FF6D00')
    ax3.tick_params(axis='y', labelcolor='#2962FF')
    ax3_twin.tick_params(axis='y', labelcolor='#FF6D00')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.2)
    
    # Format x-axis dates
    first_date = df['DATE'].iloc[0]
    last_date = df['DATE'].iloc[-1]
    date_range = last_date - first_date
    num_ticks = min(8, len(df)) if date_range.days <= 30 else 8 if date_range.days <= 90 else 10
    tick_indices = [0] + list(range(len(df) // (num_ticks - 2), len(df) - 1, len(df) // (num_ticks - 2)))[:num_ticks-2] + [len(df) - 1]
    
    # Handle possible index out of range issues
    tick_indices = [i for i in tick_indices if 0 <= i < len(df)]
    if not tick_indices:
        tick_indices = [0] if len(df) > 0 else []
    
    ax3.set_xticks([df['DATE'].iloc[i] for i in tick_indices])
    date_format = '%Y-%m-%d' if date_range.days > 30 else '%m-%d'
    tick_labels = [df['DATE'].iloc[i].strftime(date_format) for i in tick_indices]
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure to temporary file
    temp_dir = tempfile.gettempdir()
    chart_filename = f"{ticker}_{frequency.lower()}_technical_chart.png"
    temp_path = os.path.join(temp_dir, chart_filename)
    fig.savefig(temp_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    
    return temp_path

def combine_charts(daily_path, weekly_path, daily_start, daily_end, weekly_start, weekly_end):
    """Combine daily and weekly charts into a single side-by-side image."""
    logger.info("Combining daily and weekly charts...")
    
    # Read the images
    daily_img = plt.imread(daily_path)
    weekly_img = plt.imread(weekly_path)
    
    # Format dates for display
    daily_start_str = daily_start.strftime('%d %b %Y')
    daily_end_str = daily_end.strftime('%d %b %Y')
    weekly_start_str = weekly_start.strftime('%d %b %Y')
    weekly_end_str = weekly_end.strftime('%d %b %Y')
    
    # Create a new figure with appropriate size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Display images
    ax1.imshow(daily_img)
    ax2.imshow(weekly_img)
    
    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Add titles with date ranges on single line
    ax1.set_title(f'Daily Chart ({daily_start_str} to {daily_end_str})', fontsize=14, fontweight='bold', pad=10)
    ax2.set_title(f'Weekly Chart ({weekly_start_str} to {weekly_end_str})', fontsize=14, fontweight='bold', pad=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save combined figure
    temp_dir = tempfile.gettempdir()
    combined_path = os.path.join(temp_dir, "combined_technical_chart.png")
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return combined_path

def format_data_for_analysis(df, title):
    """Format DataFrame as markdown table for analysis."""
    logger.info(f"Formatting data for analysis: {title}")
    
    # Convert DataFrame to markdown table string with clear header
    header = f"### {title} (Last 20 rows)\n"
    
    # Make sure dates are formatted nicely
    df_copy = df.copy()
    if 'DATE' in df_copy.columns:
        df_copy['DATE'] = pd.to_datetime(df_copy['DATE']).dt.strftime('%Y-%m-%d')
    
    # Create markdown table rows
    rows = []
    
    # Header row
    rows.append("| " + " | ".join(str(col) for col in df_copy.columns) + " |")
    
    # Separator row
    rows.append("| " + " | ".join(["---"] * len(df_copy.columns)) + " |")
    
    # Data rows
    for _, row in df_copy.iterrows():
        formatted_row = []
        for val in row:
            if isinstance(val, (int, float)):
                # Format numbers with 2 decimal places
                formatted_row.append(f"{val:.2f}" if isinstance(val, float) else str(val))
            else:
                formatted_row.append(str(val))
        rows.append("| " + " | ".join(formatted_row) + " |")
    
    return header + "\n".join(rows)

async def generate_analysis_with_gemini(
    ticker, 
    daily_display_df, 
    weekly_display_df, 
    daily_chart_path,
    weekly_chart_path,
    combined_image_path
):
    """Generate technical analysis report using Gemini API."""
    logger.info(f"Generating analysis with Gemini for {ticker}...")
    
    # Get the latest data points
    latest_daily = daily_display_df.iloc[-1]
    latest_weekly = weekly_display_df.iloc[-1]
    
    # Get last 20 rows for additional data
    last_20_days = daily_display_df.tail(20)
    last_20_weeks = weekly_display_df.tail(20)
    
    # Create formatted data tables for Gemini analysis
    daily_data_for_analysis = format_data_for_analysis(last_20_days, "Daily Price & Technical Data")
    weekly_data_for_analysis = format_data_for_analysis(last_20_weeks, "Weekly Price & Technical Data")
    
    # Create tables with last 5 days of data for both daily and weekly
    last_5_days = daily_display_df.tail(5)[['DATE', 'CLOSE', 'EMA_26', 'ROC_14', 'RSI_14']]
    last_5_weeks = weekly_display_df.tail(5)[['DATE', 'CLOSE', 'EMA_26', 'ROC_14', 'RSI_14']]
    
    # Create HTML table with daily and weekly data
    table_html_parts = []
    
    # Add opening wrapper div
    table_html_parts.append('<div style="display: flex; justify-content: space-between;">')
    
    # Daily table
    table_html_parts.append('<div style="width: 48%; display: inline-block;">')
    table_html_parts.append('<table style="border-collapse: collapse; width: 100%; font-size: 7pt;">')
    table_html_parts.append('<thead><tr>')
    
    # Headers
    headers = ["DAILY", "CLOSE", "EMA-26", "ROC", "RSI"]
    for header in headers:
        table_html_parts.append(f'<th style="border: 0.25pt solid #000; padding: 2pt; text-align: center;">{header}</th>')
    
    table_html_parts.append('</tr></thead><tbody>')
    
    # Add daily rows
    for _, row in last_5_days.iterrows():
        date = pd.to_datetime(row['DATE'])
        date_str = date.strftime('%d-%b')
        table_html_parts.append('<tr>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: center;">{date_str}</td>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: right;">{row["CLOSE"]:.1f}</td>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: right;">{row["EMA_26"]:.1f}</td>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: right;">{row["ROC_14"]:.1f}</td>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: right;">{int(row["RSI_14"])}</td>')
        table_html_parts.append('</tr>')
    
    table_html_parts.append('</tbody></table></div>')
    
    # Weekly table
    table_html_parts.append('<div style="width: 48%; display: inline-block;">')
    table_html_parts.append('<table style="border-collapse: collapse; width: 100%; font-size: 7pt;">')
    table_html_parts.append('<thead><tr>')
    
    # Headers
    headers = ["WEEKLY", "CLOSE", "EMA-26", "ROC", "RSI"]
    for header in headers:
        table_html_parts.append(f'<th style="border: 0.25pt solid #000; padding: 2pt; text-align: center;">{header}</th>')
    
    table_html_parts.append('</tr></thead><tbody>')
    
    # Add weekly rows
    for _, row in last_5_weeks.iterrows():
        date = pd.to_datetime(row['DATE'])
        date_str = date.strftime('%d-%b')
        table_html_parts.append('<tr>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: center;">{date_str}</td>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: right;">{row["CLOSE"]:.1f}</td>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: right;">{row["EMA_26"]:.1f}</td>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: right;">{row["ROC_14"]:.1f}</td>')
        table_html_parts.append(f'<td style="border: 0.25pt solid #000; padding: 2pt; text-align: right;">{int(row["RSI_14"])}</td>')
        table_html_parts.append('</tr>')
    
    table_html_parts.append('</tbody></table></div>')
    
    # Close the wrapper div
    table_html_parts.append('</div>')
    
    # Join all parts
    table_section = ''.join(table_html_parts)
    
    # Convert both charts to base64 for Gemini API
    with open(daily_chart_path, "rb") as daily_file:
        daily_chart_base64 = base64.b64encode(daily_file.read()).decode('utf-8')
    with open(weekly_chart_path, "rb") as weekly_file:
        weekly_chart_base64 = base64.b64encode(weekly_file.read()).decode('utf-8')
    
    # Build the prompt parts
    prompt_parts = []
    prompt_parts.append("""
    [SYSTEM INSTRUCTIONS]
    You are a professional technical analyst. Analyze the provided daily and weekly technical analysis charts and supporting data to generate a comprehensive combined analysis report. Focus on integrating insights from both timeframes to provide a complete market perspective.

    Your primary task is chart-based technical analysis. The data tables are provided as supporting information only. Make sure to analyze the full charts for the complete time period shown, not just the last 20 rows of data.

    Please structure your response in markdown format with the following EXACT structure and formatting:

    # Integrated Technical Analysis""")
    
    prompt_parts.append(f"# {ticker}")
    prompt_parts.append("""## Daily and Weekly Charts""")
    
    prompt_parts.append(f"\n![Combined Technical Analysis](charts/{combined_image_path})")
    
    # Insert the table HTML
    prompt_parts.append("\n" + table_section)
    
    # Continue with the rest of the prompt
    prompt_parts.append("""
    ### 1. Price Action and Trend Analysis
    **Daily**: [Provide detailed analysis of daily price action, trend direction, key movements]
    
    **Weekly**: [Provide detailed analysis of weekly price action, trend direction, key movements]
    
    **Confirmation/Divergence**: [Analyze how daily and weekly trends align or diverge]

    ### 2. Support and Resistance Levels
    **Daily Levels**: [List and analyze key daily support and resistance levels]
    
    **Weekly Levels**: [List and analyze key weekly support and resistance levels]
    
    **Level Alignment**: [Discuss how daily and weekly levels interact]

    ### 3. Technical Indicator Analysis
    **Daily Indicators**:
    - EMAs (12 & 26): [Analysis]
    - MACD: [Analysis]
    - RSI & ROC: [Analysis]
    - Bollinger Bands: [Analysis]

    **Weekly Indicators**:
    - EMAs (12 & 26): [Analysis]
    - MACD: [Analysis]
    - RSI & ROC: [Analysis]
    - Bollinger Bands: [Analysis]

    ### 4. Pattern Recognition
    **Daily Patterns**: [Identify and analyze patterns on daily timeframe]
    
    **Weekly Patterns**: [Identify and analyze patterns on weekly timeframe]
    
    **Pattern Alignment**: [Discuss how patterns on different timeframes confirm or contradict]

    ### 5. Volume Analysis
    **Daily Volume**: [Analyze daily volume patterns and significance]
    
    **Weekly Volume**: [Analyze weekly volume patterns and significance]
    
    **Volume Trends**: [Discuss overall volume trends and implications]

    ### 6. Technical Outlook
    [Provide integrated conclusion combining all above analysis points]
    """)
    
    # Add the technical data
    prompt_parts.append(f"""
    Current Technical Data:
    **Daily Data**:
    - Close: {latest_daily['CLOSE']} | EMA_12: {latest_daily['EMA_12']:.2f} | EMA_26: {latest_daily['EMA_26']:.2f}
    - MACD: {latest_daily['MACD_12_26']:.2f} | Signal: {latest_daily['MACD_SIGNAL_9']:.2f}
    - RSI: {latest_daily['RSI_14']:.2f} | BB Upper: {latest_daily['BBANDS_UPPER_20_2']:.2f} | BB Lower: {latest_daily['BBANDS_LOWER_20_2']:.2f}

    **Weekly Data**:
    - Close: {latest_weekly['CLOSE']} | EMA_12: {latest_weekly['EMA_12']:.2f} | EMA_26: {latest_weekly['EMA_26']:.2f}
    - MACD: {latest_weekly['MACD_12_26']:.2f} | Signal: {latest_weekly['MACD_SIGNAL_9']:.2f}
    - RSI: {latest_weekly['RSI_14']:.2f} | BB Upper: {latest_weekly['BBANDS_UPPER_20_2']:.2f} | BB Lower: {latest_weekly['BBANDS_LOWER_20_2']:.2f}
    
    Below you will find the last 20 rows of data for both daily and weekly timeframes. These are provided as supporting information for your chart analysis. Note the different date patterns to distinguish daily from weekly data:
    - Daily data: Consecutive trading days
    - Weekly data: Weekly intervals, typically Friday closing prices
    """)
    
    prompt_parts.append(daily_data_for_analysis)
    prompt_parts.append(weekly_data_for_analysis)
    
    prompt_parts.append("""
    IMPORTANT:
    1. Follow the EXACT markdown structure and formatting shown above
    2. Use bold (**) for timeframe headers as shown
    3. Maintain consistent section ordering
    4. Ensure each section has Daily, Weekly, and Confirmation/Alignment analysis
    5. Keep the analysis concise but comprehensive
    6. Focus primarily on chart analysis, using the data tables as supporting information only
    7. Analyze the complete timeframe shown in the charts, not just the last 20 rows of data
    """)
    
    # Join the prompt parts
    prompt = ''.join(prompt_parts)
    
    # Prepare API payload with both full-size images and text
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": daily_chart_base64
                    }
                },
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": weekly_chart_base64
                    }
                },
                {
                    "text": prompt
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 7500
        }
    }
    
    # Make API call to Gemini
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            response_json = response.json()
            if 'candidates' in response_json:
                analysis = response_json['candidates'][0]['content']['parts'][0]['text']
                
                # Add disclaimer
                disclaimer_note = """
                
                #### Important Disclaimer

The content in this report is for demonstration purposes only and is not investment research, investment analysis, or financial advice. This is a technical demonstration of how to use FastAPI to send data and charts to an LLM/AI via a web API call, receive a markdown-formatted response, convert it to PDF, and return downloadable links.

"""
                final_markdown = f"{disclaimer_note}{analysis}"
                
                return final_markdown
            else:
                raise HTTPException(status_code=500, detail="No analysis generated from Gemini API")
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Gemini API call failed: {response.text}"
            )
    except Exception as e:
        logger.error(f"Error in Gemini API call: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in Gemini API call: {str(e)}")

# Create route for frontend UI
@app.get("/")
async def read_root(request: Request):
    """Serve the technical analysis frontend UI"""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
