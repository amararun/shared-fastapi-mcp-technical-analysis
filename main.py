import os
import sys
import json
import logging
import traceback
import tempfile
import uuid
import time
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Set matplotlib backend BEFORE importing pyplot (thread-safe, non-interactive)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
# requests module removed - using httpx for async HTTP calls
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Query, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Send
from pydantic import BaseModel, Field
from finta import TA
import httpx
import base64
import io
import re
import markdown
from fastapi_mcp import FastApiMCP
from dotenv import load_dotenv
from tigzig_api_monitor import APIMonitorMiddleware
import urllib.parse
import glob
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
# get_remote_address removed - using custom get_real_client_ip() instead

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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-preview-09-2025")

# Allowed models for technical analysis
ALLOWED_MODELS = [
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "google/gemini-2.5-flash-preview-09-2025",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5-mini",
    "openai/gpt-4.1",
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.2",
]

# Default model (used when no model specified in request and no env var)
DEFAULT_MODEL = "google/gemini-2.5-flash-preview-09-2025"

# Get rate limit from environment variable with fallback
RATE_LIMIT = os.getenv("RATE_LIMIT", "15/hour")  # Per individual client IP limit
GLOBAL_RATE_LIMIT = os.getenv("GLOBAL_RATE_LIMIT", "100/minute")  # Global across all IPs

# Concurrency limits (this app is very heavy - AI + PDF generation)
MAX_CONCURRENT_PER_IP = int(os.getenv("MAX_CONCURRENT_PER_IP", "2"))
MAX_CONCURRENT_GLOBAL = int(os.getenv("MAX_CONCURRENT_GLOBAL", "4"))

# Log rate limit configuration at module load time (very early, before app starts)
print("=" * 60)
print("RATE LIMIT CONFIGURATION (from environment variables):")
print(f"   RATE_LIMIT: {RATE_LIMIT} {'(from env var)' if os.getenv('RATE_LIMIT') else '(DEFAULT)'} - per individual client IP")
print(f"   GLOBAL_RATE_LIMIT: {GLOBAL_RATE_LIMIT} {'(from env var)' if os.getenv('GLOBAL_RATE_LIMIT') else '(DEFAULT)'} - across all IPs")
print(f"   MAX_CONCURRENT_PER_IP: {MAX_CONCURRENT_PER_IP}, MAX_CONCURRENT_GLOBAL: {MAX_CONCURRENT_GLOBAL}")
print("=" * 60)

# Global matplotlib process pool (initialized in lifespan)
matplotlib_pool = None


def get_real_client_ip(request: Request) -> str:
    """
    Get real client IP from request headers with fallbacks.
    Priority: X-Forwarded-For (first IP) > X-Real-IP > request.client.host > 'unknown'
    For localhost requests (MCP agent), also checks for custom headers (X-Client-IP, X-MCP-Session-ID).
    Never errors - always returns a string.
    """
    try:
        # Check if this is a localhost request (likely from MCP agent)
        client_host = request.client.host if request.client else None
        is_localhost = client_host in ["127.0.0.1", "localhost", "::1"] or "localhost" in str(request.url)
        
        # For localhost requests, check for custom headers that N8N might pass
        if is_localhost:
            # Check for X-Client-IP header (custom header N8N can pass)
            x_client_ip = request.headers.get("x-client-ip")
            if x_client_ip and x_client_ip.strip() and x_client_ip.strip() != "undefined":
                ip = x_client_ip.split(",")[0].strip()
                # Remove port if present
                if ":" in ip and not ip.startswith("["):
                    ip = ip.split(":")[0]
                logger.debug(f"Using X-Client-IP header for localhost request: {ip}")
                return ip
            
            # Check for X-MCP-Session-ID header (alternative: use session ID for rate limiting)
            mcp_session_id = request.headers.get("x-mcp-session-id")
            if mcp_session_id and mcp_session_id.strip() and mcp_session_id.strip() != "undefined":
                # Use session ID as rate limit key for MCP requests
                logger.debug(f"Using X-MCP-Session-ID header for localhost request: {mcp_session_id[:8]}...")
                return f"mcp_session:{mcp_session_id}"
            
            # Check for X-Execution-ID header (N8N execution ID - always available)
            execution_id = request.headers.get("x-execution-id")
            if execution_id and execution_id.strip() and execution_id.strip() != "undefined":
                # Use execution ID as rate limit key for MCP requests
                logger.debug(f"Using X-Execution-ID header for localhost request: {execution_id[:8]}...")
                return f"n8n_exec:{execution_id}"
        
        # Check Cloudflare headers first (most reliable behind CF proxy)
        for cf_header in ("x-original-client-ip", "cf-connecting-ip"):
            val = request.headers.get(cf_header)
            if val and val.strip():
                return val.split(",")[0].strip()

        # Check X-Forwarded-For header (most common proxy header)
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            # X-Forwarded-For can contain multiple IPs: "client_ip, proxy1_ip, proxy2_ip"
            # The first IP is the original client IP
            real_ip = x_forwarded_for.split(",")[0].strip()
            if real_ip:
                return real_ip
        
        # Check X-Real-IP header (alternative proxy header)
        x_real_ip = request.headers.get("x-real-ip")
        if x_real_ip:
            return x_real_ip.strip()
        
        # Fallback to direct connection IP
        if request.client and request.client.host:
            return request.client.host
        
        # Last resort fallback
        return "unknown"
    except Exception as e:
        logger.warning(f"Error extracting client IP: {e}, using fallback")
        return "unknown"

# Initialize rate limiter with per-IP limit only
# Using a custom key function that returns the real client IP
def rate_limit_key_func(request: Request) -> str:
    """
    Custom key function for rate limiting.
    Returns a key based on the real client IP address.
    """
    real_ip = get_real_client_ip(request)
    return f"ip:{real_ip}"

# Initialize rate limiter with per-IP limit and global limit
limiter = Limiter(
    key_func=rate_limit_key_func,
    default_limits=[RATE_LIMIT],  # Per-IP limit applied to all endpoints via middleware
    application_limits=[GLOBAL_RATE_LIMIT],  # Global limit across all IPs
)
logger.info("=" * 60)
logger.info("RATE LIMITER INITIALIZED - ACTIVE LIMITS:")
logger.info(f"   Per-IP limit: {RATE_LIMIT} {'(from env var)' if os.getenv('RATE_LIMIT') else '(DEFAULT)'}")
logger.info(f"   Global limit: {GLOBAL_RATE_LIMIT} {'(from env var)' if os.getenv('GLOBAL_RATE_LIMIT') else '(DEFAULT)'}")
logger.info(f"   Concurrency: {MAX_CONCURRENT_PER_IP}/IP, {MAX_CONCURRENT_GLOBAL} global")
logger.info("=" * 60)

# --- Concurrency tracking ---
_concurrency_lock = asyncio.Lock()
_concurrency_per_ip: Dict[str, int] = {}  # ip -> active count
_concurrency_global = 0


async def check_concurrency(client_ip: str) -> None:
    """Acquire a concurrency slot or raise 429 if limits exceeded."""
    global _concurrency_global
    async with _concurrency_lock:
        ip_count = _concurrency_per_ip.get(client_ip, 0)
        if ip_count >= MAX_CONCURRENT_PER_IP:
            logger.warning(f"Concurrency limit per-IP hit: {client_ip} has {ip_count} active requests")
            raise HTTPException(
                status_code=429,
                detail=f"Too many concurrent requests. Max {MAX_CONCURRENT_PER_IP} per client. Please wait for current analysis to complete."
            )
        if _concurrency_global >= MAX_CONCURRENT_GLOBAL:
            logger.warning(f"Global concurrency limit hit: {_concurrency_global} active requests")
            raise HTTPException(
                status_code=429,
                detail=f"Server is at maximum capacity ({MAX_CONCURRENT_GLOBAL} concurrent analyses). Please try again shortly."
            )
        _concurrency_per_ip[client_ip] = ip_count + 1
        _concurrency_global += 1
        logger.info(f"Concurrency acquired: IP={client_ip} ({ip_count + 1}/{MAX_CONCURRENT_PER_IP}), global={_concurrency_global}/{MAX_CONCURRENT_GLOBAL}")


async def release_concurrency(client_ip: str) -> None:
    """Release a concurrency slot. Safe to call even if check_concurrency was not called."""
    global _concurrency_global
    async with _concurrency_lock:
        ip_count = _concurrency_per_ip.get(client_ip, 0)
        if ip_count > 0:
            _concurrency_per_ip[client_ip] = ip_count - 1
            if _concurrency_per_ip[client_ip] == 0:
                del _concurrency_per_ip[client_ip]
        if _concurrency_global > 0:
            _concurrency_global -= 1
        logger.info(f"Concurrency released: IP={client_ip}, global={_concurrency_global}/{MAX_CONCURRENT_GLOBAL}")

def cleanup_old_files():
    """
    Clean up old temporary files created by the technical analysis API.
    Deletes files older than 3 days from the temporary directory.
    """
    try:
        # Get the temporary directory
        temp_dir = tempfile.gettempdir()
        logger.info(f"Starting cleanup process in directory: {temp_dir}")
        
        # Calculate cutoff time (3 days ago)
        cutoff_time = time.time() - (3 * 24 * 60 * 60)  # 3 days in seconds
        cutoff_date = datetime.fromtimestamp(cutoff_time).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Deleting files older than: {cutoff_date}")
        
        # File patterns to clean up based on the code analysis
        file_patterns = [
            "*_technical_chart.png",      # Individual technical charts
            "combined_technical_chart.png", # Combined charts
            "*_daily_technical_chart.png", # Daily charts
            "*_weekly_technical_chart.png", # Weekly charts
            "technical_analysis_*.pdf",   # Any PDF reports
            "technical_analysis_*.html",  # Any HTML reports
            "*.tmp",                      # General temp files
        ]
        
        files_deleted = 0
        files_kept = 0
        
        for pattern in file_patterns:
            file_path_pattern = os.path.join(temp_dir, pattern)
            matching_files = glob.glob(file_path_pattern)
            
            for file_path in matching_files:
                try:
                    # Check if file exists and get its modification time
                    if os.path.exists(file_path):
                        file_mtime = os.path.getmtime(file_path)
                        file_date = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        
                        if file_mtime < cutoff_time:
                            # File is older than 3 days, delete it
                            os.remove(file_path)
                            files_deleted += 1
                            logger.info(f"Deleted old file: {os.path.basename(file_path)} (created: {file_date})")
                        else:
                            # File is recent, keep it
                            files_kept += 1
                            logger.debug(f"Kept recent file: {os.path.basename(file_path)} (created: {file_date})")
                
                except OSError as e:
                    # Handle cases where file might be in use or permission issues
                    logger.warning(f"Could not delete file {file_path}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error processing file {file_path}: {str(e)}")
        
        logger.info(f"Cleanup completed: {files_deleted} files deleted, {files_kept} recent files kept")
        
    except Exception as e:
        logger.error(f"Error during cleanup process: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise the exception to avoid breaking server startup

# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("=" * 40)
    logger.info("FastAPI server with MCP integration is starting!")
    logger.info("=" * 40)
    logger.info("🚀 VERSION 2.0 - Async HTTP + Model Selection Support")
    logger.info("   - All HTTP calls converted to async (httpx)")
    logger.info("   - Model selection parameter added (9 models supported)")
    logger.info("   - Backward compatible (model parameter optional)")
    logger.info("=" * 40)
    
    # Log rate limiting status at startup
    logger.info("=" * 60)
    logger.info("🛡️ RATE LIMITING STATUS AT STARTUP:")
    if hasattr(app.state, 'limiter'):
        limits = getattr(app.state.limiter, '_default_limits', [RATE_LIMIT])
        logger.info(f"   ✅ Per-IP limit (middleware): {RATE_LIMIT} {'(from env var)' if os.getenv('RATE_LIMIT') else '(DEFAULT)'}")
        logger.info("   ✅ Rate limiting is ACTIVE and configured correctly")
    else:
        logger.warning("   ⚠️ Rate limiting not properly configured!")
    logger.info("=" * 60)
    
    # Run cleanup process on startup
    logger.info("Running cleanup process for old temporary files...")
    cleanup_old_files()
    
    # Initialize ProcessPoolExecutor for matplotlib chart creation
    logger.info("=" * 60)
    logger.info("🎨 INITIALIZING MATPLOTLIB PROCESS POOL")
    logger.info("=" * 60)
    global matplotlib_pool
    matplotlib_pool = ProcessPoolExecutor(
        max_workers=3,  # Use 3 workers, leave 1 core for uvicorn
        mp_context=multiprocessing.get_context('spawn')  # Required for ARM64 and Windows
    )
    logger.info("✅ ProcessPoolExecutor initialized with 3 workers")
    logger.info("   - Chart creation will run in separate processes")
    logger.info("   - Bypasses Python GIL for true parallelism")
    logger.info("=" * 60)
    
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
    logger.info("Shutting down matplotlib process pool...")
    matplotlib_pool.shutdown(wait=True, cancel_futures=False)
    logger.info("Process pool shut down successfully")
    logger.info("FastAPI server is shutting down")
    logger.info("=" * 40)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Technical Analysis API",
    description="API for generating technical analysis reports for stocks using data from Yahoo Finance and analysis from Google's Gemini AI",
    version="1.0.0",
    lifespan=lifespan,
    root_path=os.environ.get("API_ROOT_PATH", ""),
    servers=[
        {"url": os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")}
    ]
)

# Configure rate limiting
app.state.limiter = limiter

# Custom rate limit exception handler with logging
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded exceptions with proper logging."""
    real_client_ip = get_real_client_ip(request)
    endpoint = request.url.path
    method = request.method
    
    logger.warning(f"🚫 RATE LIMIT EXCEEDED - Real Client IP: {real_client_ip}, Endpoint: {method} {endpoint}")
    logger.warning(f"🚫 Rate limit details: {exc.detail}")
    logger.info(f"🔄 Client should retry after: {getattr(exc, 'retry_after', 'unknown')} seconds")
    
    # Create user-friendly error message
    error_message = f"Rate limit exceeded. {exc.detail}"
    if getattr(exc, 'retry_after', None):
        retry_after = int(exc.retry_after)
        minutes = retry_after // 60
        seconds = retry_after % 60
        if minutes > 0:
            error_message += f" Please try again in {minutes} minute{'s' if minutes != 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}."
        else:
            error_message += f" Please try again in {seconds} second{'s' if seconds != 1 else ''}."
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": error_message,
            "retry_after": getattr(exc, "retry_after", None),
            "endpoint": f"{method} {endpoint}",
            "timestamp": datetime.now().isoformat()
        }
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# Global exception handler - safety net for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Safe wrapper for SlowAPIMiddleware that catches ASGI errors gracefully
class SafeRateLimitMiddleware:
    """
    Wrapper around SlowAPIMiddleware that catches ASGI protocol errors gracefully.
    This prevents server crashes when rate limiting conflicts with MCP response handling.
    """
    def __init__(self, app: ASGIApp):
        self.app = app
        # Create the actual slowapi middleware
        self.slowapi_middleware = SlowAPIMiddleware(app)
    
    async def __call__(self, scope: dict, receive: Receive, send: Send) -> None:
        # Skip non-HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Skip MCP endpoints - they don't need rate limiting and middleware conflicts with MCP
        path = scope.get("path", "")
        if path.startswith("/mcp"):
            # Bypass rate limiting middleware for MCP endpoints to avoid conflicts
            await self.app(scope, receive, send)
            return
        
        # Track response state
        response_started = False
        response_completed = False
        original_send = send
        
        async def safe_send(message: Message) -> None:
            nonlocal response_started, response_completed
            try:
                msg_type = message.get("type", "")
                
                # Prevent duplicate response.start
                if msg_type == "http.response.start":
                    if response_started:
                        logger.warning("⚠️ ASGI: Duplicate response.start detected, ignoring to prevent crash")
                        return
                    response_started = True
                
                # Track response completion
                if msg_type == "http.response.body" and not message.get("more_body", True):
                    response_completed = True
                
                await original_send(message)
            except (AssertionError, RuntimeError, ValueError) as e:
                # Catch ASGI protocol errors gracefully
                error_msg = str(e)
                if "Unexpected message" in error_msg or "response" in error_msg.lower():
                    logger.warning(f"⚠️ ASGI protocol error caught in send: {error_msg}")
                    logger.warning("⚠️ Server continuing normally despite middleware conflict")
                    # Don't crash - just log and continue
                    return
                else:
                    # Re-raise non-ASGI errors
                    raise
        
        try:
            # Try to use slowapi middleware
            await self.slowapi_middleware(scope, receive, safe_send)
        except AssertionError as e:
            # Catch ASGI AssertionErrors (the main issue we're seeing)
            error_msg = str(e)
            if "Unexpected message" in error_msg:
                logger.warning(f"⚠️ ASGI AssertionError caught gracefully: {error_msg}")
                logger.warning("⚠️ This is due to middleware conflict with MCP. Server continuing normally.")
                # If response wasn't started, let app handle it
                if not response_started:
                    try:
                        await self.app(scope, receive, safe_send)
                    except Exception as fallback_error:
                        logger.error(f"❌ Error in fallback: {fallback_error}")
                # Don't re-raise - server continues
            else:
                # Re-raise other AssertionErrors
                raise
        except (RuntimeError, ValueError) as e:
            # Catch other ASGI-related errors
            error_msg = str(e)
            if "response" in error_msg.lower() or "ASGI" in error_msg:
                logger.warning(f"⚠️ ASGI error caught gracefully: {error_msg}")
                if not response_started:
                    try:
                        await self.app(scope, receive, safe_send)
                    except Exception as fallback_error:
                        logger.error(f"❌ Error in fallback: {fallback_error}")
            else:
                # Re-raise non-ASGI errors
                raise
        except Exception as e:
            # Catch any other unexpected errors but log them
            logger.error(f"❌ Unexpected error in rate limiting middleware: {e}")
            logger.error(traceback.format_exc())
            # Try to continue if response wasn't started
            if not response_started:
                try:
                    await self.app(scope, receive, safe_send)
                except Exception as fallback_error:
                    logger.error(f"❌ Error in fallback after middleware error: {fallback_error}")
                    # Last resort: try to send error response
                    if not response_started:
                        try:
                            await safe_send({
                                "type": "http.response.start",
                                "status": 500,
                                "headers": [(b"content-type", b"application/json")],
                            })
                            await safe_send({
                                "type": "http.response.body",
                                "body": json.dumps({"error": "Internal server error"}).encode(),
                            })
                        except:
                            pass  # If we can't send error, just log it

# Configure CORS FIRST (outermost middleware - added last so it executes first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API Monitor middleware (logs to centralized tigzig logger service)
# v1.3.0: Whitelist mode - only log OUR endpoints, ignore all scanner junk
app.add_middleware(
    APIMonitorMiddleware,
    app_name="FASTAPI_TECHNICAL_ANALYSIS",
    include_prefixes=("/api/technical-analysis",),  # Specific endpoint only
)

# Add safe rate limiting middleware AFTER CORS and API Monitor
# This wrapper catches ASGI errors gracefully to prevent server crashes
app.add_middleware(SafeRateLimitMiddleware)

logger.info("🛡️ Rate limiting middleware and exception handler configured")

# Simple request logging without custom middleware to avoid ASGI conflicts
import logging
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.disabled = False

# Mount static files directory for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

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
    model: Optional[str] = Field(
        default=None,
        description="""Optional LLM model to use for analysis. 

IMPORTANT FOR AI AGENTS:
1. DEFAULT BEHAVIOR: If not provided, always use "google/gemini-2.5-flash-preview-09-2025" (Gemini 2.5 Flash). 

2. WHEN TO USE: Only include this parameter if the user EXPLICITLY requests a specific model. Otherwise, omit it entirely to use the default.

3. WHAT TO SHOW USERS: When user asks for model options, show ONLY the display names in a simple list:
   - Gemini 2.5 Flash Lite
   - Gemini 2.5 Flash (default)
   - Claude Haiku 4.5
   - GPT 4.1 Mini
   - GPT 5 Nano
   - GPT 5 Mini
   - GPT 4.1
   - GPT 5.1
   - Claude Sonnet 4.5
   - GPT-5.2
   DO NOT show the JSON mapping or API names to users.

4. DISPLAY NAMES vs API NAMES MAPPING (for internal use only - DO NOT show to users):
{
  "Gemini 2.5 Flash Lite": "google/gemini-2.5-flash-lite-preview-09-2025",
  "Gemini 2.5 Flash": "google/gemini-2.5-flash-preview-09-2025",
  "Claude Haiku 4.5": "anthropic/claude-haiku-4.5",
  "GPT 4.1 Mini": "openai/gpt-4.1-mini",
  "GPT 5 Nano": "openai/gpt-5-nano",
  "GPT 5 Mini": "openai/gpt-5-mini",
  "GPT 4.1": "openai/gpt-4.1",
  "GPT 5.1": "openai/gpt-5.1",
  "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
  "GPT-5.2": "openai/gpt-5.2"
}

5. WHEN USER SELECTS A MODEL: After user chooses a display name, look up the corresponding API name from the mapping above and pass that API name value to this parameter. Never pass display names to this API - always use the API name.

Allowed API values: google/gemini-2.5-flash-lite-preview-09-2025, google/gemini-2.5-flash-preview-09-2025, anthropic/claude-haiku-4.5, openai/gpt-4.1-mini, openai/gpt-5-nano, openai/gpt-5-mini, openai/gpt-4.1, openai/gpt-5.1, anthropic/claude-sonnet-4.5, openai/gpt-5.2""",
        example="google/gemini-2.5-flash-preview-09-2025"
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
    # Start timing the request
    request_start_time = time.time()
    request_id = str(uuid.uuid4())[:8]  # Short unique ID for this request
    
    # Log all request headers for IP detection debugging
    logger.info(f"📋 [REQ-{request_id}] ========== REQUEST HEADERS DEBUG ==========")
    logger.info(f"📋 [REQ-{request_id}] Request URL: {request.url}")
    logger.info(f"📋 [REQ-{request_id}] Request method: {request.method}")
    logger.info(f"📋 [REQ-{request_id}] Request path: {request.url.path}")
    logger.info(f"📋 [REQ-{request_id}] All request headers:")
    for header_name, header_value in request.headers.items():
        logger.info(f"📋 [REQ-{request_id}]   {header_name}: {header_value}")
    
    # Log request scope information (might contain MCP session info)
    logger.info(f"📋 [REQ-{request_id}] Request scope info (for MCP debugging):")
    logger.info(f"📋 [REQ-{request_id}]   scope type: {request.scope.get('type', 'N/A')}")
    logger.info(f"📋 [REQ-{request_id}]   scope path: {request.scope.get('path', 'N/A')}")
    query_string = request.scope.get('query_string', b'')
    if query_string:
        logger.info(f"📋 [REQ-{request_id}]   scope query_string: {query_string.decode()}")
    
    # Check for MCP-specific information in query params or path
    query_params = dict(request.query_params)
    if query_params:
        logger.info(f"📋 [REQ-{request_id}] Query parameters (might contain MCP session info):")
        for param_name, param_value in query_params.items():
            logger.info(f"📋 [REQ-{request_id}]   {param_name}: {param_value}")
    
    # Log specific IP-related headers
    x_forwarded_for = request.headers.get("x-forwarded-for", "NOT PRESENT")
    x_real_ip = request.headers.get("x-real-ip", "NOT PRESENT")
    x_forwarded_proto = request.headers.get("x-forwarded-proto", "NOT PRESENT")
    forwarded = request.headers.get("forwarded", "NOT PRESENT")
    client_host = request.client.host if request.client else "NOT AVAILABLE"
    client_port = request.client.port if request.client else "NOT AVAILABLE"
    
    logger.info(f"📋 [REQ-{request_id}] IP-related headers:")
    logger.info(f"📋 [REQ-{request_id}]   X-Forwarded-For: {x_forwarded_for}")
    logger.info(f"📋 [REQ-{request_id}]   X-Real-IP: {x_real_ip}")
    logger.info(f"📋 [REQ-{request_id}]   X-Forwarded-Proto: {x_forwarded_proto}")
    logger.info(f"📋 [REQ-{request_id}]   Forwarded: {forwarded}")
    logger.info(f"📋 [REQ-{request_id}]   request.client.host: {client_host}")
    logger.info(f"📋 [REQ-{request_id}]   request.client.port: {client_port}")
    
    # Check if this is a localhost request (likely from MCP agent)
    is_localhost = client_host in ["127.0.0.1", "localhost", "::1"] or "localhost" in str(request.url)
    if is_localhost:
        logger.info(f"📋 [REQ-{request_id}] ⚠️ LOCALHOST REQUEST DETECTED - Likely from MCP agent")
        
        # Check for custom headers that N8N might pass
        x_client_ip = request.headers.get("x-client-ip", "NOT PRESENT")
        x_mcp_session = request.headers.get("x-mcp-session-id", "NOT PRESENT")
        x_execution_id = request.headers.get("x-execution-id", "NOT PRESENT")
        
        if x_client_ip != "NOT PRESENT" and x_client_ip.strip() and x_client_ip.strip() != "undefined":
            logger.info(f"📋 [REQ-{request_id}] ✅ Found X-Client-IP header: {x_client_ip} - Will use for rate limiting")
        elif x_mcp_session != "NOT PRESENT" and x_mcp_session.strip() and x_mcp_session.strip() != "undefined":
            logger.info(f"📋 [REQ-{request_id}] ✅ Found X-MCP-Session-ID header: {x_mcp_session[:8]}... - Will use for rate limiting")
        elif x_execution_id != "NOT PRESENT" and x_execution_id.strip() and x_execution_id.strip() != "undefined":
            logger.info(f"📋 [REQ-{request_id}] ✅ Found X-Execution-ID header: {x_execution_id[:8]}... - Will use for rate limiting")
        else:
            logger.info(f"📋 [REQ-{request_id}] ⚠️ No custom headers found - Will share rate limit with all localhost requests")
            logger.info(f"📋 [REQ-{request_id}] 💡 SOLUTION: Configure N8N to pass X-Execution-ID header (simplest option)")
    
    # Get real client IP for rate limiting
    real_client_ip = get_real_client_ip(request)
    
    logger.info(f"📋 [REQ-{request_id}] IP Detection:")
    logger.info(f"📋 [REQ-{request_id}]   Real Client IP (for rate limiting): {real_client_ip}")
    logger.info(f"📋 [REQ-{request_id}] ==========================================")
    
    try:
        # Get current rate limit state (this is a quick check)
        rate_limiter = app.state.limiter
        # Try to get the current state without consuming a request
        logger.info(f"🛡️ [REQ-{request_id}] Rate limit check - Real Client IP: {real_client_ip}")
    except Exception as e:
        logger.debug(f"Rate limit check failed: {e}")
    
    logger.info(f"[REQ-{request_id}] Starting technical analysis request at {datetime.now().strftime('%H:%M:%S')}")

    # Enforce concurrency limits before doing any heavy work
    await check_concurrency(real_client_ip)
    try:
        # If analysis_request is provided directly (normal FastAPI route)
        if analysis_request:
            logger.info(f"📥 [REQ-{request_id}] Received analysis_request: ticker={analysis_request.ticker}, model={analysis_request.model}")
            ticker = analysis_request.ticker
            daily_start_date = analysis_request.daily_start_date.isoformat()
            daily_end_date = analysis_request.daily_end_date.isoformat()
            weekly_start_date = analysis_request.weekly_start_date.isoformat() 
            weekly_end_date = analysis_request.weekly_end_date.isoformat()
            requested_model = analysis_request.model
            logger.info(f"🔍 [REQ-{request_id}] Extracted requested_model: {requested_model}")
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
                raise HTTPException(status_code=400, detail="Invalid JSON format")
            
            logger.info(f"Parsed request body: {body}")
            logger.info(f"📥 [REQ-{request_id}] Request body model parameter: {body.get('model', 'NOT PROVIDED')}")
            
            ticker = body.get("ticker")
            daily_start_date = body.get("daily_start_date")
            daily_end_date = body.get("daily_end_date")
            weekly_start_date = body.get("weekly_start_date")
            weekly_end_date = body.get("weekly_end_date")
            requested_model = body.get("model")
            logger.info(f"🔍 [REQ-{request_id}] Extracted requested_model from body: {requested_model}")
        
        # Validate required parameters
        if not all([ticker, daily_start_date, daily_end_date, weekly_start_date, weekly_end_date]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        # Model selection logic with priority: request parameter > env var > default
        logger.info(f"🔍 [REQ-{request_id}] Model selection check - requested_model: {requested_model}, OPENROUTER_MODEL env: {OPENROUTER_MODEL}")
        if requested_model:
            # Validate requested model
            if requested_model not in ALLOWED_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model '{requested_model}'. Allowed models: {', '.join(ALLOWED_MODELS)}"
                )
            selected_model = requested_model
            logger.info(f"✅ [REQ-{request_id}] Using REQUESTED model: {selected_model}")
        elif OPENROUTER_MODEL:
            selected_model = OPENROUTER_MODEL
            logger.info(f"✅ [REQ-{request_id}] Using ENVIRONMENT VARIABLE model: {selected_model}")
        else:
            selected_model = DEFAULT_MODEL
            logger.info(f"✅ [REQ-{request_id}] Using DEFAULT model: {selected_model}")
        
        logger.info(f"🎯 [REQ-{request_id}] FINAL SELECTED MODEL: {selected_model}")
        
        logger.info(f"📊 [REQ-{request_id}] Processing technical analysis for ticker: {ticker}")
        logger.info(f"📅 [REQ-{request_id}] Daily range: {daily_start_date} to {daily_end_date}")
        logger.info(f"📅 [REQ-{request_id}] Weekly range: {weekly_start_date} to {weekly_end_date}")
        
        # Process daily data
        try:
            logger.info(f"📡 [REQ-{request_id}] Fetching daily data from Yahoo Finance API...")
            # Call Yahoo Finance API for daily data
            daily_api_url = f"https://yfin-h.tigzig.com/get-all-prices/?tickers={ticker}&start_date={daily_start_date}&end_date={daily_end_date}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                daily_response = await client.get(daily_api_url)
            
            if not daily_response.is_success:
                logger.error(f"Daily data fetch failed: {daily_response.status_code} {daily_response.text[:200]}")
                raise HTTPException(status_code=502, detail="Failed to fetch daily data from upstream service")

            daily_data = daily_response.json()

            if isinstance(daily_data, dict) and "error" in daily_data:
                logger.error(f"Daily data error: {daily_data['error']}")
                raise HTTPException(status_code=400, detail="Error in daily data. Check ticker symbol and date range.")
                
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
            
            logger.info(f"⚙️ [REQ-{request_id}] Processing daily data: {len(daily_df)} rows retrieved")
            # Calculate daily technical indicators
            daily_display_df = calculate_technical_indicators(daily_df.copy())
            
            # Create daily chart (async - runs in process pool)
            logger.info(f"📈 [REQ-{request_id}] Creating daily chart (async)...")
            daily_chart_path = await create_chart_async(daily_display_df, ticker, "Technical Analysis Charts", "Daily", request_id)
            logger.info(f"📈 [REQ-{request_id}] Daily chart created: {os.path.basename(daily_chart_path)}")
        
        except Exception as e:
            logger.error(f"Error processing daily data: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Error processing daily data. Please try again.")
        
        # Process weekly data
        try:
            logger.info(f"📡 [REQ-{request_id}] Fetching weekly data from Yahoo Finance API...")
            # Call Yahoo Finance API for weekly data
            weekly_api_url = f"https://yfin-h.tigzig.com/get-all-prices/?tickers={ticker}&start_date={weekly_start_date}&end_date={weekly_end_date}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                weekly_response = await client.get(weekly_api_url)
            
            if not weekly_response.is_success:
                logger.error(f"Weekly data fetch failed: {weekly_response.status_code} {weekly_response.text[:200]}")
                raise HTTPException(status_code=502, detail="Failed to fetch weekly data from upstream service")

            weekly_data = weekly_response.json()

            if isinstance(weekly_data, dict) and "error" in weekly_data:
                logger.error(f"Weekly data error: {weekly_data['error']}")
                raise HTTPException(status_code=400, detail="Error in weekly data. Check ticker symbol and date range.")
                
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
            
            logger.info(f"⚙️ [REQ-{request_id}] Processing weekly data: {len(weekly_df)} rows retrieved after resampling")
            # Calculate weekly technical indicators
            weekly_display_df = calculate_technical_indicators(weekly_df.copy())
            
            # Create weekly chart (async - runs in process pool)
            logger.info(f"📈 [REQ-{request_id}] Creating weekly chart (async)...")
            weekly_chart_path = await create_chart_async(weekly_display_df, ticker, "Technical Analysis Charts", "Weekly", request_id)
            logger.info(f"📈 [REQ-{request_id}] Weekly chart created: {os.path.basename(weekly_chart_path)}")
        
        except Exception as e:
            logger.error(f"Error processing weekly data: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Error processing weekly data. Please try again.")
        
        # Create combined chart for PDF using actual data dates
        try:
            logger.info(f"🔗 [REQ-{request_id}] Creating combined chart for PDF (async)...")
            combined_chart_path = await combine_charts_async(
                daily_chart_path, 
                weekly_chart_path,
                daily_display_df['DATE'].iloc[0],
                daily_display_df['DATE'].iloc[-1],
                weekly_display_df['DATE'].iloc[0],
                weekly_display_df['DATE'].iloc[-1],
                request_id=request_id
            )
            logger.info(f"🔗 [REQ-{request_id}] Combined chart created: {os.path.basename(combined_chart_path)}")
        except Exception as e:
            logger.error(f"Error creating combined chart: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Error creating chart. Please try again.")
        
        # Upload charts to server for Gemini
        try:
            logger.info(f"📤 [REQ-{request_id}] Uploading charts to PDF service...")
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Upload daily chart
                with open(daily_chart_path, 'rb') as daily_file:
                    files = {'file': ('daily_chart.png', daily_file, 'image/png')}
                    daily_upload_response = await client.post(
                        "https://mdtopdf.tigzig.com/api/upload-image",
                        files=files
                    )
                
                if not daily_upload_response.is_success:
                    logger.error(f"Daily image upload failed: {daily_upload_response.status_code} {daily_upload_response.text[:200]}")
                    raise HTTPException(status_code=502, detail="Failed to upload daily chart image")
                
                daily_upload_data = daily_upload_response.json()
                daily_image_path = daily_upload_data['image_path']
                logger.info(f"📤 [REQ-{request_id}] Daily chart uploaded successfully: {daily_image_path}")
                
                # Upload weekly chart
                with open(weekly_chart_path, 'rb') as weekly_file:
                    files = {'file': ('weekly_chart.png', weekly_file, 'image/png')}
                    weekly_upload_response = await client.post(
                        "https://mdtopdf.tigzig.com/api/upload-image",
                        files=files
                    )
                
                if not weekly_upload_response.is_success:
                    logger.error(f"Weekly image upload failed: {weekly_upload_response.status_code} {weekly_upload_response.text[:200]}")
                    raise HTTPException(status_code=502, detail="Failed to upload weekly chart image")
                
                weekly_upload_data = weekly_upload_response.json()
                weekly_image_path = weekly_upload_data['image_path']
                logger.info(f"📤 [REQ-{request_id}] Weekly chart uploaded successfully: {weekly_image_path}")
                
                # Upload combined chart
                with open(combined_chart_path, 'rb') as combined_file:
                    files = {'file': ('combined_chart.png', combined_file, 'image/png')}
                    combined_upload_response = await client.post(
                        "https://mdtopdf.tigzig.com/api/upload-image",
                        files=files
                    )
                
                if not combined_upload_response.is_success:
                    logger.error(f"Combined image upload failed: {combined_upload_response.status_code} {combined_upload_response.text[:200]}")
                    raise HTTPException(status_code=502, detail="Failed to upload combined chart image")
                
                combined_upload_data = combined_upload_response.json()
                combined_image_path = combined_upload_data['image_path']
                logger.info(f"📤 [REQ-{request_id}] Combined chart uploaded successfully: {combined_image_path}")
        
        except Exception as e:
            logger.error(f"Error uploading images: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Error uploading images. Please try again.")
        
        # Generate analysis with Gemini API
        try:
            logger.info(f"🤖 [REQ-{request_id}] Starting AI analysis generation with model: {selected_model}...")
            analysis_markdown = await generate_analysis_with_gemini(
                ticker,
                daily_display_df,
                weekly_display_df,
                daily_chart_path,
                weekly_chart_path,
                combined_image_path,
                selected_model
            )
            logger.info(f"🤖 [REQ-{request_id}] AI analysis generated successfully ({len(analysis_markdown)} characters)")
            
            # Convert to PDF and save URL
            pdf_api_url = "https://mdtopdf.tigzig.com/text-input"
            
            logger.info(f"📄 [REQ-{request_id}] Sending to PDF conversion service...")
            logger.info(f"📄 [REQ-{request_id}] Analysis length: {len(analysis_markdown)} chars, Combined image: {combined_image_path}")
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                pdf_response = await client.post(
                    pdf_api_url,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    json={"text": analysis_markdown, "image_path": combined_image_path}
                )
            
            if not pdf_response.is_success:
                logger.error(f"PDF conversion failed: {pdf_response.status_code} {pdf_response.text[:200]}")
                raise HTTPException(status_code=502, detail="Failed to convert analysis to PDF")
            
            response_data = pdf_response.json()
            logger.info(f"📄 [REQ-{request_id}] PDF service response: {pdf_response.status_code}")
            logger.info(f"📄 [REQ-{request_id}] PDF URL: {response_data.get('pdf_url', 'NOT FOUND')}")
            logger.info(f"📄 [REQ-{request_id}] HTML URL: {response_data.get('html_url', 'NOT FOUND')}")
            
            # Calculate total processing time
            total_time = time.time() - request_start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            
            # Create response object
            final_response = TechnicalAnalysisResponse(
                pdf_url=response_data["pdf_url"],
                html_url=response_data["html_url"]
            )
            
            # Log final response being sent to frontend
            logger.info(f"✅ [REQ-{request_id}] Request completed successfully in {time_str}")
            real_client_ip = get_real_client_ip(request)
            logger.info(f"🛡️ [REQ-{request_id}] Rate limit OK - Request processed successfully for Real Client IP: {real_client_ip}")
            logger.info(f"✅ [REQ-{request_id}] Final response to frontend:")
            logger.info(f"   📄 PDF URL: {final_response.pdf_url}")
            logger.info(f"   🌐 HTML URL: {final_response.html_url}")
            
            return final_response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Error generating analysis. Please try again.")
        
        finally:
            # Cleanup temporary files
            try:
                temp_files_to_cleanup = [daily_chart_path, weekly_chart_path, combined_chart_path]
                for temp_file in temp_files_to_cleanup:
                    if temp_file and os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary files: {str(cleanup_error)}")
            
    except HTTPException as he:
        # Calculate time even for HTTP exceptions
        if 'request_start_time' in locals():
            total_time = time.time() - request_start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            logger.error(f"❌ [REQ-{request_id if 'request_id' in locals() else 'UNKNOWN'}] Request failed after {time_str}: {he.detail}")
        raise
    except Exception as e:
        # Calculate time for unexpected errors
        if 'request_start_time' in locals():
            total_time = time.time() - request_start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            logger.error(f"❌ [REQ-{request_id if 'request_id' in locals() else 'UNKNOWN'}] Unexpected error after {time_str}: {str(e)}")
        else:
            logger.error(f"❌ Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
    finally:
        # Always release concurrency slot, even on error or cancellation
        await asyncio.shield(release_concurrency(real_client_ip))

# Create MCP server and include all relevant endpoints AFTER defining the endpoint
# NOTE: Rate limiting middleware has been configured BEFORE this MCP initialization
# to ensure all MCP-exposed endpoints are properly rate-limited
logger.info("🔧 Initializing MCP server (rate limiting already configured)")
mcp = FastApiMCP(
    app,
    name="Technical Analysis MCP API",
    description="MCP server for technical analysis endpoints. Note: Some operations may take up to 3 minutes due to data fetching and analysis requirements.",
    include_operations=[
        "create_technical_analysis"
    ],
    # Better schema descriptions
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Use localhost base_url to avoid Cloudflare overwriting cf-connecting-ip
    http_client=httpx.AsyncClient(
        timeout=300.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        base_url="http://localhost:8000"
    )
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
logger.info(f"🛡️ Rate limiting ACTIVE: {RATE_LIMIT} per IP")
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

def create_chart(df, ticker, title, frequency, request_id=None):
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
    
    # Calculate evenly spaced indices for exactly 10 ticks (including first and last)
    num_intervals = 9  # This will give us 10 ticks total
    if len(df) > 1:
        step = (len(df) - 1) / num_intervals
        tick_indices = [0]  # Always include first index
        for i in range(1, num_intervals):
            tick_indices.append(int(i * step))
        tick_indices.append(len(df) - 1)  # Always include last index
    else:
        tick_indices = [0] if len(df) > 0 else []
    
    ax3.set_xticks([df['DATE'].iloc[i] for i in tick_indices])
    
    # Format dates as "dd-mmm-'yy"
    tick_labels = [df['DATE'].iloc[i].strftime("%d-%b-'%y") for i in tick_indices]
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure to temporary file
    temp_dir = tempfile.gettempdir()
    # Make filename unique per request to avoid race conditions with parallel requests
    unique_suffix = f"_{request_id}" if request_id else f"_{int(time.time() * 1000)}"
    chart_filename = f"{ticker}_{frequency.lower()}_technical_chart{unique_suffix}.png"
    temp_path = os.path.join(temp_dir, chart_filename)
    fig.savefig(temp_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    
    return temp_path

def combine_charts(daily_path, weekly_path, daily_start, daily_end, weekly_start, weekly_end, request_id=None):
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
    
    # Display images (weekly first, then daily)
    ax1.imshow(weekly_img)
    ax2.imshow(daily_img)
    
    # Remove axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Add titles with date ranges on single line
    ax1.set_title(f'Weekly Chart ({weekly_start_str} to {weekly_end_str})', fontsize=14, fontweight='bold', pad=10)
    ax2.set_title(f'Daily Chart ({daily_start_str} to {daily_end_str})', fontsize=14, fontweight='bold', pad=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save combined figure
    temp_dir = tempfile.gettempdir()
    # Make filename unique per request to avoid race conditions with parallel requests
    unique_suffix = f"_{request_id}" if request_id else f"_{int(time.time() * 1000)}"
    combined_path = os.path.join(temp_dir, f"combined_technical_chart{unique_suffix}.png")
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return combined_path

# Async wrappers for chart creation using ProcessPoolExecutor
async def create_chart_async(df, ticker, title, frequency, request_id=None):
    """
    Async wrapper for create_chart that runs in a separate process.
    Bypasses Python GIL for true parallelism.
    """
    loop = asyncio.get_event_loop()
    # Convert DataFrame to dict for pickling (processes can't share pandas objects directly)
    df_dict = {
        'DATE': df['DATE'].tolist(),
        'OPEN': df['OPEN'].tolist(),
        'HIGH': df['HIGH'].tolist(),
        'LOW': df['LOW'].tolist(),
        'CLOSE': df['CLOSE'].tolist(),
        'VOLUME': df['VOLUME'].tolist(),
        'EMA_12': df['EMA_12'].tolist(),
        'EMA_26': df['EMA_26'].tolist(),
        'RSI_14': df['RSI_14'].tolist(),
        'ROC_14': df['ROC_14'].tolist(),
        'MACD_12_26': df['MACD_12_26'].tolist(),
        'MACD_SIGNAL_9': df['MACD_SIGNAL_9'].tolist(),
        'BBANDS_UPPER_20_2': df['BBANDS_UPPER_20_2'].tolist(),
        'BBANDS_MIDDLE_20_2': df['BBANDS_MIDDLE_20_2'].tolist(),
        'BBANDS_LOWER_20_2': df['BBANDS_LOWER_20_2'].tolist(),
    }
    
    # Run in process pool
    return await loop.run_in_executor(
        matplotlib_pool,
        _create_chart_worker,
        df_dict, ticker, title, frequency, request_id
    )

def _create_chart_worker(df_dict, ticker, title, frequency, request_id):
    """
    Worker function that runs in subprocess.
    Reconstructs DataFrame and calls create_chart.
    """
    # Reconstruct DataFrame from dict
    df = pd.DataFrame(df_dict)
    # Convert DATE back to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    # Call original create_chart function
    return create_chart(df, ticker, title, frequency, request_id)

async def combine_charts_async(daily_path, weekly_path, daily_start, daily_end, weekly_start, weekly_end, request_id=None):
    """
    Async wrapper for combine_charts that runs in a separate process.
    Bypasses Python GIL for true parallelism.
    """
    loop = asyncio.get_event_loop()
    # Run in process pool
    return await loop.run_in_executor(
        matplotlib_pool,
        combine_charts,
        daily_path, weekly_path, daily_start, daily_end, weekly_start, weekly_end, request_id
    )


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
    combined_image_path,
    model: str
):
    """Generate technical analysis report using OpenRouter API with specified model."""
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
    Your analysis is for professional use, so prioritize clarity, precision, and actionable insights. You will receive two types of data:

    1. REPORT STRUCTURE DATA: Pre-formatted HTML tables showing the last 5 rows of data
       - These tables are part of the final report structure
       - They MUST be preserved exactly as provided
       - They appear right after the chart image

    2. REFERENCE DATA: Additional 20 rows of data in markdown format
       - This data is PROVIDED ONLY FOR YOUR ANALYSIS
       - DO NOT include this data in the final report
       - Use it to inform your analysis in sections 1-6

    **CRITICAL: REQUIRED REPORT STRUCTURE**
    The final report must follow this exact structure - no additions or modifications:

        # Integrated Technical Analysis
        ## [TICKER_SYMBOL]
        ## Daily and Weekly Charts
        ![Combined Technical Analysis](charts/[CHART_FILENAME])
        [PRESERVE EXISTING HTML TABLES HERE - DO NOT MODIFY]
        ### 1. Price Action and Trend Analysis
        **Daily:** [your analysis]
        **Weekly:** [your analysis]
        **Confirmation/Divergence:** [your analysis]
        [CONTINUE WITH SECTIONS 2-6 AS SPECIFIED]

    **MANDATORY FORMATTING RULES**
    1. Keep the report structure exactly as shown above
    2. DO NOT add any new sections or data tables
    3. DO NOT modify or remove existing HTML tables
    4. Use markdown only for your analysis in sections 1-6
    5. The 20-row reference data tables MUST NOT appear in the final report
    6. Keep exactly one blank line between sections

    **ANALYSIS REQUIREMENTS**
    - Use the 20-row reference data to inform your analysis
    - Write your analysis ONLY in sections 1-6
    - Keep analysis concise and actionable
    - Focus on technical insights and patterns

    **WORD COUNT LIMITS**
    - Follow the word count limits specified in each section
    - Focus on actionable insights
    - No generic statements

    Remember: The 20-row data tables are for your reference ONLY. They should NOT appear in the final report structure.
    """)
    
    prompt_parts.append(f"# {ticker}")
    prompt_parts.append("""## Daily and Weekly Charts""")
    
    prompt_parts.append(f"\n![Combined Technical Analysis](charts/{combined_image_path})")
    
    # Insert the table HTML
    prompt_parts.append("\n" + table_section)
    
    # Continue with the rest of the prompt
    prompt_parts.append("""
    ### 1. Price Action and Trend Analysis
    Apply your comprehensive knowledge; **the examples provided below are illustrative, not exhaustive.** Word count limit: 200-250 words

    **Daily:** [Analyze the daily trend's character (e.g., strong, maturing, range-bound). Identify the current phase (e.g., impulse wave, corrective pullback, consolidation). Describe the recent sequence of highs and lows.]

    **Weekly:** [Analyze the primary trend on the weekly chart. e.g Is it well-established? Is it showing signs of acceleration or deceleration? etc. Place the recent daily action into the context of this longer-term trend.]

    **Confirmation/Divergence:** [Synthesize the timeframes. e.g Is the daily action a simple pause (consolidation) in a strong weekly uptrend? Or are there early warnings on the daily chart (e.g., a lower high) that challenge the weekly trend? etc ]



    ### 2. Support and Resistance Levels
    Apply your comprehensive knowledge; **the examples provided below are illustrative, not exhaustive.** Word count limit: 200-250 words

    **Daily Levels:**
    - **Horizontal S/R:** [Identify key horizontal support (e.g previous swing lows, consolidation bottoms) and resistance (e.g previous swing highs, consolidation tops) levels. Be specific with price zones.]
    - **Dynamic S/R:** [Analyze the EMAs and Bollinger Bands as areas of potential dynamic support or resistance.]

    **Weekly Levels:**
    - **Horizontal S/R:** [Identify the most significant, long-term horizontal support and resistance zones from the weekly chart.]
    - **Dynamic S/R:** [Analyze the weekly EMAs and Bollinger Bands as major trend-following support levels.]

    **Level Alignment:** [Discuss the interaction of levels. e.g Is a key daily resistance level just below a major weekly resistance? Does a daily support level coincide with the weekly etc] 


    ### 3. Technical Indicator Analysis
    Apply your comprehensive knowledge; **the examples provided below are illustrative, not exhaustive.** Word count limit: 200-250 words

    **Daily Indicators:**
    - **EMAs (12 & 26):** [Analyze the crossover status, the spread between the EMAs (indicating momentum), and their role as dynamic support/resistance etc]
    - **MACD:** [Analyze the MACD line vs. the signal line, its position relative to the zero line, and the momentum trajectory shown by the histogram. Look for divergences with price.]
    - **RSI & ROC:** [Analyze RSI levels (overbought/oversold context, support/resistance flips ) , ROC for momentum speed. Crucially, identify any **bullish or bearish divergences** against recent price highs/lows.]
    - **Bollinger Bands:** [Analyze the price's position relative to the bands. e.g Is it "walking the band" (strong trend)? Note the width of the bands – are they contracting (Bollinger Squeeze, indicating potential for a volatile move) or expanding?]

    **Weekly Indicators:**
    - **EMAs (12 & 26):** [Same this as Daily but now with Weekly Chart]
    - **MACD:** [Same this as Daily but now with Weekly Chart]
    - **RSI & ROC:** [Same this as Daily but now with Weekly Chart]


    ### 4. Pattern Recognition
    Apply your comprehensive knowledge; **the examples provided below are illustrative, not exhaustive.** Word count limit: 200-250 words

    **Daily Patterns:** [Identify classic chart patterns (e.g., triangles, double tops, rising tops and bottoms, flags, pennants, channels, wedges etc)]

    **Weekly Patterns:** [Identify larger, multi-month patterns on the weekly chart. Note the overall market structure.]

    **Pattern Alignment:** [How do the daily and weekly patter align? e.g Does the daily pattern (e.g., a bull flag) fit within the context of the larger weekly uptrend? This alignment provides a higher-probability trade setup.]

    ### 5. Volume Analysis
    Apply your comprehensive knowledge; **the examples provided below are illustrative, not exhaustive.** Word count limit: 200-250 words

    **Daily Volume:** [Analyze the volume trend. Correlate volume with price action. Is volume increasing on up-days and decreasing on down-days (bullish confirmation)? Note any high-volume spikes and sharp drpos and what they signify (e.g., capitulation, breakout etc).]

    **Weekly Volume:** [Analyze the weekly volume bars in context. Does volume confirm the primary trend? e.g Is there a significant drop-off in volume that suggests waning conviction?]

    **Volume Trends:** [Summarize the volume picture. Is participation generally increasing or decreasing, and what does this imply for the sustainability of the current trend?]


    ### 6. Technical Outlook
    Apply your comprehensive knowledge; **the examples provided below are illustrative, not exhaustive.** This is the most important section. Synthesize all the above points into a coherent thesis. Word count limit:250-300 words

    **Primary Scenario (Base Case):** [Based on the weight of the evidence, describe the most likely path for the price in the short-to-medium term and why you think so. Mention specific price targets for this scenario. Don't just share the scenaior and price targets, essential to share the reasoning behind the scenario synthesizing the detailed analysis of the charts and data tables. .]

    - **Confirmation:** [e.g what specific price action (e.g., "a decisive daily close above the [X] resistance on high volume") would confirm the primary bullish hypthesis?]

    - **Invalidation:** [e.g What specific price action (e.g., "a break below the [Y] support and the 26-day EMA") would invalidate the primary bullish thesis and suggest a deeper correction towards the next support at [Z]?]
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
    
    # Prepare API payload with both full-size images and text for OpenRouter
    logger.info(f"📤 [REQ] Preparing OpenRouter API call with model: {model}")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{daily_chart_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{weekly_chart_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 10000
    }
    
    # Make API call to OpenRouter
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    logger.info(f"🤖 Calling OpenRouter API for {ticker} analysis...")
    logger.info(f"📤 [MODEL DEBUG] Payload model field: {payload.get('model', 'NOT FOUND')}")
    logger.info(f"📤 [MODEL DEBUG] Full payload model: {payload['model']}")
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
        
        logger.info(f"📥 [MODEL DEBUG] OpenRouter response status: {response.status_code}")
        if response.status_code == 200:
            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                analysis = response_json['choices'][0]['message']['content']
                logger.info(f"🤖 OpenRouter API response successful ({len(analysis)} chars generated)")
                
                # Add disclaimer
                disclaimer_note = """
                
                #### Important Disclaimer


This report is generated using AI based on live technical data and is provided for informational purposes only. It is not investment advice, investment analysis, or formal research. While care has been taken to ensure accuracy, outputs should be verified and are intended to support- not replace sound human judgment.

"""
                final_markdown = f"{disclaimer_note}{analysis}"
                
                return final_markdown
            else:
                raise HTTPException(status_code=500, detail="AI analysis generation failed. Please try again.")
        else:
            logger.error(f"OpenRouter API call failed with status {response.status_code}: {response.text}")
            raise HTTPException(
                status_code=502,
                detail="AI analysis service temporarily unavailable. Please try again."
            )
    except Exception as e:
        logger.error(f"Error in OpenRouter API call: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="AI analysis service error. Please try again.")

# Create route for frontend UI
@app.get("/")
async def read_root(request: Request):
    """Serve the technical analysis frontend UI"""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
