from fastapi import FastAPI
from typing import List
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from fastapi import FastAPI, Request, Response, Query
import traceback
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
# Import FastApiMCP for MCP server integration
from fastapi_mcp import FastApiMCP
import logging
import uuid
import sys
from starlette.middleware.base import BaseHTTPMiddleware
import json
import httpx  # Import httpx for custom client configuration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("yfinance-api")

app = FastAPI()

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

# Add CORS middleware configuration
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

@app.middleware("http")
async def strip_yfin_prefix(request: Request, call_next):
    if request.url.path.startswith("/yfin"):
        request.scope["path"] = request.url.path[5:]  # Remove '/yfin' prefix
    response = await call_next(request)
    return response

# Define function to get annual balance sheet for multiple tickers
def fetch_balance_sheet_as_text(tickers: List[str]):
    try:
        # Initialize an empty string to store the concatenated text
        all_text = ""

        # Process each ticker symbol
        for ticker in tickers:
            # Remove any leading or trailing spaces from the ticker symbol
            ticker = ticker.strip()
            logger.info(f"Fetching balance sheet data for {ticker}")

            # Fetch balance sheet data for the current ticker symbol
            tkr = yf.Ticker(ticker)
            bs_data = tkr.balance_sheet

            # Check if balance sheet data is available and not empty
            if bs_data is not None and not bs_data.empty:
                # Reset index to include balance sheet line item descriptions as a regular column
                bs_data.reset_index(inplace=True)
                # Add a new column for the ticker symbol
                bs_data.insert(0, 'Ticker', ticker)
                # Convert DataFrame to text format with pipe "|" delimiter
                text = bs_data.to_csv(index=False, sep='|')
                # Concatenate the text with the all_text string
                all_text += text + "\n"
                logger.info(f"Successfully fetched balance sheet data for {ticker}")
            else:
                # If no balance sheet data is available for the current ticker, append a message to all_text
                all_text += f"No balance sheet data available for {ticker}\n"
                logger.warning(f"No balance sheet data available for {ticker}")

            # Introduce a 10-second delay before processing the next ticker symbol
            logger.debug(f"Waiting 10 seconds before processing next ticker")
            time.sleep(10)

        return all_text
    except Exception as e:
        logger.error(f"Error fetching balance sheet: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"

# Define the new function to get adjusted closing prices
def get_adj_close(tickers: List[str], start_date: str, end_date: str):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}

    try:
        print(f"Fetching data for tickers: {tickers}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Initialize empty DataFrame to store results
        all_data = pd.DataFrame()
        
        # Fetch data for each ticker using Ticker().history()
        for ticker_symbol in tickers:
            print(f"Fetching data for {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)
            ticker_data = ticker.history(start=start_date, end=end_date)
            
            print(f"Data received for {ticker_symbol}:")
            print(ticker_data.columns)
            print(f"Shape: {ticker_data.shape}")
            
            if not ticker_data.empty:
                # Use Close price if Adj Close is not available
                price_column = 'Close' if 'Adj Close' not in ticker_data.columns else 'Adj Close'
                all_data[ticker_symbol] = ticker_data[price_column]
        
        if all_data.empty:
            print("No data available for any ticker")
            return {"error": "No data available for the specified parameters"}
            
        print("\nFinal processed data:")
        print(all_data)
        
        # Convert to JSON format
        json_data = {}
        for date in all_data.index:
            date_str = date.strftime("%Y-%m-%d")
            json_data[date_str] = {}
            for ticker in tickers:
                if ticker in all_data.columns:
                    json_data[date_str][ticker] = all_data.at[date, ticker]
                else:
                    json_data[date_str][ticker] = "Data not available"
        
        return json_data
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error traceback:", traceback.format_exc())
        return {"error": f"An error occurred while fetching data: {str(e)}"}

# Define function to fetch annual income statement for multiple tickers
def fetch_income_statement_as_text(tickers: List[str]):
    try:
        all_text = ""
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            income_stmt = tkr.income_stmt
            if income_stmt is not None and not income_stmt.empty:
                income_stmt.reset_index(inplace=True)
                income_stmt.insert(0, 'Ticker', ticker)
                text = income_stmt.to_csv(index=False, sep='|')
                all_text += text + "\n"
            else:
                all_text += f"No income statement data available for {ticker}\n"
            time.sleep(10)
        return all_text
    except Exception as e:
        return f"Error: {str(e)}"

# Define function to fetch cash flow statement for multiple tickers
def fetch_cash_flow_as_text(tickers: List[str]):
    try:
        all_text = ""
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            cash_flow = tkr.cashflow
            if cash_flow is not None and not cash_flow.empty:
                cash_flow.reset_index(inplace=True)
                cash_flow.insert(0, 'Ticker', ticker)
                text = cash_flow.to_csv(index=False, sep='|')
                all_text += text + "\n"
            else:
                all_text += f"No cash flow statement data available for {ticker}\n"
            time.sleep(10)
        return all_text
    except Exception as e:
        return f"Error: {str(e)}"

# Define function to fetch quarterly income statement for multiple tickers
def fetch_quarterly_income_statement_as_text(tickers: List[str]):
    try:
        all_text = ""
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            quarterly_income_stmt = tkr.quarterly_income_stmt
            if quarterly_income_stmt is not None and not quarterly_income_stmt.empty:
                quarterly_income_stmt.reset_index(inplace=True)
                quarterly_income_stmt.insert(0, 'Ticker', ticker)
                text = quarterly_income_stmt.to_csv(index=False, sep='|')
                all_text += text + "\n"
            else:
                all_text += f"No quarterly income statement data available for {ticker}\n"
            time.sleep(10)
        return all_text
    except Exception as e:
        return f"Error: {str(e)}"

# Define the new endpoint to fetch income statement
@app.get("/get-income-statement/")
def get_income_statement_endpoint(tickers: str):
    tickers_list = tickers.split(",")
    return fetch_income_statement_as_text(tickers_list)

# Define the new endpoint to fetch cash flow statement
@app.get("/get-cash-flow/")
def get_cash_flow_endpoint(tickers: str):
    tickers_list = tickers.split(",")
    return fetch_cash_flow_as_text(tickers_list)

# Define the new endpoint to fetch quarterly income statement
@app.get("/get-quarterly-income-statement/")
def get_quarterly_income_statement_endpoint(tickers: str):
    tickers_list = tickers.split(",")
    return fetch_quarterly_income_statement_as_text(tickers_list)

# Define the existing endpoint to fetch balance sheet
@app.get("/get-balance-sheet/")
def get_balance_sheet_endpoint(tickers: str):
    tickers_list = tickers.split(",")
    return fetch_balance_sheet_as_text(tickers_list)

# Define the existing endpoint to fetch adjusted closing prices
@app.get("/get-adj-close/")
def get_adj_close_endpoint(tickers: str, start_date: str, end_date: str):
    tickers_list = tickers.split(",")
    return get_adj_close(tickers_list, start_date, end_date)

# Code to get market capitalization, free float shares, and shares outstanding
@app.get("/get-market-data/", operation_id="get_market_data")
def get_market_data(
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Fetches key market data metrics for specified stock tickers from Yahoo Finance.
    
    This endpoint provides three essential market metrics for each ticker:
    - market_cap: The total market value of all outstanding shares in USD
    - free_float: The number of shares available for public trading (excluding restricted shares)
    - total_shares_outstanding: The total number of shares issued by the company
    
    These metrics help assess company size, liquidity, and ownership structure.
    
    Example query: get_market_data?tickers=AAPL,MSFT,GOOG
    """
    tickers_list = tickers.split(",")
    market_data = {}
    try:
        for ticker in tickers_list:
            try:
                logger.info(f"Processing market data for ticker: {ticker}")
                stock = yf.Ticker(ticker)
                info = stock.info
                if not info:
                    logger.warning(f"No info available for ticker: {ticker}")
                    market_data[ticker] = {
                        "error": "No data available",
                        "market_cap": None,
                        "free_float": None,
                        "total_shares_outstanding": None
                    }
                    continue
                
                data = {
                    "market_cap": info.get("marketCap"),
                    "free_float": info.get("floatShares"),
                    "total_shares_outstanding": info.get("sharesOutstanding")
                }
                logger.info(f"Data retrieved for {ticker}: {data}")
                market_data[ticker] = data
                
            except Exception as ticker_error:
                logger.error(f"Error processing ticker {ticker}: {str(ticker_error)}")
                logger.error(traceback.format_exc())
                market_data[ticker] = {
                    "error": f"Failed to process: {str(ticker_error)}",
                    "market_cap": None,
                    "free_float": None,
                    "total_shares_outstanding": None
                }
                
        return market_data
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {"error": error_msg}

# Define the new function to get all price data
def get_all_prices(tickers: List[str], start_date: str, end_date: str):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}

    try:
        logger.info(f"Fetching all price data for tickers: {tickers}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Initialize empty DataFrame to store results
        all_data = pd.DataFrame()
        
        # Fetch data for each ticker
        for ticker_symbol in tickers:
            logger.info(f"Fetching data for {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)
            ticker_data = ticker.history(start=start_date, end=end_date)
            
            logger.info(f"Data received for {ticker_symbol}: {ticker_data.shape} rows")
            
            if not ticker_data.empty:
                # Create multi-level column names for each price type
                for column in ticker_data.columns:
                    all_data[f"{ticker_symbol}_{column}"] = ticker_data[column]
        
        if all_data.empty:
            logger.warning("No data available for any ticker")
            return {"error": "No data available for the specified parameters"}
            
        logger.info(f"Final processed data shape: {all_data.shape}")
        
        # Convert to JSON format with date as key and all price types for each ticker
        json_data = {}
        for date_idx in all_data.index:
            # Convert datetime to string in YYYY-MM-DD format
            date_str = date_idx.strftime("%Y-%m-%d")
            json_data[date_str] = {}
            
            for ticker in tickers:
                json_data[date_str][ticker] = {}
                # Add all available price types for this ticker
                for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']:
                    col_name = f"{ticker}_{column}"
                    if col_name in all_data.columns:
                        # Convert numpy types to Python native types
                        value = all_data.at[date_idx, col_name]
                        if pd.isna(value):
                            json_data[date_str][ticker][column] = None
                        else:
                            # Convert numpy types to Python native types
                            if isinstance(value, (np.integer, np.floating)):
                                value = value.item()
                            json_data[date_str][ticker][column] = value
        
        return json_data
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"An error occurred while fetching data: {str(e)}"}

# Define the new endpoint to fetch all price data
@app.get("/get-all-prices/", operation_id="get_all_prices")
def get_all_prices_endpoint(
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    ),
    start_date: str = Query(
        description="Start date in YYYY-MM-DD format (e.g., '2023-01-01')"
    ),
    end_date: str = Query(
        description="End date in YYYY-MM-DD format (e.g., '2023-12-31')"
    )
):
    """
    Fetches comprehensive historical price data for specified Yahoo Finance tickers.
    
    This endpoint retrieves complete historical price data including:
    - Open prices
    - High prices 
    - Low prices
    - Close prices
    - Trading volumes
    - Dividend payments
    - Stock splits
    
    Data is returned in a JSON format organized by date, with each date containing 
    data for all requested tickers. For each ticker, all available price metrics 
    are provided. This gives a complete picture of price movement and trading activity
    over the specified time period.
    
    The data can be used for:
    - Technical analysis and charting
    - Performance tracking
    - Dividend history analysis
    - Trading volume analysis
    - Volatility calculations
    
    Example query: get-all-prices?tickers=AAPL,MSFT&start_date=2023-01-01&end_date=2023-01-31
    """
    tickers_list = tickers.split(",")
    return get_all_prices(tickers_list, start_date, end_date)

# Define function to fetch detailed company information
def fetch_detailed_info(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            info = tkr.info
            
            if info:
                # Extract main company info (excluding longBusinessSummary and companyOfficers)
                main_info = {}
                for key, value in info.items():
                    if key not in ['longBusinessSummary', 'companyOfficers']:
                        # Convert numpy types to native Python types
                        if isinstance(value, (np.integer, np.floating)):
                            value = value.item()
                        main_info[key] = value
                
                # Extract company officers info
                officers_info = info.get('companyOfficers', [])
                
                result[ticker] = {
                    'main_info': main_info,
                    'officers': officers_info
                }
            else:
                result[ticker] = {
                    'error': 'No detailed information available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        return {'error': str(e)}

# Define the new endpoint for detailed company information
@app.get("/get-detailed-info/", operation_id="get_detailed_info")
def get_detailed_info_endpoint(
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves comprehensive company information and data points from Yahoo Finance.
    
    Note: This endpoint may take 30 seconds to 3 minutes to respond due to the extensive data being fetched.
    
    This endpoint returns approximately 200 structured data points organized across 11 key categories:
    
    1. Company Information - Basic details about the company, sector, industry, etc.
    2. Stock Price & Market Activity - Current and historical price information
    3. Valuation & Ratios - P/E ratio, PEG, price-to-book, etc.
    4. Dividend Information - Yield, payment history, ex-dividend dates
    5. Earnings & Financials - EPS, revenue, profit margins, etc.
    6. Share Statistics - Float, outstanding shares, short interest
    7. Analyst Price Targets - Consensus targets, recommendations
    8. Technical Indicators - Moving averages, relative strength
    9. Earnings Call / Fiscal Calendar - Upcoming events and dates
    10. Other - Miscellaneous financial metrics and data
    11. Key Executives - Information about company leadership
    
    The data is structured with 'main_info' containing all general data points and 'officers' 
    containing information about key executives.
    
    Example query: get_detailed_info?tickers=AAPL,MSFT,GOOG
    
    Response time: Expect 30 seconds to 3 minutes depending on the number of tickers and data availability.
    """
    tickers_list = tickers.split(",")
    return fetch_detailed_info(tickers_list)

# Define function to fetch income statement in Excel-friendly format
def fetch_income_statement_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            income_stmt = tkr.income_stmt
            
            if income_stmt is not None and not income_stmt.empty:
                # Reset index to make the metric names a column
                income_stmt.reset_index(inplace=True)
                
                # Convert the data to a dictionary format
                metrics = income_stmt['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None 
                        for d in income_stmt.columns[1:]]  # Skip 'index' column
                
                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = income_stmt[income_stmt['index'] == metric].index[0]
                        value = income_stmt.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)
                
                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No income statement data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        return {'error': str(e)}

# Define the new Excel-specific endpoint for income statement
@app.get("/excel/get-income-statement/", operation_id="get_income_statement_excel")
def get_income_statement_excel_endpoint(
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves annual income statements for specified Yahoo Finance tickers in a structured JSON format.
    
    This endpoint pulls complete annual income statement data from Yahoo Finance with all available line items, including:
    - Total Revenue
    - Cost of Revenue
    - Gross Profit
    - Operating Expenses
    - Operating Income
    - Net Income
    - EPS metrics
    - And many more financial metrics
    
    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across reporting periods
    
    The structured format makes it easy to:
    - Compare performance across different years
    - Analyze financial trends
    - Calculate financial ratios
    - Generate reports or visualizations
    
    Example query: excel/get-income-statement?tickers=AAPL,MSFT
    """
    tickers_list = tickers.split(",")
    return fetch_income_statement_excel(tickers_list)

# Define function to fetch balance sheet in Excel-friendly format
def fetch_balance_sheet_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            balance_sheet = tkr.balance_sheet
            
            if balance_sheet is not None and not balance_sheet.empty:
                # Reset index to make the metric names a column
                balance_sheet.reset_index(inplace=True)
                
                # Convert the data to a dictionary format
                metrics = balance_sheet['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None 
                        for d in balance_sheet.columns[1:]]  # Skip 'index' column
                
                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = balance_sheet[balance_sheet['index'] == metric].index[0]
                        value = balance_sheet.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)
                
                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No balance sheet data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        return {'error': str(e)}

# Define the new Excel-specific endpoint for balance sheet
@app.get("/excel/get-balance-sheet/", operation_id="get_balance_sheet_excel")
def get_balance_sheet_excel_endpoint(
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves annual balance sheets for specified Yahoo Finance tickers in a structured JSON format.
    
    This endpoint pulls complete annual balance sheet data from Yahoo Finance with all available line items, including:
    - Assets (Current Assets, Long-term Assets)
    - Liabilities (Current Liabilities, Long-term Debt)
    - Shareholders' Equity
    - Cash and Equivalents
    - Inventory
    - Accounts Receivable/Payable
    - And many more balance sheet items
    
    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across reporting periods
    
    The structured format makes it easy to:
    - Assess financial position at different points in time
    - Calculate financial ratios (quick ratio, debt-to-equity, etc.)
    - Evaluate liquidity and solvency
    - Track changes in company financial structure
    
    Example query: excel/get-balance-sheet?tickers=AAPL,MSFT
    """
    tickers_list = tickers.split(",")
    return fetch_balance_sheet_excel(tickers_list)

# Define function to fetch cash flow in Excel-friendly format
def fetch_cash_flow_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            cash_flow = tkr.cashflow
            
            if cash_flow is not None and not cash_flow.empty:
                # Reset index to make the metric names a column
                cash_flow.reset_index(inplace=True)
                
                # Convert the data to a dictionary format
                metrics = cash_flow['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None 
                        for d in cash_flow.columns[1:]]  # Skip 'index' column
                
                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = cash_flow[cash_flow['index'] == metric].index[0]
                        value = cash_flow.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)
                
                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No cash flow data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        return {'error': str(e)}

# Define the new Excel-specific endpoint for cash flow
@app.get("/excel/get-cash-flow/", operation_id="get_cash_flow_excel")
def get_cash_flow_excel_endpoint(
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves annual cash flow statements for specified Yahoo Finance tickers in a structured JSON format.
    
    This endpoint pulls complete annual cash flow data from Yahoo Finance with all available line items, including:
    - Operating Cash Flow
    - Investing Cash Flow
    - Financing Cash Flow
    - Capital Expenditures
    - Dividends Paid
    - Stock Repurchases
    - Free Cash Flow
    - And many more cash flow metrics
    
    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across reporting periods
    
    The structured format makes it easy to:
    - Analyze cash generation and usage over time
    - Assess operational efficiency
    - Evaluate investment and financing activities
    - Determine sustainability of dividends and share repurchases
    
    Example query: excel/get-cash-flow?tickers=AAPL,MSFT
    """
    tickers_list = tickers.split(",")
    return fetch_cash_flow_excel(tickers_list)

# Define function to fetch quarterly income statement in Excel-friendly format
def fetch_quarterly_income_statement_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            quarterly_stmt = tkr.quarterly_income_stmt
            
            if quarterly_stmt is not None and not quarterly_stmt.empty:
                # Reset index to make the metric names a column
                quarterly_stmt.reset_index(inplace=True)
                
                # Convert the data to a dictionary format
                metrics = quarterly_stmt['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None 
                        for d in quarterly_stmt.columns[1:]]  # Skip 'index' column
                
                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = quarterly_stmt[quarterly_stmt['index'] == metric].index[0]
                        value = quarterly_stmt.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)
                
                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No quarterly income statement data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        return {'error': str(e)}

# Define the new Excel-specific endpoint for quarterly income statement
@app.get("/excel/get-quarterly-income-statement/", operation_id="get_quarterly_income_statement_excel")
def get_quarterly_income_statement_excel_endpoint(
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves quarterly income statements for specified Yahoo Finance tickers in a structured JSON format.
    
    This endpoint pulls complete quarterly income statement data from Yahoo Finance with all available line items, including:
    - Total Revenue
    - Cost of Revenue
    - Gross Profit
    - Operating Expenses
    - Operating Income
    - Net Income
    - EPS metrics
    - And many more financial metrics
    
    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of quarterly reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across quarters
    
    The structured format makes it easy to:
    - Identify seasonal patterns in business performance
    - Track quarter-over-quarter growth rates
    - Analyze short-term business trends
    - Compare quarterly results to analyst expectations
    - Detect turning points in business performance more quickly than annual data
    
    Example query: excel/get-quarterly-income-statement?tickers=AAPL,MSFT
    """
    tickers_list = tickers.split(",")
    return fetch_quarterly_income_statement_excel(tickers_list)

# Create MCP server and include all relevant endpoints
mcp = FastApiMCP(
    app,
    name="YFinance MCP API",
    description="MCP server for YFinance data endpoints. Note: Some operations may take up to 3 minutes due to data fetching requirements.",
    base_url="http://localhost:8000",
    include_operations=[
        "get_market_data", 
        "get_detailed_info", 
        "get_all_prices",
        "get_income_statement_excel",
        "get_balance_sheet_excel",
        "get_cash_flow_excel",
        "get_quarterly_income_statement_excel"
    ],
    # Inject custom httpx client with 5-minute (300 seconds) timeout
    http_client=httpx.AsyncClient(timeout=300.0)
)

# Mount the MCP server to the FastAPI app
mcp.mount()

# Log available operations when server starts
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 40)
    logger.info("FastAPI server with MCP integration is starting!")
    logger.info("MCP endpoint is available at: http://localhost:8000/mcp")
    logger.info("Using custom httpx client with 5-minute (300 second) timeout")
    
    # Log all available routes and their operation IDs
    logger.info("Available routes and operation IDs:")
    for route in app.routes:
        if hasattr(route, "operation_id"):
            logger.info(f"Route: {route.path}, Operation ID: {route.operation_id}")
    
    logger.info("MCP server exposing financial data endpoints from Yahoo Finance")
    logger.info("=" * 40)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 40)
    logger.info("FastAPI server is shutting down")
    logger.info("=" * 40)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)