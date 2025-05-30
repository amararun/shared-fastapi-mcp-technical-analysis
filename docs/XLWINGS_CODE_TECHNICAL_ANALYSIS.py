import xlwings as xw
import pandas as pd
from finta import TA
import requests
from xlwings import script
from datetime import datetime
import os
import matplotlib.pyplot as plt
import tempfile
import base64
import time
import asyncio

# FastAPI CORS Configuration for Excel/xlwings:
# When setting up a FastAPI server to work with xlwings in Excel, use these origins:
#
# from fastapi.middleware.cors import CORSMiddleware
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "https://addin.xlwings.org",    # Main xlwings add-in domain - THIS IS THE PRIMARY ONE NEEDED
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# Note: The actual request comes from the Excel WebView2 browser component via the
# xlwings add-in hosted at addin.xlwings.org, NOT from Excel/Office domains directly.



@script
async def create_technicals(book: xw.Book):
    """Create both daily and weekly technical indicators for a ticker using price data."""
    print("🔵 STARTING create_technicals (Daily & Weekly)")
    
    # Get the MASTER sheet
    master_sheet = book.sheets["MASTER"]
    
    # Read ticker, start_date, and end_date from MASTER sheet
    ticker = str(master_sheet["B3"].value).strip().upper() if master_sheet["B3"].value else None
    daily_start_date_raw = master_sheet["D3"].value
    daily_end_date_raw = master_sheet["F3"].value
    weekly_start_date_raw = master_sheet["D4"].value
    weekly_end_date_raw = master_sheet["F4"].value
    
    if not all([ticker, daily_start_date_raw, daily_end_date_raw, weekly_start_date_raw, weekly_end_date_raw]):
        master_sheet["B8"].value = "Please enter ticker (B3), daily dates (D3,F3), and weekly dates (D4,F4)"
        return
    
    # Convert Excel dates to YYYY-MM-DD format for daily
    if isinstance(daily_start_date_raw, (datetime, pd.Timestamp)):
        daily_start_date = daily_start_date_raw.strftime("%Y-%m-%d")
    else:
        daily_start_date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(daily_start_date_raw))
        daily_start_date = daily_start_date.strftime("%Y-%m-%d")
    
    if isinstance(daily_end_date_raw, (datetime, pd.Timestamp)):
        daily_end_date = daily_end_date_raw.strftime("%Y-%m-%d")
    else:
        daily_end_date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(daily_end_date_raw))
        daily_end_date = daily_end_date.strftime("%Y-%m-%d")
    
    # Convert Excel dates to YYYY-MM-DD format for weekly
    if isinstance(weekly_start_date_raw, (datetime, pd.Timestamp)):
        weekly_start_date = weekly_start_date_raw.strftime("%Y-%m-%d")
    else:
        weekly_start_date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(weekly_start_date_raw))
        weekly_start_date = weekly_start_date.strftime("%Y-%m-%d")
    
    if isinstance(weekly_end_date_raw, (datetime, pd.Timestamp)):
        weekly_end_date = weekly_end_date_raw.strftime("%Y-%m-%d")
    else:
        weekly_end_date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(weekly_end_date_raw))
        weekly_end_date = weekly_end_date.strftime("%Y-%m-%d")
    
    # Process daily data
    print(f"\nProcessing Daily Data for {ticker}")
    daily_api_url = f"https://yfin.hosting.tigzig.com/get-all-prices/?tickers={ticker}&start_date={daily_start_date}&end_date={daily_end_date}"
    print(f"Daily API URL: {daily_api_url}")
    
    daily_response = requests.get(daily_api_url)
    print(f"Daily Response status: {daily_response.status_code}")
    
    if daily_response.ok:
        daily_data = daily_response.json()
        if isinstance(daily_data, dict) and not daily_data.get("error"):
            # Process daily data
            print("\nProcessing daily data...")
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
            daily_display_df = daily_df.copy()
            daily_display_df['EMA_12'] = TA.EMA(daily_df, 12)
            daily_display_df['EMA_26'] = TA.EMA(daily_df, 26)
            daily_display_df['RSI_14'] = TA.RSI(daily_df)
            daily_display_df['ROC_14'] = TA.ROC(daily_df, 14)
            
            macd = TA.MACD(daily_df)
            if isinstance(macd, pd.DataFrame):
                daily_display_df['MACD_12_26'] = macd['MACD']
                daily_display_df['MACD_SIGNAL_9'] = macd['SIGNAL']
            
            bb = TA.BBANDS(daily_df)
            if isinstance(bb, pd.DataFrame):
                daily_display_df['BBANDS_UPPER_20_2'] = bb['BB_UPPER']
                daily_display_df['BBANDS_MIDDLE_20_2'] = bb['BB_MIDDLE']
                daily_display_df['BBANDS_LOWER_20_2'] = bb['BB_LOWER']
            
            # Rename columns for display
            daily_display_df.rename(columns={
                    'date': 'DATE',
                    'open': 'OPEN',
                    'high': 'HIGH',
                    'low': 'LOW',
                    'close': 'CLOSE',
                'volume': 'VOLUME'
                }, inplace=True)
            
            # Reorder columns to ensure DATE is first
            columns_order = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 
                           'EMA_12', 'EMA_26', 'RSI_14', 'ROC_14', 
                           'MACD_12_26', 'MACD_SIGNAL_9',
                           'BBANDS_UPPER_20_2', 'BBANDS_MIDDLE_20_2', 'BBANDS_LOWER_20_2']
            daily_display_df = daily_display_df[columns_order]
            
            # Write daily data to sheet
            try:
                print("\nWriting daily data to PRICES_DAILY sheet...")
                if "PRICES_DAILY" in [s.name for s in book.sheets]:
                    prices_daily_sheet = book.sheets["PRICES_DAILY"]
                    prices_daily_sheet.clear()
                else:
                    prices_daily_sheet = book.sheets.add(name="PRICES_DAILY", after=master_sheet)
                
                # Write header and data
                start_date_dt = pd.to_datetime(daily_start_date)
                end_date_dt = pd.to_datetime(daily_end_date)
                header = f"Daily Prices and Technical Indicators for {ticker} from {start_date_dt.strftime('%d%b %Y')} to {end_date_dt.strftime('%d%b %Y')}"
                prices_daily_sheet["A1"].value = header
                header_range = prices_daily_sheet.range("A1:H1")
                header_range.color = "#A7D9AB"
                
                prices_daily_sheet["A2"].value = daily_display_df.columns.tolist()
                prices_daily_sheet["A3"].value = daily_display_df.values.tolist()
                
                data_range = prices_daily_sheet["A2"].resize(len(daily_display_df) + 1, len(daily_display_df.columns))
                prices_daily_sheet.tables.add(data_range)
                
                print("✓ Daily data written successfully")
                
                # Create daily chart
                if "CHARTS_DAILY" in [s.name for s in book.sheets]:
                    charts_daily_sheet = book.sheets["CHARTS_DAILY"]
                    charts_daily_sheet.clear()
                else:
                    charts_daily_sheet = book.sheets.add(name="CHARTS_DAILY", after=prices_daily_sheet)
                
                create_chart(charts_daily_sheet, daily_display_df, ticker, "Technical Analysis Charts", "Daily")
                print("Daily chart created successfully")
                
            except Exception as e:
                print(f"[ERROR] Error processing daily data: {str(e)}")
                master_sheet["B8"].value = f"Error processing daily data: {str(e)}"
                return
            
            # Process weekly data
            print(f"\nProcessing Weekly Data for {ticker}")
            weekly_api_url = f"https://yfin.hosting.tigzig.com/get-all-prices/?tickers={ticker}&start_date={weekly_start_date}&end_date={weekly_end_date}"
            print(f"Weekly API URL: {weekly_api_url}")
            
            weekly_response = requests.get(weekly_api_url)
            print(f"Weekly Response status: {weekly_response.status_code}")
            
            if weekly_response.ok:
                weekly_data = weekly_response.json()
                if isinstance(weekly_data, dict) and not weekly_data.get("error"):
                    # Process weekly data
                    print("\nProcessing weekly data...")
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
                    weekly_display_df = weekly_df.copy()
                    weekly_display_df['EMA_12'] = TA.EMA(weekly_df, 12)
                    weekly_display_df['EMA_26'] = TA.EMA(weekly_df, 26)
                    weekly_display_df['RSI_14'] = TA.RSI(weekly_df)
                    weekly_display_df['ROC_14'] = TA.ROC(weekly_df, 14)
                    
                    macd = TA.MACD(weekly_df)
                    if isinstance(macd, pd.DataFrame):
                        weekly_display_df['MACD_12_26'] = macd['MACD']
                        weekly_display_df['MACD_SIGNAL_9'] = macd['SIGNAL']
                    
                    bb = TA.BBANDS(weekly_df)
                    if isinstance(bb, pd.DataFrame):
                        weekly_display_df['BBANDS_UPPER_20_2'] = bb['BB_UPPER']
                        weekly_display_df['BBANDS_MIDDLE_20_2'] = bb['BB_MIDDLE']
                        weekly_display_df['BBANDS_LOWER_20_2'] = bb['BB_LOWER']
                    
                    # Rename columns for display
                    weekly_display_df.rename(columns={
                        'Date': 'DATE',
                        'Open': 'OPEN',
                        'High': 'HIGH',
                        'Low': 'LOW',
                        'Close': 'CLOSE',
                        'Volume': 'VOLUME'
                    }, inplace=True)
                    
                    # Write weekly data to sheet
                    try:
                        print("\nWriting weekly data to PRICES_WEEKLY sheet...")
                        if "PRICES_WEEKLY" in [s.name for s in book.sheets]:
                            prices_weekly_sheet = book.sheets["PRICES_WEEKLY"]
                            prices_weekly_sheet.clear()
                        else:
                            prices_weekly_sheet = book.sheets.add(name="PRICES_WEEKLY", after=prices_daily_sheet)
                        
                        # Write header and data
                        start_date_dt = pd.to_datetime(weekly_start_date)
                        end_date_dt = pd.to_datetime(weekly_end_date)
                        header = f"Weekly Prices and Technical Indicators for {ticker} from {start_date_dt.strftime('%d%b %Y')} to {end_date_dt.strftime('%d%b %Y')}"
                        prices_weekly_sheet["A1"].value = header
                        header_range = prices_weekly_sheet.range("A1:H1")
                        header_range.color = "#A7D9AB"
                        
                        prices_weekly_sheet["A2"].value = weekly_display_df.columns.tolist()
                        prices_weekly_sheet["A3"].value = weekly_display_df.values.tolist()
                        
                        data_range = prices_weekly_sheet["A2"].resize(len(weekly_display_df) + 1, len(weekly_display_df.columns))
                        prices_weekly_sheet.tables.add(data_range)
                        
                        print("✓ Weekly data written successfully")
                        
                        # Create weekly chart
                        if "CHARTS_WEEKLY" in [s.name for s in book.sheets]:
                            charts_weekly_sheet = book.sheets["CHARTS_WEEKLY"]
                            charts_weekly_sheet.clear()
                        else:
                            charts_weekly_sheet = book.sheets.add(name="CHARTS_WEEKLY", after=prices_weekly_sheet)
                        
                        create_chart(charts_weekly_sheet, weekly_display_df, ticker, "Technical Analysis Charts", "Weekly")
                        print("Weekly chart created successfully")
                        
                    except Exception as e:
                        print(f"[ERROR] Error processing weekly data: {str(e)}")
                        master_sheet["B8"].value = f"Error processing weekly data: {str(e)}"
                        return
                    
                    print("\nSuccessfully processed both daily and weekly data")
                else:
                    error_msg = weekly_data.get("error") if isinstance(weekly_data, dict) else "Invalid weekly response format"
                    master_sheet["B8"].value = f"Weekly data error: {error_msg}"
            else:
                master_sheet["B8"].value = "Weekly data service temporarily unavailable"
        else:
            error_msg = daily_data.get("error") if isinstance(daily_data, dict) else "Invalid daily response format"
            master_sheet["B8"].value = f"Daily data error: {error_msg}"
    else:
        master_sheet["B8"].value = "Daily data service temporarily unavailable"
    
    print("🟢 ENDING create_technicals (Daily & Weekly)")

def combine_charts(daily_path, weekly_path, daily_start, daily_end, weekly_start, weekly_end):
    """Combine daily and weekly charts into a single side-by-side image for PDF output."""
    # Read the images
    daily_img = plt.imread(daily_path)
    weekly_img = plt.imread(weekly_path)
    
    # Convert Excel dates to readable format
    try:
        def convert_excel_date(date_val):
            if isinstance(date_val, (datetime, pd.Timestamp)):
                return date_val
            # Convert Excel serial number to datetime
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(date_val))
        
        # Convert dates for daily chart
        daily_start_date = convert_excel_date(daily_start)
        daily_end_date = convert_excel_date(daily_end)
        
        # Convert dates for weekly chart
        weekly_start_date = convert_excel_date(weekly_start)
        weekly_end_date = convert_excel_date(weekly_end)
        
        # Format dates for display
        daily_start_str = daily_start_date.strftime('%d %b %Y')
        daily_end_str = daily_end_date.strftime('%d %b %Y')
        weekly_start_str = weekly_start_date.strftime('%d %b %Y')
        weekly_end_str = weekly_end_date.strftime('%d %b %Y')
        
        print("\nDate Ranges:")
        print(f"Daily: {daily_start_str} to {daily_end_str}")
        print(f"Weekly: {weekly_start_str} to {weekly_end_str}")
        
    except Exception as e:
        print(f"[ERROR] Error processing dates: {str(e)}")
        return None
    
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

@script
def get_technical_analysis_from_gemini(book: xw.Book):
    """Get technical analysis from Gemini API using separate charts for analysis but combined chart for PDF."""
    print("🔵 STARTING get_technical_analysis_from_gemini")
    
    # Debug flag - set to False to prevent HTML printing in logs
    DEBUG_HTML = False
    
    # Get the required sheets
    master_sheet = book.sheets["MASTER"]
    prices_daily_sheet = book.sheets["PRICES_DAILY"]
    prices_weekly_sheet = book.sheets["PRICES_WEEKLY"]
    
    # Clear previous results
    master_sheet["A11:B12"].clear_contents()
    
    # Read model name and API key from MASTER sheet
    model_name = master_sheet["B7"].value
    api_key = master_sheet["B8"].value
    
    # Get ticker and date parameters from MASTER sheet
    ticker = str(master_sheet["B3"].value).strip().upper() if master_sheet["B3"].value else None
    daily_start_date = master_sheet["D3"].value
    daily_end_date = master_sheet["F3"].value
    weekly_start_date = master_sheet["D4"].value
    weekly_end_date = master_sheet["F4"].value
    
    if not all([model_name, api_key, ticker, daily_start_date, daily_end_date, weekly_start_date, weekly_end_date]):
        master_sheet["A11"].value = "Error"
        master_sheet["A12"].value = "Please enter all required parameters (model name, API key, ticker, start dates, end dates)"
        return
    
    # Get both CHARTS_DAILY and CHARTS_WEEKLY sheets to get the chart image paths
    charts_daily_sheet = book.sheets["CHARTS_DAILY"]
    charts_weekly_sheet = book.sheets["CHARTS_WEEKLY"]
    
    # Get chart paths from cell A1 in each sheet
    daily_chart_path = charts_daily_sheet["A1"].value
    weekly_chart_path = charts_weekly_sheet["A1"].value
    
    print(f"Found daily chart at path: {daily_chart_path}")
    print(f"Found weekly chart at path: {weekly_chart_path}")
    
    if not daily_chart_path or not os.path.exists(daily_chart_path):
        master_sheet["A11"].value = "Error"
        master_sheet["A12"].value = "Please run create_technicals first to generate the daily chart"
        return
        
    if not weekly_chart_path or not os.path.exists(weekly_chart_path):
        master_sheet["A11"].value = "Error"
        master_sheet["A12"].value = "Please run create_weekly_technicals first to generate the weekly chart"
        return
    
    # Verify the paths are different
    if daily_chart_path == weekly_chart_path:
        master_sheet["A11"].value = "Error"
        master_sheet["A12"].value = "Error: Daily and weekly chart paths are the same"
        return
    
    # Upload both charts separately for Gemini
    try:
        print("\nUploading charts to server...")
        
        # Upload daily chart
        daily_files = {
            'file': ('daily_chart.png', open(daily_chart_path, 'rb'), 'image/png')
        }
        daily_upload_response = requests.post(
            "https://mdtopdf.hosting.tigzig.com/api/upload-image",
            files=daily_files
        )
        
        if not daily_upload_response.ok:
            error_msg = f"Failed to upload daily image: {daily_upload_response.status_code}"
            print(f"ERROR: {error_msg}")
            master_sheet["A11"].value = "Error"
            master_sheet["A12"].value = error_msg
            return
        
        daily_upload_data = daily_upload_response.json()
        daily_image_path = daily_upload_data['image_path']
        print(f"Daily image uploaded successfully. Path: {daily_image_path}")
        
        # Upload weekly chart
        weekly_files = {
            'file': ('weekly_chart.png', open(weekly_chart_path, 'rb'), 'image/png')
        }
        weekly_upload_response = requests.post(
            "https://mdtopdf.hosting.tigzig.com/api/upload-image",
            files=weekly_files
        )
        
        if not weekly_upload_response.ok:
            error_msg = f"Failed to upload weekly image: {weekly_upload_response.status_code}"
            print(f"ERROR: {error_msg}")
            master_sheet["A11"].value = "Error"
            master_sheet["A12"].value = error_msg
            return
        
        weekly_upload_data = weekly_upload_response.json()
        weekly_image_path = weekly_upload_data['image_path']
        print(f"Weekly image uploaded successfully. Path: {weekly_image_path}")
        
        # Create combined chart with date ranges from original parameters
        print("\nCreating combined chart for PDF...")
        combined_chart_path = combine_charts(daily_chart_path, weekly_chart_path, 
                                          daily_start=daily_start_date, daily_end=daily_end_date,
                                          weekly_start=weekly_start_date, weekly_end=weekly_end_date)
        
        # Upload combined chart
        combined_files = {
            'file': ('combined_chart.png', open(combined_chart_path, 'rb'), 'image/png')
        }
        combined_upload_response = requests.post(
            "https://mdtopdf.hosting.tigzig.com/api/upload-image",
            files=combined_files
        )
        
        if not combined_upload_response.ok:
            error_msg = f"Failed to upload combined image: {combined_upload_response.status_code}"
            print(f"ERROR: {error_msg}")
            master_sheet["A11"].value = "Error"
            master_sheet["A12"].value = error_msg
            return
        
        combined_upload_data = combined_upload_response.json()
        combined_image_path = combined_upload_data['image_path']
        print(f"Combined image uploaded successfully. Path: {combined_image_path}")
        
    except Exception as e:
        error_msg = f"Error uploading images: {str(e)}"
        print(f"ERROR: {error_msg}")
        master_sheet["A11"].value = "Error"
        master_sheet["A12"].value = error_msg
        return
    
    # Get the data from sheets
    daily_data_range = prices_daily_sheet["A2"].expand()
    weekly_data_range = prices_weekly_sheet["A2"].expand()
    daily_data = daily_data_range.options(pd.DataFrame, index=False).value
    weekly_data = weekly_data_range.options(pd.DataFrame, index=False).value
    
    # Get the latest data points
    latest_daily = daily_data.iloc[-1]
    latest_weekly = weekly_data.iloc[-1]
    
    # Get last 20 rows for additional data to send to Gemini (keeping all columns)
    last_20_days = daily_data.tail(20)
    last_20_weeks = weekly_data.tail(20)
    
    # Format the last 20 rows data as markdown tables for Gemini analysis
    # This is separate from the HTML table that's used for display
    def format_data_for_analysis(df, title):
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
    
    # Create formatted data tables for Gemini analysis
    daily_data_for_analysis = format_data_for_analysis(last_20_days, "Daily Price & Technical Data")
    weekly_data_for_analysis = format_data_for_analysis(last_20_weeks, "Weekly Price & Technical Data")
    
    # Create tables with last 5 days of data for both daily and weekly
    last_5_days = daily_data.tail(5)[['DATE', 'CLOSE', 'EMA_26', 'ROC_14', 'RSI_14']]
    last_5_weeks = weekly_data.tail(5)[['DATE', 'CLOSE', 'EMA_26', 'ROC_14', 'RSI_14']]
    
    # Create a version of the HTML table that's less likely to leak
    # by avoiding direct concatenation in f-strings
    table_html_parts = []
    
    # Add opening wrapper div
    table_html_parts.append('<div style="display: flex; justify-content: space-between;">')
    
    # Daily table - construct part by part
    table_html_parts.append('<div style="width: 48%; display: inline-block;">')
    table_html_parts.append('<table style="border-collapse: collapse; width: 100%; font-size: 7pt;">')
    table_html_parts.append('<thead><tr>')
    
    # Headers - separate each header to avoid DAILY and CLOSE leaking
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
    
    # Weekly table - construct part by part
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
    
    # Join all parts only when needed for the API call, not for printing
    table_section = ''.join(table_html_parts)
    
    # Convert both charts to base64 for Gemini API
    try:
        with open(daily_chart_path, "rb") as daily_file:
            daily_chart_base64 = base64.b64encode(daily_file.read()).decode('utf-8')
        with open(weekly_chart_path, "rb") as weekly_file:
            weekly_chart_base64 = base64.b64encode(weekly_file.read()).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Error reading chart images: {str(e)}")
        master_sheet["A11"].value = "Error"
        master_sheet["A12"].value = "Error reading chart images"
        return
    
    # Create a prompt for Gemini that includes both full-size charts
    # Note: Using triple quotes in an f-string can also lead to leaks, so we'll build this differently
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
    
    # Insert the table HTML without direct interpolation
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
    
    # Add the technical data without using f-strings
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
    
    # Join the prompt parts only when needed
    prompt = ''.join(prompt_parts)
    
    print("\nSending request to Gemini API...")
    
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
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    try:
        print("\nCalling Gemini API...")
        
        # When logging, don't print the actual payload content
        if DEBUG_HTML:
            print(f"API URL: {api_url}")
            print("Payload includes: 2 images and prompt text")
        
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            response_json = response.json()
            if 'candidates' in response_json:
                analysis = response_json['candidates'][0]['content']['parts'][0]['text']
                
                print("\nRaw Gemini Response (first 500 chars):")
                print("=" * 80)
                print(analysis[:500])
                print("=" * 80)
                
                # Add disclaimer
                disclaimer_note = """
                
                #### Important Disclaimer

The content in this report is for xlwings Lite demo purposes only and is not investment research, investment analysis, or financial advice. This is a technical demonstration of how to use xlwings Lite to send data and charts to an LLM/AI via a web API call, receive a markdown-formatted response, convert it to PDF, and display the downloadable link in a cell.

"""
                final_markdown = f"{disclaimer_note}{analysis}"
                
                # Convert to PDF and save URL
                try:
                    pdf_api_url = "https://mdtopdf.hosting.tigzig.com/text-input"
                    print("\nConverting markdown to PDF...")
                    print(f"Using combined image for PDF: {combined_image_path}")
                    
                    pdf_response = requests.post(
                        pdf_api_url,
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                        json={"text": final_markdown, "image_path": combined_image_path}
                    )
                    
                    print(f"Status Code: {pdf_response.status_code}")
                    # Don't print raw response text as it could contain HTML
                    if DEBUG_HTML:
                        print(f"Response Text: {pdf_response.text}")
                    else:
                        print("Response received. URLs processed.")
                    
                    response_data = pdf_response.json()
                    
                    # Clear only specific cells where we'll update content
                    master_sheet["A11"].clear_contents()  # PDF Report URL header
                    master_sheet["A12"].clear_contents()  # PDF URL
                    master_sheet["A16"].clear_contents()  # HTML Report URL header
                    master_sheet["A17"].clear_contents()  # HTML URL
                    
                    # Handle PDF URL - don't touch the Open Link cell
                    master_sheet["A11"].value = "PDF Report URL"
                    master_sheet["A12"].value = response_data["pdf_url"]
                    master_sheet["A12"].add_hyperlink(response_data["pdf_url"])
                    
                    # Handle HTML URL - don't touch the Open Link cell
                    master_sheet["A16"].value = "HTML Report URL"
                    master_sheet["A17"].value = response_data["html_url"]
                    master_sheet["A17"].add_hyperlink(response_data["html_url"])
                    
                    print("🟢 URLs saved successfully!")
                except Exception as e:
                    print(f"[ERROR] Error processing response: {str(e)}")
            else:
                master_sheet["A11"].value = "Error"
                master_sheet["A12"].value = "No analysis generated"
        else:
            master_sheet["A11"].value = "Error"
            master_sheet["A12"].value = f"API call failed: {response.status_code}"
    except Exception as e:
        master_sheet["A11"].value = "Error"
        master_sheet["A12"].value = f"Error: {str(e)}"
    
    print("🟢 ENDING get_technical_analysis_from_gemini")

def create_chart(sheet, df, ticker, title, frequency):
    """Create a chart in the specified sheet using the provided DataFrame."""
    print(f"\nCreating {frequency} chart in {sheet.name} sheet using matplotlib...")
    
    # Clear existing content
    sheet.clear()
    
    # Add a title to the sheet
    sheet["A1"].value = f"{title} - {frequency}"
    
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
    
    # Normalize volume to make it visible in both daily and weekly charts
    # Use a relative scale based on the price range
    price_range = df['CLOSE'].max() - df['CLOSE'].min()
    volume_scale_factor = price_range * 0.2 / df['VOLUME'].max()
    normalized_volume = df['VOLUME'] * volume_scale_factor
    
    # Plot volume bars with normalized height
    ax1v.bar(df['DATE'], normalized_volume, width=bar_width, color=volume_colors, alpha=0.3)
    
    # Set volume axis properties - remove value labels but keep "Volume" label
    ax1v.set_ylabel('Volume', fontsize=10, color='gray')
    ax1v.set_yticklabels([])  # Remove volume value labels
    ax1v.tick_params(axis='y', length=0)  # Remove volume tick marks
    
    # Make sure volume bars don't take up too much space
    ax1v.set_ylim(0, price_range * 0.3)  # Limit volume height to 30% of price range
    
    ax1.set_title(f"{ticker} - Price with EMAs and Bollinger Bands ({frequency})", fontsize=14, fontweight='bold', pad=10, loc='center')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_xticklabels([])
    
    # Plot on the second subplot (MACD)
    macd_hist = df['MACD_12_26'] - df['MACD_SIGNAL_9']
    colors = ['#26A69A' if val >= 0 else '#EF5350' for val in macd_hist]
    bar_width = (df['DATE'].iloc[-1] - df['DATE'].iloc[0]).days / len(df) * 0.8  # Adjusted bar width
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
    ax3.set_xticks([df['DATE'].iloc[i] for i in tick_indices])
    date_format = '%Y-%m-%d' if date_range.days > 30 else '%m-%d'
    tick_labels = [df['DATE'].iloc[i].strftime(date_format) for i in tick_indices]
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save compound figure to temporary file
    temp_dir = tempfile.gettempdir()
    chart_filename = f"{ticker}_{frequency.lower()}_technical_chart.png"
    temp_path = os.path.join(temp_dir, chart_filename)
    fig.savefig(temp_path, dpi=150, bbox_inches='tight')
    
    # Store the chart path in a cell for later use
    sheet["A1"].value = temp_path
    
    # Insert picture into Excel
    anchor_cell = sheet["A3"]
    sheet.pictures.add(temp_path, name=f"{ticker}_{frequency.lower()}_compound", update=True, anchor=anchor_cell, format="png")
    
    plt.close(fig)
    
    print(f"{frequency} chart created and inserted successfully in {sheet.name}")

def convert_excel_date(excel_date):
    """Convert Excel date to YYYY-MM-DD format."""
    if isinstance(excel_date, (datetime, pd.Timestamp)):
        return excel_date.strftime("%Y-%m-%d")
    else:
        # Convert Excel serial number to datetime
        date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(excel_date))
        return date.strftime("%Y-%m-%d")


def calculate_technical_indicators(df):
    """Calculate technical indicators for a DataFrame."""
    print("\nCalculating indicators...")
    
    # 1. EMA - Exponential Moving Average (12 days)
    print("Calculating EMA-12...")
    df['EMA_12'] = TA.EMA(df, 12)
    
    # 2. EMA - Exponential Moving Average (26 days)
    print("Calculating EMA-26...")
    df['EMA_26'] = TA.EMA(df, 26)
    
    # 3. RSI - Relative Strength Index (14 periods)
    print("Calculating RSI...")
    df['RSI_14'] = TA.RSI(df)
    
    # 4. ROC - Rate of Change (14 periods)
    print("Calculating ROC...")
    df['ROC_14'] = TA.ROC(df, 14)
    
    # 5. MACD - Moving Average Convergence Divergence (12/26)
    print("Calculating MACD...")
    macd = TA.MACD(df)  # Using default 12/26 periods
    if isinstance(macd, pd.DataFrame):
        df['MACD_12_26'] = macd['MACD']
        df['MACD_SIGNAL_9'] = macd['SIGNAL']
    
    # 6. Bollinger Bands (20 periods, 2 standard deviations)
    print("Calculating Bollinger Bands...")
    bb = TA.BBANDS(df)
    if isinstance(bb, pd.DataFrame):
        df['BBANDS_UPPER_20_2'] = bb['BB_UPPER']
        df['BBANDS_MIDDLE_20_2'] = bb['BB_MIDDLE']
        df['BBANDS_LOWER_20_2'] = bb['BB_LOWER']
    
    # 7. Stochastic Oscillator
    print("Calculating Stochastic Oscillator...")
    try:
        stoch = TA.STOCH(df)
        if isinstance(stoch, pd.DataFrame):
            df['STOCH_K'] = stoch['K']
            df['STOCH_D'] = stoch['D']
    except Exception as stoch_error:
        print(f"Stochastic calculation error: {str(stoch_error)}")
    
    # 8. Williams %R
    print("Calculating Williams %R...")
    df['WILLIAMS_R'] = TA.WILLIAMS(df)
