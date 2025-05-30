# Technical Analysis API with FastAPI, MCP, and Flask Frontend

A comprehensive technical analysis platform that combines FastAPI endpoints, Model Context Protocol (MCP) integration, and a Flask frontend. The API endpoints can be used independently by other applications, while the MCP integration enables seamless AI/LLM interactions.

### Statcounter Note
The application includes a Statcounter web analytics code patch in `index.html`. This tracking code is linked to my personal account, so all analytics data will be sent there. Please replace it with your own Statcounter ID or other analytics tracking code, or remove it entirely if you don't need web analytics.

## How It Works

1. **Data Collection**: Historical price data (daily and weekly) is fetched from Yahoo Finance via a custom FastAPI server (yfin.hosting.com)
2. **Technical Analysis**: 
   - Calculates key technical indicators (EMAs, MACD, RSI, Bollinger Bands)
   - Generates custom charts for both timeframes
   - Performs pattern recognition
3. **AI Analysis**: Sends data and charts to Google's Gemini AI for comprehensive market interpretation
4. **Report Generation**: Converts analysis into professionally formatted PDF and HTML reports via mdtopdf.hosting.com

## Endpoints

The application exposes three types of endpoints:

- **Frontend**: `/` - Flask-based web interface
- **API**: `/api/*` - FastAPI endpoints for direct programmatic access
- **MCP**: `/mcp` - Model Context Protocol endpoint for AI/LLM integration

## FastAPI Implementation

The FastAPI server is enhanced with MCP capabilities using the `fastapi-mcp` package. Key implementation features:

1. **MCP Integration**:
```python
mcp = FastApiMCP(
    app,
    name="Technical Analysis API",
    description="API for generating technical analysis reports",
    base_url="http://localhost:8000"
)
mcp.mount()
```

2. **Enhanced Documentation**:
- Detailed docstrings for each endpoint
- Parameter descriptions using FastAPI's Field/Query
- Request/response model examples
- Operation IDs for MCP tool identification

3. **Structured Models**:
- Pydantic models for request/response validation
- Clear parameter descriptions and examples
- Type hints and validation rules

## Configuration Required

Create a `.env` file in the root directory with:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-1.5-flash-latest
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up environment variables
4. Run the server:
```bash
uvicorn main:app --reload
```

## API Usage

The API can be accessed in three ways:

1. **Web Interface**: Visit `/` for the Flask frontend
2. **Direct API**: Make HTTP requests to `/api/technical-analysis`
3. **MCP/LLM**: Connect to `/mcp` using any MCP-compatible client (e.g., Cursor, Claude)

## Example Request

```python
POST /api/technical-analysis

{
    "ticker": "AAPL",
    "daily_start_date": "2023-07-01",
    "daily_end_date": "2023-12-31",
    "weekly_start_date": "2022-01-01",
    "weekly_end_date": "2023-12-31"
}
```

## Response

The API returns URLs for both PDF and HTML versions of the analysis:

```json
{
    "pdf_url": "https://mdtopdf.hosting.com/reports/analysis_123.pdf",
    "html_url": "https://mdtopdf.hosting.com/reports/analysis_123.html"
}
```

## Dependencies

- FastAPI
- fastapi-mcp
- Flask
- yfinance
- pandas
- matplotlib
- Google Generative AI
- Other requirements listed in requirements.txt

## License

MIT License 