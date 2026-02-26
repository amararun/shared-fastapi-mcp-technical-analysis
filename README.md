# Technical Analysis API with FastAPI, MCP, and Flask Frontend

A comprehensive technical analysis platform that combines FastAPI endpoints, Model Context Protocol (MCP) integration, and a Flask frontend. The API endpoints can be used independently by other applications, while the MCP integration enables seamless AI/LLM interactions.

## How It Works

1. **Data Collection**: Historical price data (daily and weekly) is fetched from Yahoo Finance
2. **Technical Analysis**: Calculates key technical indicators (EMAs, MACD, RSI, Bollinger Bands), generates charts for both timeframes, and performs pattern recognition
3. **AI Analysis**: Sends data and charts to an LLM (Google Gemini by default, or any model via OpenRouter) for comprehensive market interpretation
4. **Report Generation**: Converts analysis into professionally formatted PDF and HTML reports

## Endpoints

- **Frontend**: `/` - Flask-based web interface
- **API**: `/api/technical-analysis` - FastAPI endpoint for direct programmatic access
- **MCP**: `/mcp` - Model Context Protocol endpoint for AI/LLM integration

## Security Hardening

This application has been hardened with the following security measures:

- **Rate Limiting**: 15 requests/hour per client IP + 100 requests/minute global (configurable via env vars)
- **Concurrency Limits**: 2 concurrent requests per IP + 4 global (configurable via env vars)
- **Error Sanitization**: All error responses return generic messages. Full error details are logged server-side only — never exposed to clients.
- **Global Exception Handler**: Catches unhandled exceptions as a safety net, returns generic 500 response
- **Cloudflare IP Extraction**: Extracts real client IP from `cf-connecting-ip`, `x-forwarded-for`, and other proxy headers for accurate rate limiting behind reverse proxies
- **CORS**: Configurable allowed origins
- **Input Validation**: Pydantic models with type hints and validation rules on all endpoints

## Configuration

Create a `.env` file in the root directory. See `.envExample` for all available options:

```env
# Required: at least one of these
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional: model selection (defaults shown)
GEMINI_MODEL_NAME=gemini-1.5-flash-latest
OPENROUTER_MODEL=google/gemini-2.5-flash-preview-09-2025

# Optional: rate limiting and concurrency (defaults shown)
RATE_LIMIT=15/hour
GLOBAL_RATE_LIMIT=100/minute
MAX_CONCURRENT_PER_IP=2
MAX_CONCURRENT_GLOBAL=4
```

## Installation

```bash
git clone https://github.com/amararun/shared-fastapi-mcp-technical-analysis.git
cd shared-fastapi-mcp-technical-analysis
pip install -r requirements.txt
```

Set up environment variables, then run:
```bash
uvicorn main:app --reload
```

## API Usage

The API can be accessed in three ways:

1. **Web Interface**: Visit `/` for the Flask frontend
2. **Direct API**: Make HTTP requests to `/api/technical-analysis`
3. **MCP/LLM**: Connect to `/mcp` using any MCP-compatible client (e.g., Cursor, Claude)

## Example Request

```
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
    "pdf_url": "https://example.com/reports/analysis_123.pdf",
    "html_url": "https://example.com/reports/analysis_123.html"
}
```

## MCP Integration

The FastAPI server is enhanced with MCP capabilities using the `fastapi-mcp` package:

```python
mcp = FastApiMCP(
    app,
    name="Technical Analysis API",
    description="API for generating technical analysis reports",
    base_url="http://localhost:8000"
)
mcp.mount()
```

## Statcounter Note

The application includes a Statcounter web analytics code in `templates/index.html`. This tracking code is linked to the author's personal account. Please replace it with your own Statcounter ID or other analytics tracking code, or remove it entirely if you don't need web analytics.

## API Monitoring

This API uses [tigzig-api-monitor](https://pypi.org/project/tigzig-api-monitor/), an open-source centralized logging middleware for FastAPI. The middleware captures request metadata including client IP addresses and request bodies for API monitoring and error tracking.

**Data Capture**: The middleware captures client IP, request path, status codes, response times, and request bodies. This data is sent to a configurable logging endpoint.

**Data Retention**: The middleware captures data but does not manage its lifecycle. It is the deployer's responsibility to implement appropriate data retention and deletion policies in accordance with their own compliance requirements (GDPR, CCPA, etc.).

**Graceful Degradation**: If the logging service is unavailable, API calls proceed normally — logging fails silently without affecting functionality.

**Self-Hosting**: The package is available on [PyPI](https://pypi.org/project/tigzig-api-monitor/). To self-host, configure your own database endpoint.

## Dependencies

- FastAPI + fastapi-mcp
- Flask (frontend)
- pandas, matplotlib, finta (technical indicators)
- Google Generative AI / OpenRouter
- slowapi (rate limiting)
- tigzig-api-monitor (API logging)
- Other requirements listed in requirements.txt

## License

MIT License

## Author

Built by [Amar Harolikar](https://www.linkedin.com/in/amarharolikar/)

Explore 30+ open source AI tools for analytics, databases & automation at [tigzig.com](https://tigzig.com)
