# FastAPI-MCP Base URL Fix Documentation

## Issue
When using FastAPI-MCP, you might encounter the following error:
```
httpcore.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol
```

This error occurs because the MCP server is trying to make requests without proper URL protocols.

## Solution That Worked

### 1. Base URL Configuration
First, define a base URL that includes the protocol:

```python
base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
```

### 2. Updated MCP Configuration
Configure the FastAPI-MCP with an `httpx.AsyncClient` that includes the base URL:

```python
mcp = FastApiMCP(
    app,
    name="Technical Analysis MCP API",
    description="MCP server for technical analysis endpoints",
    include_operations=[
        "create_technical_analysis"
    ],
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Inject custom httpx client with base URL
    http_client=httpx.AsyncClient(
        timeout=300.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        base_url=base_url
    )
)
```

### 3. FastAPI App Configuration
Make sure your FastAPI app has proper server configuration:

```python
app = FastAPI(
    title="Your API Title",
    description="Your API Description",
    version="1.0.0",
    root_path=os.environ.get("API_ROOT_PATH", ""),
    servers=[
        {"url": os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")}
    ]
)
```

## Important Notes

1. **No base_url Parameter**: FastAPI-MCP 0.3.0 and above removed the `base_url` parameter. Instead, use the `http_client` with `base_url` as shown above.

2. **Environment Variables**: 
   - Use `RENDER_EXTERNAL_URL` for production deployments
   - Fallback to `http://localhost:8000` for local development

3. **CORS Configuration**: Ensure proper CORS configuration if needed:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
```

## Troubleshooting

If you still encounter URL-related issues:

1. Check that your `base_url` includes the protocol (`http://` or `https://`)
2. Verify that environment variables are properly set
3. Ensure the MCP server is properly mounted:
```python
mcp.mount()
```

## Version Compatibility

This solution works with:
- FastAPI-MCP >= 0.3.0
- httpx >= 0.24.0
- FastAPI >= 0.68.0

## Additional Resources

For more information, refer to:
- [FastAPI-MCP Documentation](https://fastapi-mcp.readthedocs.io/)
- [httpx Documentation](https://www.python-httpx.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
