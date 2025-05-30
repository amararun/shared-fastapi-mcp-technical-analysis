from examples.shared.apps import items
from examples.shared.setup import setup_logging

from fastapi_mcp import FastApiMCP

setup_logging()


# Add MCP server to the FastAPI app
mcp = FastApiMCP(
    items.app,
    name="Item API MCP",
    description="MCP server for the Item API",
    base_url="http://localhost:8000",
)

mcp.mount()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(items.app, host="0.0.0.0", port=8000)
