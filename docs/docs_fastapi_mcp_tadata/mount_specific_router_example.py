from examples.shared.apps import items
from examples.shared.setup import setup_logging

from fastapi import APIRouter
from fastapi_mcp import FastApiMCP

setup_logging()


router = APIRouter(prefix="/other/route")
items.app.include_router(router)

mcp = FastApiMCP(
    items.app,
    name="Item API MCP",
    description="MCP server for the Item API",
    base_url="http://localhost:8000",
)

# Mount the MCP server to a specific router.
# It will now only be available at `/other/route/mcp`
mcp.mount(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(items.app, host="0.0.0.0", port=8000)
