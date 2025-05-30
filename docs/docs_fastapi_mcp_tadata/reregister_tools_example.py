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


# MCP server
mcp.mount()


# This endpoint will not be registered as a tool, since it was added after the MCP instance was created
@items.app.get("/new/endpoint/", operation_id="new_endpoint", response_model=dict[str, str])
async def new_endpoint():
    return {"message": "Hello, world!"}


# But if you re-run the setup, the new endpoints will now be exposed.
mcp.setup_server()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(items.app, host="0.0.0.0", port=8000)
