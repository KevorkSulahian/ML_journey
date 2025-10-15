from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP

app = FastAPI(title="MCP Calculator API", version="1.0.0")

# ---- Your calculator endpoints (unchanged) ----

from fastapi import Response

@app.get("/mcp")
def mcp_get_not_supported():
    return Response(content="SSE not supported; use POST Streamable HTTP", status_code=405)



@app.post("/add")
def add(a: float, b: float) -> dict:
    return {"result": a + b}

@app.post("/subtract")
def subtract(a: float, b: float) -> dict:
    return {"result": a - b}

@app.post("/multiply")
def multiply(a: float, b: float) -> dict:
    return {"result": a * b}

@app.post("/divide")
def divide(a: float, b: float) -> dict:
    if b == 0:
        return {"result": float("inf")}
    return {"result": a / b}

# ---- Make the MCP endpoint and CORS-friendly ----
# The Inspector UI runs on http://localhost:6274 by default.
# Allow it (and 127.0.0.1 variant) to call your /mcp endpoint.
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:6274", "http://127.0.0.1:6274"],
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)


# Mount the MCP Streamable HTTP endpoint at /mcp.
# (If fastapi_mcp supports a path argument, it's usually default "/mcp" anyway.)
mcp = FastApiMCP(app, name="Calculator")
mcp.mount_http()  # if your version doesn't accept path=, just call mcp.mount_http()

# Handle browser preflight cleanly so the POST can proceed.
@app.options("/mcp")
async def mcp_options():
    # CORS middleware will inject the right headers; 204 is enough.
    return Response(status_code=204)

# Optional: if the SDK mounted a GET /mcp for SSE and you *don’t* implement SSE,
# you can be explicit (not required, but clarifies logs):
@app.get("/mcp")
async def mcp_get_not_supported():
    return Response(content="SSE not supported; use Streamable HTTP POST", status_code=405)

if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 (sometimes Windows treats 'localhost' differently)
    uvicorn.run(app, host="127.0.0.1", port=8001)
