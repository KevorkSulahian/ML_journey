# libraries

from fastmcp import FastMCP

mcp = FastMCP(name = 'Calculator', 
              )

@mcp.tool()
def add(a: float, b: float) -> float:
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    if b == 0:
        return float('inf')
    return a / b


if __name__ == "__main__":
    mcp.run()