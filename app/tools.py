import json
import subprocess
import tempfile
import requests

def tool_web_search(query: str) -> str:
    """Fake/stub search tool (replace with real API like SerpAPI, Tavily, or DuckDuckGo)."""
    return f"Search results for '{query}': [stubbed result 1, stubbed result 2]"

def tool_code_runner(code: str, lang: str = "python") -> str:
    """Run a small Python snippet safely in subprocess."""
    if lang != "python":
        return "Only Python supported in demo."
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            fname = f.name
        result = subprocess.run(["python", fname], capture_output=True, text=True, timeout=5)
        return result.stdout if result.stdout else result.stderr
    except Exception as e:
        return f"Error: {e}"

def tool_data_converter(data: str, target_format: str = "json") -> str:
    """Convert plain text/CSV to JSON (demo)."""
    try:
        if target_format == "json":
            lines = data.strip().splitlines()
            return json.dumps({"lines": lines}, indent=2)
        elif target_format == "csv":
            return ",".join(data.strip().split())
        else:
            return f"Unsupported format: {target_format}"
    except Exception as e:
        return f"Conversion failed: {e}"

TOOLS = {
    "web_search": tool_web_search,
    "code_runner": tool_code_runner,
    "data_converter": tool_data_converter,
}
