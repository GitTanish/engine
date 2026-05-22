from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
import os

app = FastAPI(title="SimAstra Visualizer UI")

@app.get("/")
def serve_visualizer():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "drone_visualizer.html")
    if not os.path.exists(html_path):
        return {"error": "drone_visualizer.html not found in tests/ directory"}
    return FileResponse(html_path)

if __name__ == "__main__":
    print("[INFO] Starting FastAPI frontend server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
