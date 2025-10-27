from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Look Ma, I'm deployed!"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)