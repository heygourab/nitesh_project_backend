from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.spam_detection import app as spam_detection_app
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the spam detection routes
app.mount("/api", spam_detection_app)

@app.get("/hello")
async def root():
    return {"message": "Welcome to the Spam Detection API!"}
# This is the main entry point for the FastAPI application.
# It initializes the app, sets up CORS middleware, and mounts the spam detection routes.

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)