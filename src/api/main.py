"""
Digital Twin API - FastAPI Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path to import your modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.run_agent import DigitalTwin
from src.api.tts_service import CoquiTTSService

# Initialize services (only once at startup)
digital_twin = None
tts_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global digital_twin, tts_service
    print("Initializing Digital Twin...")
    digital_twin = DigitalTwin()
    print("Initializing TTS...")
    tts_service = CoquiTTSService(language="en_vctk", speaker="p225")
    print("✓ Ready!")
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    print("Shutting down...")

app = FastAPI(title="Digital Twin API", lifespan=lifespan)

# CORS - allow your GitHub Pages domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://yourusername.github.io",  # Replace with your GitHub Pages URL
        "*"  # Remove this in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    user_id: str = "anonymous"

class AskResponse(BaseModel):
    text: str
    audio_url: str

@app.post("/api/ask")
async def ask_question(request: AskRequest):
    """
    Ask a question - returns text answer and audio URL
    """
    try:
        # Pass user_id for tracking
        answer_text = digital_twin.ask(request.question, user_id=request.user_id)
        
        audio_bytes = tts_service.text_to_audio(answer_text)
        
        return {
            "text": answer_text,
            "audio": audio_bytes.hex()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}