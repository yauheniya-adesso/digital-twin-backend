"""
Digital Twin API - FastAPI Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

app = FastAPI(title="Digital Twin API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization - services load on first request
_digital_twin = None
_tts_service = None


def get_digital_twin():
    """Initialize Digital Twin on first use"""
    global _digital_twin
    if _digital_twin is None:
        print("Initializing Digital Twin...")
        from src.run_agent import DigitalTwin
        _digital_twin = DigitalTwin()
        print("✓ Digital Twin ready")
    return _digital_twin


def get_tts_service():
    """Initialize TTS on first use"""
    global _tts_service
    if _tts_service is None:
        print("Initializing TTS...")
        from src.api.tts_service import CoquiTTSService
        _tts_service = CoquiTTSService(language="en_vctk", speaker="p225")
        print("✓ TTS ready")
    return _tts_service


class AskRequest(BaseModel):
    question: str
    user_id: str = "anonymous"


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Digital Twin API", "status": "online"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/ask")
async def ask_question(request: AskRequest):
    """
    Ask a question - returns text answer and audio
    """
    try:
        # Get services (lazy initialization)
        digital_twin = get_digital_twin()
        tts_service = get_tts_service()
        
        # Get answer
        answer_text = digital_twin.ask(request.question, user_id=request.user_id)
        
        # Generate audio
        audio_bytes = tts_service.text_to_audio(answer_text)
        
        return {
            "text": answer_text,
            "audio": audio_bytes.hex()
        }
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))