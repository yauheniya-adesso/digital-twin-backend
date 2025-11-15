"""
TTS Service for Digital Twin
"""
from TTS.api import TTS
import io
import wave
import tempfile
from pathlib import Path

class CoquiTTSService:
    """Wrapper for Coqui TTS to generate audio bytes"""
    
    MODELS = {
        "en_vctk": "tts_models/en/vctk/vits",
        "en_jenny": "tts_models/en/jenny/jenny",
    }
    
    def __init__(self, language: str = "en_vctk", speaker: str = "p225", gpu: bool = False):
        self.language = language
        self.speaker = speaker
        self.model_path = self.MODELS[language]
        
        print(f"Loading TTS model: {self.model_path}...")
        self.tts = TTS(model_name=self.model_path, gpu=gpu)
        print("✓ TTS model loaded")
    
    def text_to_audio(self, text: str) -> bytes:
        """
        Convert text to audio bytes
        Returns: WAV audio as bytes
        """
        # Generate to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Generate audio
            self.tts.tts_to_file(
                text=text,
                file_path=tmp_path,
                speaker=self.speaker
            )
            
            # Read file as bytes
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            
            return audio_bytes
        
        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
