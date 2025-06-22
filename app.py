import os
import re
import logging
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Added for CORS
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the request model for the POST endpoint
class QuestionRequest(BaseModel):
    question: str
    prompt_type: Optional[str] = "instruct"  # Optional prompt type, default to 'instruct'

# Initialize FastAPI app
app = FastAPI(
    title="Fitness Chatbot API",
    description="A FastAPI-based API for a fitness Q&A chatbot using a fine-tuned T5 model.",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:5173/",  # Vite development server
    "http://127.0.0.1:5173",  # Alternative for localhost
    "http://localhost:8000",  # Allow same-origin requests
    "http://127.0.0.1:8000",  # Allow same-origin requests
    "http://localhost",       # Allow broad localhost access
    "http://127.0.0.1",       # Allow broad localhost access
    "*"                      # Allow all origins (use cautiously in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# FitnessChatbot class
class FitnessChatbot:
    FITNESS_KEYWORDS = [
        'exercise', 'workout', 'fitness', 'nutrition', 'muscle', 'cardio', 'strength',
        'yoga', 'running', 'sleep', 'stress', 'recovery', 'flexibility', 'balance',
        'posture', 'hydration', 'motivation', 'diet', 'weight', 'training', 'gym',
        'health', 'stretch', 'protein', 'calorie', 'endurance', 'aerobic', 'anaerobic'
    ]

    def __init__(self, model_path: str, max_length: int = 512):
        self.max_length = max_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.tokenizer = None
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path {model_path} does not exist.")
            logger.info(f"Loading model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.model.eval()
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    def is_fitness_related(self, question: str) -> bool:
        question = self.clean_text(question)
        if not question or len(question.split()) < 3:
            return False
        pattern = '|'.join(self.FITNESS_KEYWORDS)
        return bool(re.search(pattern, question, re.IGNORECASE))

    def format_prompt(self, question: str, prompt_type: str = "instruct") -> str:
        if prompt_type == "instruct":
            return f"Answer this fitness-related question with specific advice: {question}"
        elif prompt_type == "qa":
            return f"question: {question}\nanswer:"
        return question

    def generate_response(self, question: str, prompt_type: str = "instruct") -> str:
        try:
            question = self.clean_text(question)
            if not question or len(question.split()) < 3:
                return "Please ask a valid question with at least a few words."
            if not self.is_fitness_related(question):
                return "Sorry, I can only answer fitness-related questions. Please ask about exercise, nutrition, or health!"
            prompt = self.format_prompt(question, prompt_type)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if len(response.split()) < 5 or "sorry" in response.lower():
                return "I couldn't provide a detailed answer. Try rephrasing or asking something like 'What are good protein sources for muscle gain?'"
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "An error occurred while generating the response. Please try again."

# Initialize the chatbot with the corrected model path
MODEL_PATH = "C:\\Users\\jules\\Desktop\\apifiteness\\fitness_qa_model_best_model_1"  # Corrected path from screenshot
try:
    chatbot = FitnessChatbot(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {e}")
    sys.exit(1)

# Define the POST endpoint for asking questions
@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    """
    Endpoint to ask a fitness-related question and get a response from the chatbot.
    Args:
        request: JSON payload with 'question' (required) and 'prompt_type' (optional, default='instruct').
    Returns:
        dict: Response containing the answer or an error message.
    """
    try:
        response = chatbot.generate_response(request.question, request.prompt_type)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Endpoint to check if the API is running and the model is loaded.
    """
    return {"status": "healthy", "model_loaded": chatbot.model is not None}

if __name__ == "__main__":
    # Run the FastAPI server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=True)