from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv
import logging
import json
from datetime import datetime
from httpx import TimeoutException
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_request_response(request_data, response_data, endpoint):
    """Helper function to log request and response data"""
    try:
        # Log request
        logger.info(f"\n{'='*50}\n{endpoint} - REQUEST:\n{json.dumps(request_data, indent=2)}\n{'='*50}")
        
        # Log response
        logger.info(f"\n{'='*50}\n{endpoint} - RESPONSE:\n{json.dumps(response_data, indent=2)}\n{'='*50}")
    except Exception as e:
        logger.error(f"Error logging request/response: {str(e)}")

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware with permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tigzig.com",
        "https://realtime.tigzig.com",
        "https://rex.tigzig.com",
        "https://rexdb.tigzig.com",
        "https://rexrc.tigzig.com",
        "https://mf.tigzig.com",
        "https://rexdb2.tigzig.com",
        "https://n8n-iii2.onrender.com",
        "http://localhost:8123",   # For local development
        "http://localhost:5199"   # For local development

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/session")
async def get_session(
    model: str = Query(..., description="The model to use"),
    voice: str = Query(..., description="The voice to use")
):
    request_data = {"model": model, "voice": voice}
    logger.info(f"Received session request - Model: {model}, Voice: {voice}")
    try:
        async with httpx.AsyncClient() as client:
            logger.debug("Making request to OpenAI realtime sessions API")
            response = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "voice": voice,
                    "temperature": 0.6,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.7
                    }
                }
            )
            response_data = response.json()
            log_request_response(request_data, response_data, "SESSION")
            logger.info(f"Session created successfully with status code: {response.status_code}")
            return response_data
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: {"error": "Request failed after multiple retries"}
)
async def make_api_request(url: str, headers: dict, json_data: dict, timeout: int = 30):
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(url, headers=headers, json=json_data)

async def try_openrouter_fallback(request_body: dict):
    """Helper function to try the OpenRouter fallback"""
    logger.warning("Falling back to OpenRouter")
    
    # Create OpenRouter request
    openrouter_request = request_body.copy()
    fallback_model = os.getenv('OPENAI_FALLBACK')
    if fallback_model:
        openrouter_request["model"] = fallback_model
    
    # Make request to OpenRouter
    async with httpx.AsyncClient() as client:
        openrouter_response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json=openrouter_request,
            timeout=60
        )
        
        openrouter_data = openrouter_response.json()
        log_request_response(openrouter_request, openrouter_data, "OPENROUTER_FALLBACK")
        logger.info(f"Fallback to OpenRouter successful with status code: {openrouter_response.status_code}")
        return openrouter_data

@app.post("/open-chat-completion")
async def proxy_chat_completion(request_body: dict):
    logger.info("Received OpenAI chat completion request")
    try:
        logger.debug(f"Making request to OpenAI chat completions API with model: {request_body.get('model', 'unknown')}")
        response = await make_api_request(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json_data=request_body,
            timeout=60  # Increased timeout for chat completions
        )
        
        # Check if we got a successful response
        if response.status_code >= 200 and response.status_code < 300:
            response_data = response.json()
            log_request_response(request_body, response_data, "OPENAI_CHAT")
            logger.info(f"Chat completion successful with status code: {response.status_code}")
            return response_data
        else:
            # OpenAI request failed, fallback to OpenRouter
            logger.warning(f"OpenAI request failed with status code: {response.status_code}.")
            return await try_openrouter_fallback(request_body)
                
    except TimeoutException as e:
        error_msg = "Request timed out while waiting for OpenAI response"
        logger.error(f"{error_msg}: {str(e)}")
        
        # Try fallback on timeout too
        try:
            return await try_openrouter_fallback(request_body)
        except Exception as fallback_error:
            logger.error(f"Fallback to OpenRouter also failed: {str(fallback_error)}")
            return {"error": error_msg}
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        
        # Try fallback on any other exception
        try:
            logger.warning(f"OpenAI request failed with error: {str(e)}.")
            return await try_openrouter_fallback(request_body)
        except Exception as fallback_error:
            logger.error(f"Fallback to OpenRouter also failed: {str(fallback_error)}")
            raise

@app.post("/open-router-completion")
async def open_router_completion(request_body: dict):
    logger.info("Received OpenRouter completion request")
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Making request to OpenRouter API with model: {request_body.get('model', 'unknown')}")
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json=request_body
            )
            response_data = response.json()
            log_request_response(request_body, response_data, "OPENROUTER")
            logger.info(f"OpenRouter completion successful with status code: {response.status_code}")
            return response_data
    except Exception as e:
        logger.error(f"Error in OpenRouter completion: {str(e)}")
        raise

@app.post("/gemini-chat-completion")
async def gemini_chat_completion(request_body: dict):
    logger.info("Received Gemini chat completion request")
    try:
        async with httpx.AsyncClient() as client:
            API_KEY = os.getenv('GEMINI_API_KEY')
            model = request_body.get("model", "gemini-1.5-flash")
            logger.debug(f"Making request to Gemini API with model: {model}")
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}",
                headers={
                    "Content-Type": "application/json"
                },
                json=request_body
            )
            response_data = response.json()
            log_request_response(request_body, response_data, "GEMINI")
            logger.info(f"Gemini completion successful with status code: {response.status_code}")
            return response_data
    except Exception as e:
        logger.error(f"Error in Gemini completion: {str(e)}")
        raise

# To run the app, use:
# uvicorn app:app --reload

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))