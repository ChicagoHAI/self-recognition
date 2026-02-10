import os
import time
import requests
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig

load_dotenv()

# Set up detailed logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        self.default_models = [
            ModelConfig(name="moonshotai/kimi-k2:free", display_name="Kimi K2"),
            ModelConfig(name="z-ai/glm-4.5-air:free", display_name="GLM-4.5-Air"),
        ]
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.ConnectionError))
    )
    def generate_text(self, model: str, prompt: str, max_tokens: int = 150, temperature: float = 0.7, reasoning_effort: str = "low") -> Dict[str, str]:
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning": {
                "effort": reasoning_effort
            },
        }
        
        logger.info(f"🚀 Starting generation for model: {model}")
        logger.info(f"📝 Prompt: {prompt[:100]}...")
        logger.info(f"🔢 Max tokens requested: {max_tokens}")
        logger.info(f"🌡️ Temperature: {payload['temperature']}")
        
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            logger.info(f"📡 API response status: {response.status_code}")
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                retry_after = int(response.headers.get('retry-after', 60))
                logger.warning(f"⏳ Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                # Make the request again after waiting
                response = requests.post(url, json=payload, headers=self.headers)
                logger.info(f"🔄 Retry API response status: {response.status_code}")
            
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"🔍 API response keys: {list(data.keys())}")
            
            # Log usage information if available
            if "usage" in data:
                usage = data["usage"]
                logger.info(f"📊 Token usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                           f"Completion: {usage.get('completion_tokens', 'N/A')}, "
                           f"Total: {usage.get('total_tokens', 'N/A')}")
            
            if "choices" not in data or not data["choices"]:
                logger.error(f"❌ No choices in API response for model {model}")
                logger.error(f"🔍 Full response: {data}")
                raise ValueError(f"No choices in API response for model {model}: {data}")
            
            choice = data["choices"][0]
            logger.info(f"🎯 Choice details: {list(choice.keys())}")
            
            # Log finish reason (this is crucial for understanding why generation stops)
            finish_reason = choice.get("finish_reason")
            logger.info(f"🏁 Finish reason: {finish_reason}")
            
            if finish_reason == "length":
                logger.warning(f"⚠️ Generation stopped due to token limit ({max_tokens} tokens)")
            elif finish_reason == "stop":
                logger.info(f"✅ Generation completed normally (stop token)")
            elif finish_reason == "content_filter":
                logger.warning(f"🚫 Generation stopped due to content filtering")
            elif finish_reason:
                logger.warning(f"❓ Unexpected finish reason: {finish_reason}")
            print(choice)
            content = choice["message"]["content"]
            reasoning = choice["message"].get("reasoning", "")  # Extract reasoning if present
            
            logger.info(f"📏 Raw content length: {len(content) if content else 0} characters")
            if reasoning:
                logger.info(f"🧠 Reasoning length: {len(reasoning)} characters")
            
            if content is None:
                logger.error(f"❌ Content is None for model {model}")
                logger.error(f"🔍 Choice data: {choice}")
                raise ValueError(f"Content is None for model {model}")
            
            result_text = content.strip()
            result_reasoning = reasoning.strip() if reasoning else ""
            
            word_count = len(result_text.split()) if result_text else 0
            logger.info(f"📝 Final text length: {len(result_text)} characters, {word_count} words")
            logger.info(f"🔤 Text preview: {result_text[:150]}...")
            
            if result_reasoning:
                reasoning_word_count = len(result_reasoning.split())
                logger.info(f"🧠 Reasoning preview: {result_reasoning[:100]}...")
                logger.info(f"🧠 Reasoning word count: {reasoning_word_count}")
            
            # Log information about word count patterns
            if word_count == 1:
                logger.info(f"📝 Single word generated: '{result_text}'")
            elif word_count < 20:
                logger.info(f"📏 Short generation: {word_count} words (typical target: ~100)")
            elif word_count < 50:
                logger.info(f"📊 Moderate length: {word_count} words (typical target: ~100)")
            
            return {
                "content": result_text,
                "reasoning": result_reasoning
            }
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"🚨 HTTP error for model {model}: {e}")
            logger.error(f"📡 Response status: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
            
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    logger.error(f"📄 Error response data: {error_data}")
                    
                    # Log specific error details that might explain generation issues
                    if 'error' in error_data:
                        error_info = error_data['error']
                        if isinstance(error_info, dict):
                            error_type = error_info.get('type', 'unknown')
                            error_code = error_info.get('code', 'unknown')
                            error_message = error_info.get('message', 'No message')
                            logger.error(f"🔍 Error details - Type: {error_type}, Code: {error_code}")
                            logger.error(f"💬 Error message: {error_message}")
                        
                except:
                    logger.error(f"📄 Error response text: {e.response.text[:500]}")
                    
                # Log headers that might contain useful debugging info
                logger.error(f"📋 Response headers: {dict(e.response.headers)}")
            raise
        except Exception as e:
            logger.error(f"💥 Unexpected error for model {model}: {type(e).__name__}: {e}")
            logger.error(f"🔍 Exception details: {str(e)}")
            raise
    
    async def generate_text_async(self, model: str, prompt: str, max_tokens: int = 150, temperature: float = 0.7, reasoning_effort: str = "low") -> Dict[str, str]:
        """Async version of generate_text for concurrent API calls."""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning": {
                "effort": reasoning_effort
            },
        }
        
        logger.info(f"🚀 Starting async generation for model: {model}")
        logger.info(f"📝 Prompt: {prompt[:100]}...")
        logger.info(f"🔢 Max tokens requested: {max_tokens}")
        logger.info(f"🌡️ Temperature: {payload['temperature']}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    logger.info(f"📡 API response status: {response.status}")
                    
                    # Handle rate limiting specifically
                    if response.status == 429:
                        retry_after = int(response.headers.get('retry-after', 60))
                        logger.warning(f"⏳ Rate limit hit. Waiting {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        # Make the request again after waiting
                        async with session.post(url, json=payload, headers=self.headers) as retry_response:
                            logger.info(f"🔄 Retry API response status: {retry_response.status}")
                            response = retry_response
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"🚨 HTTP error {response.status}: {error_text}")
                        raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status)
                    
                    data = await response.json()
                    logger.info(f"🔍 API response keys: {list(data.keys())}")
                    
                    # Log usage information if available
                    if "usage" in data:
                        usage = data["usage"]
                        logger.info(f"📊 Token usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                                   f"Completion: {usage.get('completion_tokens', 'N/A')}, "
                                   f"Total: {usage.get('total_tokens', 'N/A')}")
                    
                    if "choices" not in data or not data["choices"]:
                        logger.error(f"❌ No choices in API response for model {model}")
                        logger.error(f"🔍 Full response: {data}")
                        raise ValueError(f"No choices in API response for model {model}: {data}")
                    
                    choice = data["choices"][0]
                    logger.info(f"🎯 Choice details: {list(choice.keys())}")
                    
                    # Log finish reason (this is crucial for understanding why generation stops)
                    finish_reason = choice.get("finish_reason")
                    logger.info(f"🏁 Finish reason: {finish_reason}")
                    
                    if finish_reason == "length":
                        logger.warning(f"⚠️ Generation stopped due to token limit ({max_tokens} tokens)")
                    elif finish_reason == "stop":
                        logger.info(f"✅ Generation completed normally (stop token)")
                    elif finish_reason == "content_filter":
                        logger.warning(f"🚫 Generation stopped due to content filtering")
                    elif finish_reason:
                        logger.warning(f"❓ Unexpected finish reason: {finish_reason}")
                    
                    content = choice["message"]["content"]
                    reasoning = choice["message"].get("reasoning", "")  # Extract reasoning if present
                    
                    logger.info(f"📏 Raw content length: {len(content) if content else 0} characters")
                    if reasoning:
                        logger.info(f"🧠 Reasoning length: {len(reasoning)} characters")
                    
                    if content is None:
                        logger.error(f"❌ Content is None for model {model}")
                        logger.error(f"🔍 Choice data: {choice}")
                        raise ValueError(f"Content is None for model {model}")
                    
                    result_text = content.strip()
                    result_reasoning = reasoning.strip() if reasoning else ""
                    
                    word_count = len(result_text.split()) if result_text else 0
                    logger.info(f"📝 Final text length: {len(result_text)} characters, {word_count} words")
                    logger.info(f"🔤 Text preview: {result_text[:150]}...")
                    
                    if result_reasoning:
                        reasoning_word_count = len(result_reasoning.split())
                        logger.info(f"🧠 Reasoning preview: {result_reasoning[:100]}...")
                        logger.info(f"🧠 Reasoning word count: {reasoning_word_count}")
                    
                    # Log information about word count patterns
                    if word_count == 1:
                        logger.info(f"📝 Single word generated: '{result_text}'")
                    elif word_count < 20:
                        logger.info(f"📏 Short generation: {word_count} words (typical target: ~100)")
                    elif word_count < 50:
                        logger.info(f"📊 Moderate length: {word_count} words (typical target: ~100)")
                    
                    return {
                        "content": result_text,
                        "reasoning": result_reasoning
                    }
                        
        except Exception as e:
            logger.error(f"💥 Unexpected async error for model {model}: {type(e).__name__}: {e}")
            logger.error(f"🔍 Exception details: {str(e)}")
            raise
    
    def get_available_models(self) -> List[ModelConfig]:
        return self.default_models