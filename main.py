from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
import aiohttp
import asyncio
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import uuid
from datetime import datetime, UTC
import logging
from logging.handlers import RotatingFileHandler
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('api.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

app = FastAPI(title="URL Classification API",
             description="API for classifying URLs based on their content type",
             version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "75"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))

# In-memory storage for job status and results
jobs_store: Dict[str, Dict] = {}
sem = Semaphore(MAX_CONCURRENT_REQUESTS)

class URLInput(BaseModel):
    urls: List[str]
    callback_url: Optional[str] = None

class ClassificationResult(BaseModel):
    job_id: str
    status: str
    total_urls: int
    processed_urls: int = 0
    results: Optional[Dict] = None
    error: Optional[str] = None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def classify_urls_batch(session: aiohttp.ClientSession, urls: list) -> list:
    """Classify a batch of URLs with retry logic"""
    async with sem:
        try:
            start_time = time.time()
            logger.info(f"Starting classification for batch of {len(urls)} URLs")
            
            urls_text = "\n".join(urls)
            prompt = f"""Classify these URLs based on your knowledge:
{urls_text}

STRICT OUTPUT FORMAT:
- Return exactly one classification per line
- Use ONLY these categories: brand, affiliate, news, other
- brand = gambling operators/betting sites/casinos
- affiliate = gambling review/comparison/tips sites
- news = news media sites
- other = none of the above categories"""

            payload = {
                "model": "gpt-4-turbo-preview",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise URL classifier specializing in gambling-related websites."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0,
                "max_tokens": 200,
                "response_format": {"type": "text"}
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

            async with session.post(
                OPENAI_URL,
                json=payload,
                headers=headers,
                timeout=TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "choices" in data and data["choices"]:
                    classifications = data["choices"][0]["message"]["content"].strip().split("\n")
                    
                    # Validate output length matches input length
                    if len(classifications) != len(urls):
                        logger.warning(f"Mismatch in classification count. URLs: {len(urls)}, Classifications: {len(classifications)}")
                        # Pad or truncate classifications to match URL count
                        classifications = classifications[:len(urls)] if len(classifications) > len(urls) else classifications + ["other"] * (len(urls) - len(classifications))
                    
                    results = [{"url": url, "category": cat.lower()} 
                             for url, cat in zip(urls, classifications)]
                    
                    end_time = time.time()
                    logger.info(f"Batch classification completed in {end_time - start_time:.2f} seconds")
                    return results
                
                raise ValueError("Invalid response format from OpenAI API")

        except Exception as e:
            logger.error(f"Error in batch classification: {str(e)}")
            raise

async def process_urls(urls: List[str], job_id: str):
    """Process URLs in batches and update job status"""
    try:
        jobs_store[job_id]["status"] = "processing"
        results = {}
        failed_urls = []
        
        async with aiohttp.ClientSession() as session:
            batches = [urls[i:i + BATCH_SIZE] for i in range(0, len(urls), BATCH_SIZE)]
            
            for batch_index, batch in enumerate(batches):
                try:
                    logger.info(f"Processing batch {batch_index + 1}/{len(batches)} for job {job_id}")
                    batch_results = await classify_urls_batch(session, batch)
                    for result in batch_results:
                        results[result["url"]] = result["category"]
                    
                    # Update progress
                    jobs_store[job_id]["processed_urls"] += len(batch)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_index + 1} in job {job_id}: {str(e)}")
                    failed_urls.extend(batch)
                    continue

        # Update job status
        completion_status = "completed" if not failed_urls else "completed_with_errors"
        jobs_store[job_id].update({
            "status": completion_status,
            "results": results,
            "failed_urls": failed_urls if failed_urls else None,
            "completed_at": datetime.now(UTC).isoformat()
        })

        # Send callback if provided
        if jobs_store[job_id].get("callback_url"):
            await send_callback(jobs_store[job_id]["callback_url"], job_id)

    except Exception as e:
        logger.error(f"Error in job {job_id}: {str(e)}")
        jobs_store[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(UTC).isoformat()
        })

async def send_callback(callback_url: str, job_id: str):
    """Send callback with job results"""
    try:
        # Validate callback URL format
        if not callback_url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid callback URL format for job {job_id}: {callback_url}")
            return

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    callback_url,
                    json={"job_id": job_id, "results": jobs_store[job_id]},
                    timeout=10
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Callback failed for job {job_id}: HTTP {response.status}")
                    else:
                        logger.info(f"Callback successful for job {job_id}")
            except asyncio.TimeoutError:
                logger.error(f"Callback timeout for job {job_id}")
            except aiohttp.ClientError as e:
                logger.error(f"Callback network error for job {job_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in callback for job {job_id}: {str(e)}")

@app.post("/classify", response_model=ClassificationResult)
async def classify_sites(url_input: URLInput, background_tasks: BackgroundTasks):
    """Submit URLs for classification"""
    try:
        # Validate input
        if not url_input.urls:
            raise HTTPException(status_code=400, detail="No URLs provided")

        # Validate and clean URLs
        cleaned_urls = []
        for url in url_input.urls:
            # Basic URL cleaning
            url = url.strip().lower()
            if not url.startswith(('http://', 'https://')):
                url = f"http://{url}"
            cleaned_urls.append(url)

        # Create job
        job_id = str(uuid.uuid4())
        jobs_store[job_id] = {
            "status": "queued",
            "total_urls": len(cleaned_urls),
            "processed_urls": 0,
            "created_at": datetime.now(UTC).isoformat(),
            "callback_url": url_input.callback_url
        }

        # Start processing in background
        background_tasks.add_task(process_urls, cleaned_urls, job_id)

        return ClassificationResult(
            job_id=job_id,
            status="queued",
            total_urls=len(cleaned_urls)
        )

    except Exception as e:
        logger.error(f"Error submitting classification job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=ClassificationResult)
async def get_job_status(job_id: str):
    """Get status of a classification job"""
    try:
        if job_id not in jobs_store:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs_store[job_id]
        return ClassificationResult(
            job_id=job_id,
            status=job["status"],
            total_urls=job["total_urls"],
            processed_urls=job["processed_urls"],
            results=job.get("results"),
            error=job.get("error")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    try:
        if job_id not in jobs_store:
            raise HTTPException(status_code=404, detail="Job not found")

        del jobs_store[job_id]
        return JSONResponse(content={"message": "Job deleted successfully"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)