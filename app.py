from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from mailing_bot import MailingBot
from initialize_bot import initialize_system
import uvicorn
from typing import Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Email Bot API")
bot = None
initialization_task: Optional[asyncio.Task] = None

class EmailResponse(BaseModel):
    status: str
    message: str
    details: Optional[dict] = None

async def initialize_bot_async():
    """Initialize the bot asynchronously"""
    global bot
    try:
        logger.info("Starting async initialization...")
        # Run initialization in a thread pool
        bot_stores = await asyncio.get_event_loop().run_in_executor(None, initialize_system)
        bot = MailingBot()
        logger.info("Bot initialization completed successfully")
    except Exception as e:
        logger.error(f"Bot initialization failed: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    global initialization_task
    # Start initialization in background
    initialization_task = asyncio.create_task(initialize_bot_async())

@app.get("/", response_model=EmailResponse)
async def root():
    return {
        "status": "success",
        "message": "Email Bot API is running",
        "details": {"initialization_status": "in_progress" if not bot else "completed"}
    }

@app.post("/process-emails", response_model=EmailResponse)
async def process_emails(background_tasks: BackgroundTasks):
    """Trigger email processing in the background"""
    if not bot:
        return {
            "status": "error",
            "message": "Bot not initialized yet",
            "details": {"initialization_status": "in_progress"}
        }

    background_tasks.add_task(bot.process_new_emails)
    return {
        "status": "success",
        "message": "Email processing started in background"
    }

@app.get("/health", response_model=EmailResponse)
async def health_check():
    """Health check endpoint that returns immediately"""
    init_status = "not_started"
    if initialization_task:
        if initialization_task.done():
            init_status = "completed" if not initialization_task.exception() else "failed"
        else:
            init_status = "in_progress"

    return {
        "status": "success",
        "message": "Service is healthy",
        "details": {
            "initialization_status": init_status,
            "bot_ready": bot is not None
        }
    }

@app.get("/initialization-status", response_model=EmailResponse)
async def get_initialization_status():
    """Get detailed initialization status"""
    if not initialization_task:
        return {
            "status": "pending",
            "message": "Initialization not started",
            "details": {"initialization_status": "not_started"}
        }

    if initialization_task.done():
        if initialization_task.exception():
            return {
                "status": "error",
                "message": f"Initialization failed: {str(initialization_task.exception())}",
                "details": {"initialization_status": "failed"}
            }
        return {
            "status": "success",
            "message": "Initialization completed",
            "details": {"initialization_status": "completed"}
        }

    return {
        "status": "pending",
        "message": "Initialization in progress",
        "details": {"initialization_status": "in_progress"}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
