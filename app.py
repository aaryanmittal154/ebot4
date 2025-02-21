from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from mailing_bot import MailingBot
from initialize_bot import initialize_system
import uvicorn
from typing import Optional
import asyncio
import logging
import os
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Email Bot API")

# Global variables
bot = None
initialization_task: Optional[asyncio.Task] = None
initialization_status = {
    "state": "not_started",
    "error": None,
    "details": {},
    "last_update": None,
}


def update_init_status(state: str, error: Optional[str] = None, details: dict = None):
    """Update initialization status"""
    initialization_status["state"] = state
    initialization_status["error"] = error
    initialization_status["details"] = details or {}
    initialization_status["last_update"] = time.time()


async def initialize_bot_async():
    """Initialize the bot asynchronously"""
    global bot
    try:
        update_init_status("in_progress", details={"step": "connecting to services"})

        # Initialize in thread pool to not block
        bot_stores = await asyncio.get_event_loop().run_in_executor(
            None, initialize_system
        )
        bot = MailingBot()

        update_init_status("completed", details={"step": "ready for new emails"})
        logger.info("Bot initialization completed successfully")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Bot initialization failed: {error_msg}")
        update_init_status("failed", error=error_msg)


@app.on_event("startup")
async def startup_event():
    # Load environment variables
    load_dotenv()

    # Initialize the bot
    global initialization_task
    initialization_task = asyncio.create_task(initialize_bot_async())
    logger.info("Application startup complete - bot initializing")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "success",
        "message": "Email Bot API is running",
        "details": {
            "initialization_status": initialization_status["state"],
            "last_update": initialization_status["last_update"],
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/initialization-status")
async def get_initialization_status():
    """Get initialization status"""
    return {
        "status": "success",
        "data": initialization_status,
    }


@app.post("/process-emails")
async def process_emails(background_tasks: BackgroundTasks):
    """Process new emails endpoint"""
    if initialization_status["state"] != "completed":
        return {
            "status": "error",
            "message": "Bot initialization not complete",
            "details": initialization_status,
        }

    if not bot:
        return {
            "status": "error",
            "message": "Bot not initialized",
        }

    try:
        # Process only new emails in background
        background_tasks.add_task(lambda: bot.process_new_emails())
        return {
            "status": "success",
            "message": "Processing new emails in background",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to process emails: {str(e)}",
        }


if __name__ == "__main__":
    uvicorn.run(
        "app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False
    )
