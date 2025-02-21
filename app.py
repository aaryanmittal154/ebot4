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
vector_stores = None
initialization_task: Optional[asyncio.Task] = None
initialization_lock = asyncio.Lock()
polling_task: Optional[asyncio.Task] = None
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


async def poll_emails():
    """Poll for new emails periodically"""
    global bot
    while True:
        try:
            if bot and initialization_status["state"] == "completed":
                logger.info("Checking for new emails...")
                bot.process_new_emails()
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in email polling: {str(e)}")
            await asyncio.sleep(60)  # Wait before retrying


async def initialize_bot_async():
    """Initialize the bot asynchronously"""
    global bot, vector_stores, initialization_task, polling_task

    # Use lock to prevent parallel initialization
    async with initialization_lock:
        # Check if already initialized or initializing
        if initialization_status["state"] == "completed":
            logger.info("Bot already initialized")
            return
        if initialization_status["state"] == "in_progress":
            logger.info("Bot initialization already in progress")
            return

        try:
            # Check environment variables first
            required_vars = [
                "EMAIL_ADDRESS",
                "EMAIL_PASSWORD",
                "PINECONE_API_KEY",
                "PINECONE_ENVIRONMENT",
                "OPENAI_API_KEY",
            ]

            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )

            update_init_status(
                "in_progress", details={"step": "checking environment variables"}
            )

            # Initialize vector stores with timeout
            if not vector_stores:
                update_init_status(
                    "in_progress", details={"step": "initializing vector stores"}
                )
                try:
                    vector_stores = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, initialize_system
                        ),
                        timeout=60,  # 60 seconds timeout
                    )
                    logger.info("Vector stores initialized successfully")
                except asyncio.TimeoutError:
                    raise TimeoutError("Vector store initialization timed out")

            # Initialize bot with timeout
            if not bot:
                update_init_status("in_progress", details={"step": "initializing bot"})
                try:
                    bot = MailingBot()
                    bot.vector_store = vector_stores[0]
                    bot.job_store = vector_stores[1]
                    logger.info("Bot initialized with existing vector stores")
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize bot: {str(e)}")

            update_init_status("completed", details={"step": "initialization complete"})
            logger.info("Bot initialization completed successfully")

            # Start email polling after successful initialization
            if not polling_task or polling_task.done():
                polling_task = asyncio.create_task(poll_emails())
                logger.info("Started automatic email polling")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Bot initialization failed: {error_msg}")
            update_init_status("failed", error=error_msg)
            raise


@app.on_event("startup")
async def startup_event():
    # Load environment variables
    load_dotenv()

    # Start initialization in background
    global initialization_task
    if not initialization_task or initialization_task.done():
        initialization_task = asyncio.create_task(initialize_bot_async())
        logger.info("Application startup complete - bot initializing in background")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global polling_task
    if polling_task:
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            pass
        logger.info("Email polling stopped")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "success",
        "message": "Email Bot API is running",
        "details": {
            "initialization_status": initialization_status["state"],
            "last_update": initialization_status["last_update"],
            "error": initialization_status.get("error"),
            "polling_active": (
                polling_task is not None and not polling_task.done()
                if polling_task
                else False
            ),
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/initialization-status")
async def get_initialization_status():
    """Get initialization status"""
    # Check if initialization task failed
    if (
        initialization_task
        and initialization_task.done()
        and initialization_task.exception()
    ):
        return {
            "status": "error",
            "message": "Initialization failed",
            "details": {
                "error": str(initialization_task.exception()),
                "state": initialization_status["state"],
                "last_update": initialization_status["last_update"],
            },
        }

    return {
        "status": "success",
        "data": initialization_status,
        "polling_active": (
            polling_task is not None and not polling_task.done()
            if polling_task
            else False
        ),
    }


@app.post("/process-emails")
async def process_emails(background_tasks: BackgroundTasks):
    """Process new emails endpoint (manual trigger)"""
    # Check initialization status
    if initialization_status["state"] == "failed":
        return {
            "status": "error",
            "message": "Bot initialization failed",
            "details": initialization_status,
        }
    elif initialization_status["state"] != "completed":
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
        logger.error(f"Failed to process emails: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to process emails: {str(e)}",
        }


if __name__ == "__main__":
    # Force single worker to prevent multiple initializations
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        timeout_keep_alive=120,
        workers=1,  # Force single worker
    )
