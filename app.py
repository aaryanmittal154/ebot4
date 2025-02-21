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

logger = logging.getLogger(__name__)

app = FastAPI(title="Email Bot API")
bot = None
initialization_task: Optional[asyncio.Task] = None
initialization_attempts = 0
MAX_INITIALIZATION_ATTEMPTS = 3


class EmailResponse(BaseModel):
    status: str
    message: str
    details: Optional[dict] = None


def check_required_env_vars() -> tuple[bool, list[str]]:
    """Check if all required environment variables are set"""
    required_vars = [
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "OPENAI_API_KEY",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    return len(missing_vars) == 0, missing_vars


async def initialize_bot_async():
    """Initialize the bot asynchronously with retries"""
    global bot, initialization_attempts

    # Check environment variables first
    env_vars_set, missing_vars = check_required_env_vars()
    if not env_vars_set:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    while initialization_attempts < MAX_INITIALIZATION_ATTEMPTS:
        try:
            logger.info(
                f"Starting initialization attempt {initialization_attempts + 1}/{MAX_INITIALIZATION_ATTEMPTS}"
            )
            # Run initialization in a thread pool
            bot_stores = await asyncio.get_event_loop().run_in_executor(
                None, initialize_system
            )
            bot = MailingBot()
            logger.info("Bot initialization completed successfully")
            return
        except Exception as e:
            initialization_attempts += 1
            if initialization_attempts >= MAX_INITIALIZATION_ATTEMPTS:
                logger.error(
                    f"Bot initialization failed after {MAX_INITIALIZATION_ATTEMPTS} attempts: {str(e)}"
                )
                raise
            logger.warning(
                f"Initialization attempt {initialization_attempts} failed: {str(e)}"
            )
            await asyncio.sleep(5)  # Wait before retrying


@app.on_event("startup")
async def startup_event():
    global initialization_task
    # Load environment variables
    load_dotenv()
    # Start initialization in background
    initialization_task = asyncio.create_task(initialize_bot_async())


@app.get("/", response_model=EmailResponse)
async def root():
    env_vars_set, missing_vars = check_required_env_vars()
    status_details = {
        "initialization_status": "in_progress" if not bot else "completed",
        "environment_variables_set": env_vars_set,
    }
    if not env_vars_set:
        status_details["missing_variables"] = missing_vars

    return {
        "status": "success",
        "message": "Email Bot API is running",
        "details": status_details,
    }


@app.post("/process-emails", response_model=EmailResponse)
async def process_emails(background_tasks: BackgroundTasks):
    """Trigger email processing in the background"""
    if not bot:
        status_details = {
            "initialization_status": "in_progress",
            "attempts": initialization_attempts,
        }
        if (
            initialization_task
            and initialization_task.done()
            and initialization_task.exception()
        ):
            status_details["error"] = str(initialization_task.exception())

        return {
            "status": "error",
            "message": "Bot not initialized yet",
            "details": status_details,
        }

    background_tasks.add_task(bot.process_new_emails)
    return {"status": "success", "message": "Email processing started in background"}


@app.get("/health", response_model=EmailResponse)
async def health_check():
    """Health check endpoint that returns immediately"""
    env_vars_set, missing_vars = check_required_env_vars()

    init_status = "not_started"
    if initialization_task:
        if initialization_task.done():
            init_status = (
                "completed" if not initialization_task.exception() else "failed"
            )
        else:
            init_status = "in_progress"

    return {
        "status": "success",
        "message": "Service is healthy",
        "details": {
            "initialization_status": init_status,
            "bot_ready": bot is not None,
            "environment_variables_set": env_vars_set,
            "initialization_attempts": initialization_attempts,
        },
    }


@app.get("/initialization-status", response_model=EmailResponse)
async def get_initialization_status():
    """Get detailed initialization status"""
    env_vars_set, missing_vars = check_required_env_vars()

    if not env_vars_set:
        return {
            "status": "error",
            "message": "Missing required environment variables",
            "details": {
                "initialization_status": "failed",
                "missing_variables": missing_vars,
            },
        }

    if not initialization_task:
        return {
            "status": "pending",
            "message": "Initialization not started",
            "details": {
                "initialization_status": "not_started",
                "attempts": initialization_attempts,
            },
        }

    if initialization_task.done():
        if initialization_task.exception():
            return {
                "status": "error",
                "message": f"Initialization failed: {str(initialization_task.exception())}",
                "details": {
                    "initialization_status": "failed",
                    "attempts": initialization_attempts,
                    "error": str(initialization_task.exception()),
                },
            }
        return {
            "status": "success",
            "message": "Initialization completed",
            "details": {
                "initialization_status": "completed",
                "attempts": initialization_attempts,
            },
        }

    return {
        "status": "pending",
        "message": "Initialization in progress",
        "details": {
            "initialization_status": "in_progress",
            "attempts": initialization_attempts,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
