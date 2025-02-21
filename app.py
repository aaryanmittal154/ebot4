from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from mailing_bot import MailingBot
from initialize_bot import initialize_system
import uvicorn

app = FastAPI(title="Email Bot API")
bot = None

class EmailResponse(BaseModel):
    status: str
    message: str

@app.on_event("startup")
async def startup_event():
    global bot
    # Initialize the system and bot
    initialize_system()
    bot = MailingBot()

@app.get("/", response_model=EmailResponse)
async def root():
    return {"status": "success", "message": "Email Bot API is running"}

@app.post("/process-emails", response_model=EmailResponse)
async def process_emails(background_tasks: BackgroundTasks):
    """Trigger email processing in the background"""
    if not bot:
        return {"status": "error", "message": "Bot not initialized"}

    background_tasks.add_task(bot.process_new_emails)
    return {"status": "success", "message": "Email processing started in background"}

@app.get("/health", response_model=EmailResponse)
async def health_check():
    """Health check endpoint"""
    return {"status": "success", "message": "Service is healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
