import os
import time
import email
import logging
from dotenv import load_dotenv
from email_handler import EmailHandler, EmailData
from vector_store import VectorStore
from specialized_vector_store import JobCandidateStore
from config import CONFIG
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_vector_stores():
    """Initialize both vector stores"""
    # Get Pinecone configuration
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

    logger.info(f"Initializing Pinecone with environment: {pinecone_env}")

    # Update CONFIG with environment values
    CONFIG["pinecone"].API_KEY = pinecone_api_key
    CONFIG["pinecone"].ENVIRONMENT = pinecone_env

    # Initialize Pinecone client with explicit environment
    pc = PineconeGRPC(api_key=pinecone_api_key)

    # Connect to existing email vector store
    logger.info("Connecting to existing email vector store...")
    vector_store = VectorStore()
    logger.info("✅ Connected to email vector store")

    # Connect to existing job/candidate vector store
    logger.info("\nConnecting to existing job/candidate vector store...")
    job_store = JobCandidateStore()
    logger.info("✅ Connected to job/candidate store")

    return vector_store, job_store


def classify_email_with_llm(subject: str, content: str) -> tuple[str, float]:
    """Use GPT-4o-mini to classify email type"""
    client = OpenAI()

    prompt = f"""
    Analyze this email and classify it into one of these categories:
    1. JOB - If it's a job posting, hiring notice, or career opportunity
    2. CANDIDATE - If it's a job application, resume, or candidate introducing themselves
    3. GENERAL - For any other type of email

    Return only the classification and confidence (0-1) separated by a comma. Example: "JOB,0.95"

    Subject: {subject}
    Content: {content[:500]}  # Using first 500 chars for efficiency
    """

    response = client.chat.completions.create(
        model=CONFIG["llm"].MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    result = response.choices[0].message.content.strip()
    category, confidence = result.split(",")
    return category.strip(), float(confidence)


def process_email_content(
    email_data: EmailData, vector_store: VectorStore, job_store: JobCandidateStore
):
    """Process a single email and store in appropriate vector stores"""
    # Generate and store embeddings in main vector store
    subject_embedding = vector_store.generate_embedding(email_data.subject)
    content_embedding = vector_store.generate_embedding(email_data.content)

    combined_embedding = [
        s * CONFIG["search"].SUBJECT_WEIGHT + c * CONFIG["search"].CONTENT_WEIGHT
        for s, c in zip(subject_embedding, content_embedding)
    ]

    vector_store.store_email(email_data.__dict__, combined_embedding)

    # Use LLM to classify email
    category, confidence = classify_email_with_llm(
        email_data.subject, email_data.content
    )
    logger.info(f"Email classified as {category} with confidence {confidence:.2f}")

    # Only process if confidence is high enough
    if confidence < 0.7:
        logger.info("Low confidence in classification, treating as GENERAL")
        return

    if category == "JOB":
        job_data = {
            "id": email_data.message_id,
            "title": email_data.subject,
            "description": email_data.content,
            "company": "From Email",
            "requirements": email_data.content,
        }
        job_store.store_job(job_data)
        logger.info(f"Stored job posting: {email_data.subject}")

    elif category == "CANDIDATE":
        candidate_data = {
            "id": email_data.message_id,
            "name": email_data.sender,
            "skills": email_data.content,
            "experience": "From Email",
            "background": email_data.content,
        }
        job_store.store_candidate(candidate_data)
        logger.info(f"Stored candidate profile: {email_data.subject}")


def initialize_system():
    """Initialize the system without processing old emails"""
    logger.info("=== Initializing Email Bot System ===")

    # Load environment variables
    logger.info("Loading environment variables...")
    load_dotenv()

    # Update config
    logger.info("Updating configuration...")
    CONFIG["email"].EMAIL = os.getenv("EMAIL_ADDRESS")
    CONFIG["email"].PASSWORD = os.getenv("EMAIL_PASSWORD")
    CONFIG["pinecone"].API_KEY = os.getenv("PINECONE_API_KEY")
    CONFIG["pinecone"].ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Initialize vector stores
    vector_store, job_store = initialize_vector_stores()

    # Initialize email handler
    logger.info("\nConnecting to Gmail...")
    email_handler = EmailHandler()
    logger.info("✅ Connected to Gmail")

    logger.info("\n=== System Initialization Complete ===")
    logger.info("Ready to process new emails")

    return vector_store, job_store


if __name__ == "__main__":
    initialize_system()
