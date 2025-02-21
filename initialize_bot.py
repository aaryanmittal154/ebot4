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

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    logger.info(f"Initializing Pinecone with environment: {pinecone_env}")

    # Update CONFIG with environment values
    CONFIG["pinecone"].API_KEY = pinecone_api_key
    CONFIG["pinecone"].ENVIRONMENT = pinecone_env

    # Initialize Pinecone client with explicit environment
    try:
        pc = PineconeGRPC(api_key=pinecone_api_key)
        logger.info("✅ Pinecone client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {str(e)}")
        raise

    # Connect to existing email vector store
    logger.info("Connecting to existing email vector store...")
    try:
        vector_store = VectorStore()
        # Test connection
        vector_store.index.describe_index_stats()
        logger.info("✅ Connected to email vector store")
    except Exception as e:
        logger.error(f"Failed to connect to email vector store: {str(e)}")
        raise

    # Connect to existing job/candidate vector store
    logger.info("\nConnecting to existing job/candidate vector store...")
    try:
        job_store = JobCandidateStore()
        # Test connection
        job_store.job_index.describe_index_stats()
        logger.info("✅ Connected to job/candidate store")
    except Exception as e:
        logger.error(f"Failed to connect to job/candidate store: {str(e)}")
        raise

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

    # Check required environment variables
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

    # Update config
    logger.info("Updating configuration...")
    CONFIG["email"].EMAIL = os.getenv("EMAIL_ADDRESS")
    CONFIG["email"].PASSWORD = os.getenv("EMAIL_PASSWORD")
    CONFIG["pinecone"].API_KEY = os.getenv("PINECONE_API_KEY")
    CONFIG["pinecone"].ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Initialize vector stores with retry
    max_retries = 3
    retry_delay = 5
    last_error = None

    for attempt in range(max_retries):
        try:
            vector_store, job_store = initialize_vector_stores()
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.error("All initialization attempts failed")
                raise last_error

    # Initialize email handler
    logger.info("\nConnecting to Gmail...")
    try:
        email_handler = EmailHandler()
        logger.info("✅ Connected to Gmail")
    except Exception as e:
        logger.error(f"Failed to connect to Gmail: {str(e)}")
        raise

    logger.info("\n=== System Initialization Complete ===")
    logger.info("Ready to process new emails")

    return vector_store, job_store


if __name__ == "__main__":
    initialize_system()
