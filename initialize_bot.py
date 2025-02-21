import os
import time
import email
import logging
from dotenv import load_dotenv
from email_handler import EmailHandler, EmailData
from vector_store import VectorStore
from specialized_vector_store import JobCandidateStore
from config import CONFIG
from pinecone import Pinecone
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_vector_stores():
    """Initialize both vector stores"""
    pc = Pinecone(api_key=CONFIG["pinecone"].API_KEY)

    # Initialize main email vector store
    logger.info("Initializing main email vector store...")
    try:
        logger.info("Checking for existing email index...")
        pc.delete_index(CONFIG["pinecone"].INDEX_NAME)
        logger.info("✅ Old email index deleted")
        time.sleep(5)  # Wait for deletion to complete
    except Exception as e:
        logger.info(f"No existing email index to delete: {str(e)}")

    logger.info("Creating new email vector store...")
    vector_store = VectorStore()
    logger.info("✅ Main email vector store initialized")

    # Initialize job/candidate vector store
    logger.info("\nInitializing job/candidate vector store...")
    try:
        logger.info("Checking for existing job-candidates index...")
        pc.delete_index("job-candidates")
        logger.info("✅ Old job-candidates index deleted")
        time.sleep(5)  # Wait for deletion to complete
    except Exception as e:
        logger.info(f"No existing job-candidates index to delete: {str(e)}")

    logger.info("Creating new job/candidate store...")
    job_store = JobCandidateStore()
    logger.info("✅ Job/candidate store initialized")

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
    """Initialize the entire system"""
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

    # Fetch all emails from Gmail
    logger.info("\nFetching all emails from Gmail...")
    email_handler.imap.select("INBOX")
    _, messages = email_handler.imap.search(None, "ALL")

    total_emails = len(messages[0].split())
    logger.info(f"Found {total_emails} emails in inbox")

    # Process each email
    processed = 0
    errors = 0
    job_related = 0
    candidate_related = 0

    try:
        for msg_num in messages[0].split():
            try:
                logger.info(
                    f"\nProcessing email {processed + 1}/{total_emails}",
                )

                _, msg_data = email_handler.imap.fetch(msg_num, "(RFC822)")
                email_body = msg_data[0][1]
                msg = email.message_from_bytes(email_body)

                content = email_handler._get_email_content(msg)
                thread_id = msg.get("Thread-Index") or msg.get("In-Reply-To")
                references = (
                    msg.get("References", "").split() if msg.get("References") else []
                )

                email_data = EmailData(
                    subject=msg["subject"] or "",
                    content=content or "",
                    thread_id=thread_id,
                    references=references,
                    sender=msg["from"],
                    message_id=msg["message-id"] or f"generated_id_{msg_num.decode()}",
                )

                # Show progress
                logger.info(f" - Subject: {email_data.subject[:50]}...")

                # Process email content
                process_email_content(email_data, vector_store, job_store)

                processed += 1

                if processed % 10 == 0:
                    logger.info(f"✅ Processed {processed}/{total_emails} emails")

            except Exception as e:
                errors += 1
                logger.error(f"❌ Error processing email {msg_num.decode()}: {str(e)}")
                continue

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user")
    finally:
        logger.info("\n=== Processing Summary ===")
        logger.info(f"Total emails: {total_emails}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Completion rate: {(processed/total_emails)*100:.1f}%")

    logger.info("\n=== System Initialization Complete ===")
    logger.info(
        "\nYou can now run the bot with 'python run_bot.py' and send a test email."
    )

    return vector_store, job_store


if __name__ == "__main__":
    initialize_system()
