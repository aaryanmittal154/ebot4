import os
from dotenv import load_dotenv
from email_handler import EmailHandler, EmailData
from vector_store import VectorStore
from classifier import EmailClassifier
from mailing_bot import MailingBot
import requests
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_email_classification():
    classifier = EmailClassifier()

    # Test job email
    job_email = """
    We are looking for a Senior Python Developer
    Requirements:
    - 5+ years of Python experience
    - Experience with Django/Flask
    - Strong understanding of REST APIs
    """
    category, confidence = classifier.classify_email(
        "Senior Python Developer Position", job_email
    )
    print(f"Job Email Classification: {category}, Confidence: {confidence}")


def test_vector_store():
    vector_store = VectorStore()

    # Test embedding generation
    text = "This is a test email about Python development"
    embedding = vector_store.generate_embedding(text)
    print(f"Generated embedding length: {len(embedding)}")

    # Test storing email
    test_email = {
        "message_id": "test123",
        "subject": "Test Email",
        "content": text,
        "thread_id": None,
        "sender": "test@example.com",
    }
    vector_store.store_email(test_email, embedding)

    # Test similarity search
    similar_emails = vector_store.weighted_similarity_search(
        vector_store.generate_embedding("Python developer"),
        vector_store.generate_embedding("Looking for a developer"),
    )
    print("Similar emails found:", len(similar_emails))


def test_full_bot():
    bot = MailingBot()

    # Create a test email
    test_email = EmailData(
        subject="Python Developer Position",
        content="We are looking for a Senior Python Developer...",
        thread_id=None,
        references=[],
        sender="test@example.com",
        message_id="test123",
    )

    # Process the test email
    try:
        bot._process_single_email(test_email)
        print("Successfully processed test email")
    except Exception as e:
        print(f"Error processing test email: {e}")


class EmailBotTester:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        logger.info(f"Initializing tester for {self.base_url}")

    def test_initialization_status(self):
        """Test the initialization status endpoint"""
        try:
            response = requests.get(f"{self.base_url}/initialization-status")
            logger.info(f"Initialization status response: {response.status_code}")
            logger.info(f"Response body: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Initialization status check failed: {str(e)}")
            return False

    def test_process_emails(self):
        """Test the process emails endpoint"""
        try:
            response = requests.post(f"{self.base_url}/process-emails")
            logger.info(f"Process emails response: {response.status_code}")
            logger.info(f"Response body: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Process emails failed: {str(e)}")
            return False

    def run_full_test(self, max_retries=3, retry_delay=5):
        """Run a full test suite with retries"""
        logger.info("Starting full test suite")

        for attempt in range(max_retries):
            logger.info(f"\nTest attempt {attempt + 1}/{max_retries}")

            # Test initialization status
            if not self.test_initialization_status():
                logger.warning("Initialization status check failed")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return False

            # Test process emails
            if not self.test_process_emails():
                logger.warning("Process emails test failed")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return False

            logger.info("\n✅ All tests passed successfully!")
            return True

        logger.error("\n❌ Tests failed after all retries")
        return False


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Update config with environment variables
    from config import CONFIG

    CONFIG["email"].EMAIL = os.getenv("EMAIL_ADDRESS")
    CONFIG["email"].PASSWORD = os.getenv("EMAIL_PASSWORD")
    CONFIG["pinecone"].API_KEY = os.getenv("PINECONE_API_KEY")
    CONFIG["pinecone"].ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    print("Testing Email Classification...")
    test_email_classification()

    print("\nTesting Vector Store...")
    test_vector_store()

    print("\nTesting Full Bot...")
    test_full_bot()

    # Create tester instance
    tester = EmailBotTester("https://email-bot-production-d494.up.railway.app")

    # Run the full test suite
    success = tester.run_full_test()

    if success:
        logger.info("\n=== Test Summary ===")
        logger.info("All endpoints are working correctly")
        logger.info("The email bot is ready to process emails")
    else:
        logger.error("\n=== Test Summary ===")
        logger.error("Some tests failed")
        logger.error("Please check the logs above for details")
