import os
from dotenv import load_dotenv
from email_handler import EmailHandler, EmailData
from vector_store import VectorStore
from classifier import EmailClassifier
from mailing_bot import MailingBot

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
        "Senior Python Developer Position",
        job_email
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
        "sender": "test@example.com"
    }
    vector_store.store_email(test_email, embedding)

    # Test similarity search
    similar_emails = vector_store.weighted_similarity_search(
        vector_store.generate_embedding("Python developer"),
        vector_store.generate_embedding("Looking for a developer")
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
        message_id="test123"
    )

    # Process the test email
    try:
        bot._process_single_email(test_email)
        print("Successfully processed test email")
    except Exception as e:
        print(f"Error processing test email: {e}")

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
