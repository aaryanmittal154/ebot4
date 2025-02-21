import os
import time
import email
from dotenv import load_dotenv
from email_handler import EmailHandler, EmailData
from vector_store import VectorStore
from config import CONFIG
from pinecone import Pinecone

def initialize_system():
    print("\n=== Initializing Email Bot System ===")

    # Load environment variables
    print("Loading environment variables...")
    load_dotenv()

    # Update config
    print("Updating configuration...")
    CONFIG["email"].EMAIL = os.getenv("EMAIL_ADDRESS")
    CONFIG["email"].PASSWORD = os.getenv("EMAIL_PASSWORD")
    CONFIG["pinecone"].API_KEY = os.getenv("PINECONE_API_KEY")
    CONFIG["pinecone"].ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Delete existing Pinecone index
    print("\nDeleting existing Pinecone index...")
    pc = Pinecone(api_key=CONFIG["pinecone"].API_KEY)
    try:
        pc.delete_index(CONFIG["pinecone"].INDEX_NAME)
        print("✅ Old index deleted successfully")
        time.sleep(5)  # Wait for deletion to complete
    except Exception as e:
        print(f"Note: No existing index to delete ({str(e)})")

    # Initialize vector store (this will create a new index)
    print("\nInitializing vector store...")
    vector_store = VectorStore()

    # Initialize email handler
    print("\nConnecting to Gmail...")
    email_handler = EmailHandler()

    # Fetch all emails from Gmail
    print("\nFetching all emails from Gmail...")
    email_handler.imap.select('INBOX')
    _, messages = email_handler.imap.search(None, 'ALL')

    total_emails = len(messages[0].split())
    print(f"Found {total_emails} emails in inbox")

    # Process each email
    processed = 0
    errors = 0

    try:
        for msg_num in messages[0].split():
            try:
                print(f"\nProcessing email {processed + 1}/{total_emails}", end="", flush=True)

                _, msg_data = email_handler.imap.fetch(msg_num, '(RFC822)')
                email_body = msg_data[0][1]
                msg = email.message_from_bytes(email_body)

                content = email_handler._get_email_content(msg)
                thread_id = msg.get('Thread-Index') or msg.get('In-Reply-To')
                references = msg.get('References', '').split() if msg.get('References') else []

                email_data = EmailData(
                    subject=msg['subject'] or "",
                    content=content or "",
                    thread_id=thread_id,
                    references=references,
                    sender=msg['from'],
                    message_id=msg['message-id'] or f"generated_id_{msg_num.decode()}"
                )

                # Show progress
                print(f" - Subject: {email_data.subject[:50]}...")

                # Generate and store embeddings
                subject_embedding = vector_store.generate_embedding(email_data.subject)
                content_embedding = vector_store.generate_embedding(email_data.content)

                combined_embedding = [
                    s * CONFIG["search"].SUBJECT_WEIGHT + c * CONFIG["search"].CONTENT_WEIGHT
                    for s, c in zip(subject_embedding, content_embedding)
                ]

                vector_store.store_email(email_data.__dict__, combined_embedding)
                processed += 1

                if processed % 10 == 0:
                    print(f"✅ Processed {processed}/{total_emails} emails")

            except Exception as e:
                errors += 1
                print(f"\n❌ Error processing email {msg_num.decode()}: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    finally:
        print(f"\n=== Processing Summary ===")
        print(f"Total emails: {total_emails}")
        print(f"Successfully processed: {processed}")
        print(f"Errors: {errors}")
        print(f"Completion rate: {(processed/total_emails)*100:.1f}%")

    print("\n=== System Initialization Complete ===")
    print("\nYou can now run the bot with 'python run_bot.py' and send a test email.")

if __name__ == "__main__":
    initialize_system()
