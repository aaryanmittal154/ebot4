import os
import time
from dotenv import load_dotenv
from mailing_bot import MailingBot
from config import CONFIG


def main():
    print("\n=== Email Bot Starting ===")
    print("Loading environment variables...")
    load_dotenv()

    print("\nUpdating configuration...")
    CONFIG["email"].EMAIL = os.getenv("EMAIL_ADDRESS")
    CONFIG["email"].PASSWORD = os.getenv("EMAIL_PASSWORD")
    CONFIG["pinecone"].API_KEY = os.getenv("PINECONE_API_KEY")
    CONFIG["pinecone"].ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    print("\nInitializing bot...")
    bot = MailingBot()
    print("\n✅ Bot initialized successfully!")
    print("\n=== Bot is now running ===")
    print("Press Ctrl+C to stop")

    try:
        while True:
            bot.process_new_emails()
            print("\nWaiting 60 seconds before next check...")
            for i in range(60):
                if i % 10 == 0:
                    print(".", end="", flush=True)
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Bot stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error running bot: {str(e)}")


if __name__ == "__main__":
    main()
