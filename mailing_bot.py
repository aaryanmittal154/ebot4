from typing import Dict, List, Any
from email_handler import EmailHandler, EmailData
from vector_store import VectorStore
from classifier import EmailClassifier, EmailCategory
from config import CONFIG
from specialized_vector_store import JobCandidateStore
import logging
from initialize_bot import process_email_content

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MailingBot:
    def __init__(self):
        """Initialize the mailing bot"""
        self.email_handler = EmailHandler()
        self.vector_store = None
        self.job_store = None
        self.classifier = EmailClassifier()
        logger.info("MailingBot base initialization complete")

    def process_new_emails(self):
        """Main processing loop for new emails"""
        if not self.vector_store or not self.job_store:
            logger.error("Vector stores not initialized")
            raise RuntimeError("Vector stores not initialized")

        logger.info("\n=== Checking for new emails ===")
        new_emails = self.email_handler.fetch_new_emails()
        logger.info(f"Found {len(new_emails)} new emails")

        for email in new_emails:
            try:
                logger.info(f"\n=== Processing email: {email.subject} ===")
                self._process_single_email(email)
            except Exception as e:
                logger.error(f"❌ Error processing email: {str(e)}")
                continue

    def _process_single_email(self, email: EmailData):
        """Process a single email"""
        print("1. Generating embeddings...")
        subject_embedding = self.vector_store.generate_embedding(email.subject)
        content_embedding = self.vector_store.generate_embedding(email.content)

        print("2. Storing email in vector database...")
        combined_embedding = [
            s * CONFIG["search"].SUBJECT_WEIGHT + c * CONFIG["search"].CONTENT_WEIGHT
            for s, c in zip(subject_embedding, content_embedding)
        ]
        try:
            self.vector_store.store_email(email.__dict__, combined_embedding)
            print("   ✅ Email stored in vector database")
        except Exception as e:
            print(f"   ❌ Failed to store email in vector database: {str(e)}")
            raise

        print("3. Classifying email...")
        category, confidence = self.classifier.classify_email(
            email.subject, email.content
        )
        print(f"   Category: {category.value}, Confidence: {confidence:.2f}")

        if confidence < 0.7:
            print("   Low confidence, treating as OTHER")
            category = EmailCategory.OTHER

        print("4. Generating response...")
        if category == EmailCategory.JOB:
            reply = self._handle_job_email(email)
            print("   Handled as JOB email")
        elif category == EmailCategory.CANDIDATE:
            reply = self._handle_candidate_email(email)
            print("   Handled as CANDIDATE email")
        else:
            reply = self._handle_other_email(email)
            print("   Handled as OTHER email")

        print("5. Sending reply...")
        self.email_handler.send_reply(email, reply)
        print("✅ Email processed successfully!")

    def _handle_job_email(self, email: EmailData) -> str:
        """Handle job-related email"""
        print("\n   Finding matching candidates...")

        # Store the job posting
        job_data = {
            "id": email.message_id,
            "title": email.subject,
            "description": email.content,
            "company": "From Email",  # Could be extracted from email
            "requirements": email.content,  # Could be extracted more precisely
        }
        self.job_store.store_job(job_data)

        # Find matching candidates
        matching_candidates = self.job_store.find_matching_candidates(email.content)

        if matching_candidates:
            response = "Thank you for your job posting. Here are our top matching candidates:\n\n"
            for i, candidate in enumerate(matching_candidates, 1):
                response += f"{i}. {candidate['name']}\n"
                response += f"   Skills: {candidate['skills']}\n"
                response += f"   Experience: {candidate['experience']}\n\n"
        else:
            response = "Thank you for your job posting. We'll keep your requirements in mind and notify you when we find matching candidates."

        return response

    def _handle_candidate_email(self, email: EmailData) -> str:
        """Handle candidate-related email"""
        print("\n   Finding matching jobs...")

        # Store the candidate profile
        candidate_data = {
            "id": email.message_id,
            "name": email.sender,
            "skills": email.content,  # Could be extracted more precisely
            "experience": "From Email",  # Could be extracted from email
            "background": email.content,
        }
        self.job_store.store_candidate(candidate_data)

        # Find matching jobs
        matching_jobs = self.job_store.find_matching_jobs(email.content)

        if matching_jobs:
            response = "Thank you for your interest. Here are some matching job opportunities:\n\n"
            for i, job in enumerate(matching_jobs, 1):
                response += f"{i}. {job['title']}\n"
                response += f"   Company: {job['company']}\n"
                response += f"   Requirements: {job['requirements'][:200]}...\n\n"
        else:
            response = "Thank you for your profile. We'll keep your details and notify you when relevant positions become available."

        return response

    def _handle_other_email(self, email: EmailData) -> str:
        """Handle other emails"""
        print("\n   Finding similar emails...")
        similar_emails = self.vector_store.weighted_similarity_search(
            self.vector_store.generate_embedding(email.subject),
            self.vector_store.generate_embedding(email.content),
        )

        # Let the classifier determine the appropriate context
        response = self.classifier.analyze_similar_emails(
            email.__dict__,
            similar_emails,
            additional_context="",  # Remove hardcoded job context
        )

        print("\n   Generated response:")
        print(response)
        return response

    def _format_candidates(self, candidates: List[Dict[str, str]]) -> str:
        return "\n".join(f"- {c['name']} ({c['skills']})" for c in candidates)

    def _format_jobs(self, jobs: List[Dict[str, str]]) -> str:
        return "\n".join(f"- {j['title']} at {j['company']}" for j in jobs)


if __name__ == "__main__":
    bot = MailingBot()
    while True:
        bot.process_new_emails()
        time.sleep(60)  # Check every minute
