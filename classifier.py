from enum import Enum
from typing import Dict, Any, Tuple, List
from openai import OpenAI
from config import CONFIG


class EmailCategory(Enum):
    JOB = "job"
    CANDIDATE = "candidate"
    OTHER = "other"


class EmailType(Enum):
    TECHNICAL = "technical"
    JOB_RELATED = "job_related"
    GENERAL = "general"


class EmailClassifier:
    def __init__(self):
        self.client = OpenAI()

    def classify_email(self, subject: str, content: str) -> Tuple[EmailCategory, float]:
        """Classify email using GPT-4o-mini"""
        prompt = f"""
        Analyze the following email and classify it into one of these categories:
        1. JOB - Job postings, job requirements, or hiring-related content
        2. CANDIDATE - Resumes, job applications, or candidate-related content
        3. OTHER - Any other type of email

        Return only the classification and confidence score (0-1) in format: category:confidence

        Subject: {subject}
        Content: {content}
        """

        response = self.client.chat.completions.create(
            model=CONFIG["llm"].MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        result = response.choices[0].message.content.strip().lower()
        category, confidence = result.split(":")
        return EmailCategory(category), float(confidence)

    def detect_email_type(self, subject: str, content: str) -> EmailType:
        """Detect the specific type of email for better context"""
        prompt = f"""
        Analyze this email and determine its type. Choose one:
        1. TECHNICAL - Technical questions, programming issues, development queries
        2. JOB_RELATED - Anything about jobs, hiring, careers
        3. GENERAL - General inquiries, other topics

        Return only the type (e.g., "TECHNICAL")

        Subject: {subject}
        Content: {content}
        """

        response = self.client.chat.completions.create(
            model=CONFIG["llm"].MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return EmailType(response.choices[0].message.content.strip().lower())

    def analyze_similar_emails(
        self,
        new_email: Dict[str, Any],
        similar_emails: List[Dict[str, Any]],
        additional_context: str = "",
    ) -> str:
        """Analyze similar emails using GPT-4o-mini"""

        # Detect email type and set appropriate context
        email_type = self.detect_email_type(
            new_email.get("subject", ""), new_email.get("content", "")
        )

        context_map = {
            EmailType.TECHNICAL: """
                This is a technical question.
                Focus on providing specific technical guidance and solutions.
                If similar technical questions exist in the context, build upon those answers.
                Include relevant technical resources or documentation if applicable.
            """,
            EmailType.JOB_RELATED: """
                This email is related to jobs/careers.
                Focus on providing relevant job-related information and next steps.
                If similar job-related threads exist, maintain consistency in responses.
            """,
            EmailType.GENERAL: """
                This is a general inquiry.
                Provide a helpful and informative response.
                Maintain context from any similar previous conversations.
            """,
        }

        prompt = f"""
        Analyze this email and generate an appropriate response.
        Consider the context from similar previous emails when crafting the response.

        New Email:
        Subject: {new_email.get('subject', '')}
        Content: {new_email.get('content', '')}

        Similar Previous Emails:
        {self._format_similar_emails(similar_emails)}

        Context:
        {context_map[email_type]}
        {additional_context}

        Instructions:
        1. If this is a new topic, provide a relevant response based on the email content
        2. If this is part of an existing thread, ensure the response maintains context
        3. Keep the tone professional and helpful
        4. If specific action items are mentioned, acknowledge them
        5. For technical queries, provide specific technical guidance
        6. For job-related queries, provide relevant career/job information
        7. Include any relevant links or resources
        """

        # Print the prompt for debugging
        print("\n=== Email Type ===")
        print(f"Detected Type: {email_type.value}")

        print("\n=== LLM Input Prompt ===")
        print(prompt)
        print("=== End of Prompt ===\n")

        response = self.client.chat.completions.create(
            model=CONFIG["llm"].MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["llm"].TEMPERATURE,
            max_tokens=CONFIG["llm"].MAX_TOKENS,
        )

        print("\n=== LLM Response ===")
        print(response.choices[0].message.content)
        print("=== End of Response ===\n")

        return response.choices[0].message.content

    def _format_similar_emails(self, emails: List[Dict[str, Any]]) -> str:
        """Format similar emails for prompt"""
        # Deduplicate emails based on content
        seen_contents = set()
        unique_emails = []

        for email in emails:
            content = email.get("content", "")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_emails.append(email)

        # Format unique emails
        formatted = []
        for i, email in enumerate(unique_emails, 1):
            formatted.append(
                f"""
            Email {i}:
            Subject: {email.get('subject', '')}
            Content: {email.get('content', '')}
            """
            )
        return "\n".join(formatted)
