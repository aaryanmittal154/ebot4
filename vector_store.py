from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from config import CONFIG
import time
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.openai_client = OpenAI()
        # Initialize Pinecone with explicit environment
        self.pinecone = PineconeGRPC(api_key=CONFIG["pinecone"].API_KEY)
        self._init_index()

    def _init_index(self):
        """Initialize Pinecone index"""
        index_name = CONFIG["pinecone"].INDEX_NAME

        try:
            # Try to describe the index first
            logger.info(f"Checking index: {index_name}")
            self.pinecone.describe_index(index_name)
            logger.info(f"Index {index_name} already exists, connecting...")
        except Exception as e:
            logger.info(f"Creating new index: {index_name}")
            try:
                self.pinecone.create_index(
                    name=index_name,
                    dimension=CONFIG["pinecone"].DIMENSION,
                    metric=CONFIG["pinecone"].METRIC,
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info("Waiting for index to be ready...")
                while not self.pinecone.describe_index(index_name).status.ready:
                    time.sleep(1)
                    logger.info(".", end="", flush=True)
                logger.info("\nIndex created successfully!")
            except Exception as create_error:
                logger.error(f"Error creating index: {str(create_error)}")
                raise

        self.index = self.pinecone.Index(index_name)
        logger.info("Successfully connected to index!")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI's API"""
        # Use text-embedding-3-large instead of small for better quality
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",  # Better quality but more expensive
            input=text,
        )
        return response.data[0].embedding

    def store_email(self, email_data: Dict[str, Any], embedding: List[float]):
        """Store email data and its embedding in Pinecone"""
        # Clean metadata by removing None values
        metadata = {
            "subject": email_data["subject"] or "",  # Convert None to empty string
            "content": email_data["content"] or "",
            "thread_id": email_data["thread_id"] or "",
            "sender": email_data["sender"] or "",
        }

        self.index.upsert(
            vectors=[
                {
                    "id": email_data["message_id"],
                    "values": embedding,
                    "metadata": metadata,
                }
            ],
            namespace=CONFIG["pinecone"].NAMESPACE,
        )

    def weighted_similarity_search(
        self, subject_embedding: List[float], content_embedding: List[float]
    ) -> List[Dict[str, Any]]:
        """Perform weighted similarity search using Pinecone"""
        print("\n   Performing weighted similarity search...")

        # Combine embeddings with weights
        combined_embedding = [
            s * CONFIG["search"].SUBJECT_WEIGHT + c * CONFIG["search"].CONTENT_WEIGHT
            for s, c in zip(subject_embedding, content_embedding)
        ]

        print("   Querying Pinecone...")
        results = self.index.query(
            namespace=CONFIG["pinecone"].NAMESPACE,
            vector=combined_embedding,
            top_k=CONFIG["search"].TOP_K * 2,  # Get more results for deduplication
            include_metadata=True,
        )

        # Deduplicate results
        seen_contents = set()
        unique_results = []

        for match in results.matches:
            content = match.metadata.get("content", "").strip()
            content_hash = hash(content)  # Use hash for better comparison

            if content_hash not in seen_contents and content:  # Skip empty content
                seen_contents.add(content_hash)
                unique_results.append(match)
                if len(unique_results) >= CONFIG["search"].TOP_K:
                    break

        # Log the unique results for debugging
        print("\n   Similar emails found:")
        for i, match in enumerate(unique_results, 1):
            print(f"\n   {i}. Score: {match.score:.3f}")
            print(f"      Subject: {match.metadata['subject'][:100]}")
            print(f"      Content: {match.metadata['content'][:200]}...")

        similar_emails = [match.metadata for match in unique_results]
        print(f"\n   Total similar emails found: {len(similar_emails)}")
        return similar_emails

    def _preprocess_text(self, text: str) -> str:
        """Enhance text for better embedding quality"""
        # Add key information extraction
        prompt = f"""
        Extract key information from this text, focusing on:
        - Main topic/intent
        - Key requirements/skills
        - Action items/requests
        Text: {text}
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        enhanced_text = response.choices[0].message.content
        return f"{text}\n\nKey Information:\n{enhanced_text}"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using GPT"""
        prompt = f"""
        Extract key job-related terms from this text as a comma-separated list.
        Focus on:
        - Job titles
        - Skills
        - Technologies
        - Experience levels
        Text: {text}
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        keywords = response.choices[0].message.content.split(",")
        return [k.strip().lower() for k in keywords]

    def _calculate_keyword_score(
        self, metadata: Dict[str, str], keywords: List[str]
    ) -> float:
        """Calculate keyword match score"""
        text = f"{metadata['subject']} {metadata['content']}".lower()
        matches = sum(1 for keyword in keywords if keyword in text)
        return matches / len(keywords) if keywords else 0.0
