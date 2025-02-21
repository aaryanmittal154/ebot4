from typing import List, Dict, Any
from openai import OpenAI
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from config import CONFIG
import time
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class JobCandidateStore:
    def __init__(self):
        self.openai_client = OpenAI()
        # Load environment variables
        load_dotenv()

        # Get API key from environment
        api_key = os.getenv("PINECONE_API_KEY") or CONFIG["pinecone"].API_KEY
        if not api_key:
            raise ValueError("Pinecone API key not found in environment or config")

        logger.info("Initializing Pinecone GRPC client for job/candidate store...")
        try:
            self.pinecone = PineconeGRPC(api_key=api_key)
            logger.info("✅ Pinecone GRPC client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise

        self.job_index = self._init_index("job-candidates")

    def _init_index(self, index_name: str):
        """Initialize Pinecone index for jobs/candidates"""
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Checking index: {index_name} (attempt {attempt + 1}/{max_retries})"
                )
                self.pinecone.describe_index(index_name)
                logger.info(f"Index {index_name} already exists, connecting...")
                break
            except Exception as e:
                logger.warning(
                    f"Failed to find index on attempt {attempt + 1}: {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.info(f"Creating new index: {index_name}")
                    try:
                        self.pinecone.create_index(
                            name=index_name,
                            dimension=CONFIG["pinecone"].DIMENSION,
                            metric=CONFIG["pinecone"].METRIC,
                            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                        )
                        logger.info("Waiting for index to be ready...")

                        # Wait for index to be ready with timeout
                        timeout = 60  # seconds
                        start_time = time.time()
                        while time.time() - start_time < timeout:
                            try:
                                status = self.pinecone.describe_index(index_name).status
                                if status.ready:
                                    logger.info("\nIndex created successfully!")
                                    break
                            except Exception:
                                pass
                            time.sleep(1)
                            logger.info(".", end="", flush=True)
                        else:
                            raise TimeoutError(
                                f"Index creation timed out after {timeout} seconds"
                            )
                    except Exception as create_error:
                        logger.error(f"Error creating index: {str(create_error)}")
                        raise
                else:
                    time.sleep(retry_delay)
                    continue

        try:
            index = self.pinecone.Index(index_name)
            # Verify connection with a simple operation
            index.describe_index_stats()
            logger.info("Successfully connected to index!")
            return index
        except Exception as e:
            logger.error(f"Failed to connect to index: {str(e)}")
            raise

    def store_job(self, job_data: Dict[str, Any]):
        """Store job posting with its embedding"""
        max_retries = 3
        retry_delay = 2

        job_text = f"""
        Title: {job_data.get('title', '')}
        Company: {job_data.get('company', '')}
        Requirements: {job_data.get('requirements', '')}
        Description: {job_data.get('description', '')}
        """
        embedding = self.generate_embedding(job_text)

        for attempt in range(max_retries):
            try:
                self.job_index.upsert(
                    vectors=[
                        {
                            "id": f"job_{job_data['id']}",
                            "values": embedding,
                            "metadata": {**job_data, "type": "job"},
                        }
                    ],
                    namespace="jobs",
                )
                return
            except Exception as e:
                logger.warning(f"Job upsert attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All job upsert attempts failed")
                    raise
                time.sleep(retry_delay)

    def store_candidate(self, candidate_data: Dict[str, Any]):
        """Store candidate profile with its embedding"""
        max_retries = 3
        retry_delay = 2

        candidate_text = f"""
        Name: {candidate_data.get('name', '')}
        Skills: {candidate_data.get('skills', '')}
        Experience: {candidate_data.get('experience', '')}
        Background: {candidate_data.get('background', '')}
        """
        embedding = self.generate_embedding(candidate_text)

        for attempt in range(max_retries):
            try:
                self.job_index.upsert(
                    vectors=[
                        {
                            "id": f"candidate_{candidate_data['id']}",
                            "values": embedding,
                            "metadata": {**candidate_data, "type": "candidate"},
                        }
                    ],
                    namespace="candidates",
                )
                return
            except Exception as e:
                logger.warning(
                    f"Candidate upsert attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error("All candidate upsert attempts failed")
                    raise
                time.sleep(retry_delay)

    def find_matching_candidates(
        self, job_description: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find candidates matching a job description"""
        embedding = self.generate_embedding(job_description)

        results = self.job_index.query(
            namespace="candidates", vector=embedding, top_k=top_k, include_metadata=True
        )

        return [match.metadata for match in results.matches]

    def find_matching_jobs(
        self, candidate_profile: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find jobs matching a candidate profile"""
        embedding = self.generate_embedding(candidate_profile)

        results = self.job_index.query(
            namespace="jobs", vector=embedding, top_k=top_k, include_metadata=True
        )

        return [match.metadata for match in results.matches]

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI's API"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        return response.data[0].embedding
