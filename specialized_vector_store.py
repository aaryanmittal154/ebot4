from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from config import CONFIG
import time
import logging

logger = logging.getLogger(__name__)


class JobCandidateStore:
    def __init__(self):
        self.openai_client = OpenAI()
        self.pinecone = Pinecone(
            api_key=CONFIG["pinecone"].API_KEY,
            environment=CONFIG["pinecone"].ENVIRONMENT,
        )
        self.job_index = self._init_index("job-candidates")

    def _init_index(self, index_name: str):
        """Initialize Pinecone index for jobs/candidates"""
        try:
            logger.info(f"Checking index: {index_name}")
            self.pinecone.describe_index(index_name)
            logger.info(f"Index {index_name} already exists, connecting...")
        except Exception:
            logger.info(f"Creating new index: {index_name}")
            self.pinecone.create_index(
                name=index_name,
                dimension=CONFIG["pinecone"].DIMENSION,
                metric=CONFIG["pinecone"].METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-west-2"),
            )
            logger.info("Waiting for index to be ready...")
            while not self.pinecone.describe_index(index_name).status["ready"]:
                time.sleep(1)
                logger.info(".", end="", flush=True)
            logger.info("\nIndex created successfully!")

        return self.pinecone.Index(index_name)

    def store_job(self, job_data: Dict[str, Any]):
        """Store job posting with its embedding"""
        job_text = f"""
        Title: {job_data.get('title', '')}
        Company: {job_data.get('company', '')}
        Requirements: {job_data.get('requirements', '')}
        Description: {job_data.get('description', '')}
        """
        embedding = self.generate_embedding(job_text)

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

    def store_candidate(self, candidate_data: Dict[str, Any]):
        """Store candidate profile with its embedding"""
        candidate_text = f"""
        Name: {candidate_data.get('name', '')}
        Skills: {candidate_data.get('skills', '')}
        Experience: {candidate_data.get('experience', '')}
        Background: {candidate_data.get('background', '')}
        """
        embedding = self.generate_embedding(candidate_text)

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
