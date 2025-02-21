from dataclasses import dataclass
from typing import Dict, List

@dataclass
class EmailConfig:
    IMAP_SERVER: str = "imap.gmail.com"
    IMAP_PORT: int = 993
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    EMAIL: str = "your-email@gmail.com"
    PASSWORD: str = "your-app-specific-password"

@dataclass
class PineconeConfig:
    API_KEY: str = "your-pinecone-api-key"
    ENVIRONMENT: str = "your-environment"
    INDEX_NAME: str = "email-embeddings"
    NAMESPACE: str = "default"
    DIMENSION: int = 3072  # Updated for text-embedding-3-large
    METRIC: str = "cosine"

@dataclass
class LLMConfig:
    MODEL_NAME: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500

@dataclass
class WeightedSearchConfig:
    SUBJECT_WEIGHT: float = 0.4
    CONTENT_WEIGHT: float = 0.6
    TOP_K: int = 3

CONFIG = {
    "email": EmailConfig(),
    "pinecone": PineconeConfig(),
    "llm": LLMConfig(),
    "search": WeightedSearchConfig()
}
