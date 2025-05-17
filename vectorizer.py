"""
Text vectorization utilities for Vextra
Converts text to vector embeddings for semantic search using Google's Generative AI
"""

# pylint: disable=bare-except,broad-exception-caught
# pylint: disable=logging-fstring-interpolation
# pylint: disable=redefined-outer-name
# pylint: disable=line-too-long
# pylint: disable=protected-access
# pylint: disable=invalid-name

import os
import logging
from typing import List, Optional
import numpy as np
import google.genai as genai
from google.genai.types import EmbedContentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default model settings
GOOGLE_EMBEDDING_MODEL = "text-embedding-005"
VECTOR_DIMENSIONS = 768
EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY"

# Environment variables for Vertex AI
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
GOOGLE_GENAI_USE_VERTEXAI = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")


class Vectorizer:
    """Text to vector embedding converter using Google's Generative AI"""

    def __init__(self, model_name=GOOGLE_EMBEDDING_MODEL, dimensions=VECTOR_DIMENSIONS):
        """Initialize the vectorizer with Google's Generative AI"""
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = None
        self.initialized = False

        # Initialize the embedding model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Google Generative AI client"""
        try:
            logger.info(
                f"Initializing Google Generative AI client with model: {self.model_name}"
            )

            self.client = genai.Client()

            self.initialized = True
            logger.info("Google Generative AI client initialized successfully")
        except ImportError:
            logger.error(
                "Could not import Google Generative AI. Install with: pip install google-generativeai"
            )
            self.initialized = False
        except Exception as e:
            logger.error(f"Error initializing Google Generative AI client: {str(e)}")
            self.initialized = False

    def get_embedding(
        self, text: str, task_type=EMBEDDING_TASK_TYPE
    ) -> Optional[List[float]]:
        """
        Convert text to vector embedding using Google's Generative AI

        Args:
            text: The text to convert to a vector
            task_type: The type of embedding task (RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT)

        Returns:
            List of floats representing the text embedding or None if failed
        """
        if not self.initialized:
            logger.error("Vectorizer not properly initialized")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        try:
            # Clean and prepare text
            text = text.strip()

            # Create embedding request
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimensions,
                ),
            )

            # Extract embedding values
            if response and response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values
            else:
                logger.error("No embeddings returned from Google Generative AI")
                return None

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def get_batch_embeddings(
        self, texts: List[str], task_type=EMBEDDING_TASK_TYPE
    ) -> List[Optional[List[float]]]:
        """
        Convert multiple texts to vector embeddings

        Args:
            texts: List of texts to convert to vectors
            task_type: The type of embedding task (RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT)

        Returns:
            List of embeddings (each is a list of floats) or None for failed items
        """
        if not self.initialized:
            logger.error("Vectorizer not properly initialized")
            return [None] * len(texts)

        if not texts:
            logger.warning("Empty list provided for batch embedding")
            return []

        try:
            # Clean and prepare texts
            cleaned_texts = [text.strip() for text in texts if text and text.strip()]

            if not cleaned_texts:
                logger.warning("No valid texts after cleaning")
                return [None] * len(texts)

            # Create batch embedding request
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=cleaned_texts,
                config=EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimensions,
                ),
            )

            # Extract embedding values
            if response and response.embeddings:
                return [embedding.values for embedding in response.embeddings]
            else:
                logger.error("No embeddings returned from Google Generative AI")
                return [None] * len(texts)

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [None] * len(texts)

    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding vectors"""
        return self.dimensions

    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity (0-1)
        """
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0

        return dot_product / (norm_vec1 * norm_vec2)


# Create a singleton instance
vectorizer = Vectorizer()


# Function to get the Vectorizer instance
def get_vectorizer():
    """Get the Vectorizer instance"""
    if not vectorizer.initialized:
        vectorizer._initialize_model()
    return vectorizer


# Test vectorization if run directly
if __name__ == "__main__":
    vec = get_vectorizer()

    if vec.initialized:
        print(
            f"Vectorizer initialized with Google Generative AI model: {vec.model_name}"
        )
        print(f"Embedding dimensions: {vec.get_dimensions()}")

        # Test single embedding
        test_text = "What is the average temperature in New York?"
        embedding = vec.get_embedding(test_text)
        if embedding:
            print(f"Generated embedding with {len(embedding)} dimensions")
            print(f"First 5 values: {embedding[:5]}")

            # Test similarity
            similar_text = "What's the typical temperature in NYC?"
            different_text = "What is the population of Tokyo?"

            similar_embedding = vec.get_embedding(similar_text)
            different_embedding = vec.get_embedding(different_text)

            if similar_embedding and different_embedding:
                similarity_score = vec.calculate_similarity(
                    embedding, similar_embedding
                )
                difference_score = vec.calculate_similarity(
                    embedding, different_embedding
                )

                print(f"Similarity with similar question: {similarity_score:.4f}")
                print(f"Similarity with different question: {difference_score:.4f}")

            # Test batch embedding
            batch_texts = [
                "What is the average temperature in New York?",
                "What's the typical temperature in NYC?",
                "What is the population of Tokyo?",
            ]
            batch_embeddings = vec.get_batch_embeddings(batch_texts)
            if batch_embeddings and all(batch_embeddings):
                print(f"Generated {len(batch_embeddings)} batch embeddings")
        else:
            print("Failed to generate embedding")
    else:
        print("Vectorizer initialization failed")
