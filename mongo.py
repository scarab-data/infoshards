"""
MongoDB connection for vector storage and retrieval in Vextra
Uses MongoDB Atlas Vector Search for efficient similarity queries
"""

# pylint: disable=bare-except,broad-exception-caught
# pylint: disable=logging-fstring-interpolation
# pylint: disable=redefined-outer-name
# pylint: disable=line-too-long
# pylint: disable=protected-access
# pylint: disable=invalid-name


import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")
MONGO_TIMEOUT = 5000
VECTOR_DIMENSIONS = 768


class VectorDB:
    """MongoDB-based vector database using Atlas Vector Search"""

    def __init__(
        self,
        uri=MONGO_URI,
        db_name=MONGO_DB,
        vector_dimensions=VECTOR_DIMENSIONS,
    ):
        """Initialize MongoDB connection"""
        self.uri = uri
        self.db_name = db_name
        self.vector_dimensions = vector_dimensions
        self.client = None
        self.db = None
        self.connected = False

        # Collections
        self.qa_collection = "vector_qa"  # Collection for Q&A pairs with vectors
        self.data_collection = "data_sources"  # Collection for data source metadata

        # Vector search index names
        self.qa_index_name = "vector_qa_index"

    def connect(self) -> bool:
        """Establish connection to MongoDB"""
        try:
            # Create a MongoDB client with the recommended approach
            self.client = MongoClient(self.uri, server_api=ServerApi("1"))

            # Verify connection by issuing a ping command
            self.client.admin.command("ping")

            # Get database
            self.db = self.client[self.db_name]

            # Create vector search index if it doesn't exist
            self._ensure_vector_search_index()

            self.connected = True
            logger.info(f"Connected to MongoDB: {self.uri}, database: {self.db_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.connected = False
            return False

    def _ensure_vector_search_index(self):
        """Create vector search index if it doesn't exist"""
        try:
            # Check if index already exists
            existing_indexes = list(self.db[self.qa_collection].list_search_indexes())
            index_exists = any(
                idx.get("name") == self.qa_index_name for idx in existing_indexes
            )

            if not index_exists:
                logger.info(f"Creating vector search index '{self.qa_index_name}'")

                # Create search index model for vector search
                search_index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "numDimensions": self.vector_dimensions,
                                "path": "vector",
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "question_text"},
                        ]
                    },
                    name=self.qa_index_name,
                    type="vectorSearch",
                )

                # Create the search index
                result = self.db[self.qa_collection].create_search_index(
                    model=search_index_model
                )
                logger.info(f"New search index named '{result}' is building")

                # Wait for the index to be ready
                logger.info("Waiting for index to be ready...")
                self._wait_for_index_ready(result)
                logger.info(f"Vector search index '{result}' is ready")
            else:
                logger.info(
                    f"Vector search index '{self.qa_index_name}' already exists"
                )

        except Exception as e:
            logger.error(f"Error creating vector search index: {str(e)}")

    def _wait_for_index_ready(self, index_name, max_wait_time=60):
        """Wait for the index to be ready for querying"""
        start_time = time.time()
        while True:
            if time.time() - start_time > max_wait_time:
                logger.warning(
                    f"Timed out waiting for index '{index_name}' to be ready"
                )
                break

            indices = list(self.db[self.qa_collection].list_search_indexes(index_name))
            if indices and indices[0].get("queryable") is True:
                break

            time.sleep(2)

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("MongoDB connection closed")

    def store_qa_vector(
        self,
        question_text: str,
        question_vector: List[float],
        answer_text: str,
        visualization_base64: Optional[str] = None,
        visualization_type: Optional[str] = "image/png",
        processed_data: Optional[str] = None,
        data_source: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Store a question-answer pair with the question's vector representation

        Args:
            question_text: The original question text
            question_vector: Vector representation of the question
            answer_text: The answer text
            visualization_base64: Base64 encoded visualization image
            visualization_type: MIME type of the visualization (e.g., 'image/png')
            processed_data: Processed data as CSV string
            data_source: Reference to the data source used
            metadata: Additional metadata about this Q&A pair

        Returns:
            str: ID of the inserted document or None if failed
        """
        if not self.connected and not self.connect():
            logger.error("Cannot store vector: not connected to MongoDB")
            return None

        # Convert numpy array to list if needed
        if isinstance(question_vector, np.ndarray):
            question_vector = question_vector.tolist()

        # Prepare document
        doc = {
            "question_text": question_text,
            "vector": question_vector,
            "answer_text": answer_text,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        # Add optional fields
        if visualization_base64:
            doc["visualization_base64"] = visualization_base64
            doc["visualization_type"] = visualization_type
        if processed_data:
            doc["processed_data"] = processed_data
        if data_source:
            doc["data_source"] = data_source
        if metadata:
            doc["metadata"] = metadata

        try:
            # Check if similar question already exists
            existing = self.db[self.qa_collection].find_one(
                {"question_text": question_text}
            )
            if existing:
                # Update existing document
                update_fields = {
                    "vector": question_vector,
                    "answer_text": answer_text,
                    "updated_at": datetime.utcnow(),
                }

                # Add optional fields to update
                if visualization_base64:
                    update_fields["visualization_base64"] = visualization_base64
                    update_fields["visualization_type"] = visualization_type
                if processed_data:
                    update_fields["processed_data"] = processed_data
                if data_source:
                    update_fields["data_source"] = data_source
                if metadata:
                    update_fields["metadata"] = metadata

                result = self.db[self.qa_collection].update_one(
                    {"_id": existing["_id"]}, {"$set": update_fields}
                )
                logger.info(f"Updated existing Q&A document with ID {existing['_id']}")
                return str(existing["_id"])
            else:
                # Insert new document
                result = self.db[self.qa_collection].insert_one(doc)
                logger.info(f"Inserted new Q&A document with ID {result.inserted_id}")
                return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error storing Q&A vector: {str(e)}")
            return None

    def find_similar_question(
        self,
        query_vector: List[float],
        similarity_threshold: float = 0.8,
        max_results: int = 1,
    ) -> List[Dict]:
        """
        Find similar questions using MongoDB's vector search

        Args:
            query_vector: Vector representation of the query
            similarity_threshold: Minimum similarity score (0-1) to consider a match
            max_results: Maximum number of results to return

        Returns:
            List of matching documents with similarity scores
        """
        if not self.connected and not self.connect():
            logger.error("Cannot search vectors: not connected to MongoDB")
            return []

        # Convert numpy array to list if needed
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        try:
            # Prepare vector search query
            vector_search_query = {
                "index": self.qa_index_name,
                "queryVector": query_vector,  # Changed from "vector" to "queryVector"
                "path": "vector",
                "numCandidates": max_results
                * 10,  # Retrieve more candidates for filtering
                "limit": max_results,
            }

            # Execute vector search
            search_results = self.db[self.qa_collection].aggregate(
                [
                    {"$vectorSearch": vector_search_query},
                    {
                        "$project": {
                            "_id": 1,
                            "question_text": 1,
                            "answer_text": 1,
                            "visualization_base64": 1,
                            "visualization_type": 1,
                            "processed_data": 1,
                            "data_source": 1,
                            "score": {"$meta": "vectorSearchScore"},
                        }
                    },
                ]
            )

            # Process results
            results = []
            for result in search_results:
                # MongoDB returns a score between 0 and 1 for cosine similarity
                # For other similarity metrics, you may need to transform the score
                similarity = result["score"]

                # Filter by similarity threshold
                if similarity >= similarity_threshold:
                    # Convert ObjectId to string for JSON serialization
                    result["_id"] = str(result["_id"])
                    result["similarity"] = similarity
                    results.append(result)

            logger.info(
                f"Found {len(results)} similar questions above threshold {similarity_threshold}"
            )
            return results

        except Exception as e:
            logger.error(f"Error finding similar questions: {str(e)}")
            return []

    def store_data_source(
        self,
        source_name: str,
        source_type: str,
        file_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Store information about a data source

        Args:
            source_name: Name of the data source
            source_type: Type of data source (e.g., 'csv', 'txt', 'database')
            file_path: Path to the data file if applicable
            metadata: Additional metadata about the data source

        Returns:
            str: ID of the inserted document or None if failed
        """
        if not self.connected and not self.connect():
            logger.error("Cannot store data source: not connected to MongoDB")
            return None

        try:
            # Check if source already exists
            existing = self.db[self.data_collection].find_one(
                {"source_name": source_name}
            )

            doc = {
                "source_name": source_name,
                "source_type": source_type,
                "updated_at": datetime.utcnow(),
            }

            # Add optional fields
            if file_path:
                doc["file_path"] = file_path
            if metadata:
                doc["metadata"] = metadata

            if existing:
                # Update existing document
                result = self.db[self.data_collection].update_one(
                    {"_id": existing["_id"]}, {"$set": doc}
                )
                logger.info(f"Updated existing data source with ID {existing['_id']}")
                return str(existing["_id"])
            else:
                # Add created_at for new documents
                doc["created_at"] = datetime.utcnow()

                # Insert new document
                result = self.db[self.data_collection].insert_one(doc)
                logger.info(f"Inserted new data source with ID {result.inserted_id}")
                return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error storing data source: {str(e)}")
            return None

    def get_data_sources(self) -> List[Dict]:
        """
        Get all data sources

        Returns:
            List of data source documents
        """
        if not self.connected and not self.connect():
            logger.error("Cannot get data sources: not connected to MongoDB")
            return []

        try:
            sources = list(self.db[self.data_collection].find())

            # Convert ObjectId to string for JSON serialization
            for source in sources:
                source["_id"] = str(source["_id"])

            return sources

        except Exception as e:
            logger.error(f"Error getting data sources: {str(e)}")
            return []


# Create a singleton instance
vector_db = VectorDB()


# Function to get the VectorDB instance
def get_vector_db():
    """Get the VectorDB instance"""
    if not vector_db.connected:
        vector_db.connect()
    return vector_db


# Test connection if run directly
if __name__ == "__main__":
    db = get_vector_db()
    if db.connected:
        print("Successfully connected to MongoDB")

        # Test vector storage
        test_vector = [
            0.1
        ] * VECTOR_DIMENSIONS  # Create a test vector with the right dimensions
        doc_id = db.store_qa_vector(
            question_text="What is the average temperature?",
            question_vector=test_vector,
            answer_text="The average temperature is 72°F.",
            visualization_base64="SGVsbG8gV29ybGQ=",  # Base64 encoded "Hello World"
            visualization_type="image/png",
            data_source="temperature_data.csv",
        )

        print(f"Stored Q&A vector with ID: {doc_id}")

        # Test vector search
        similar = db.find_similar_question(test_vector)
        print(f"Found {len(similar)} similar questions:")
        for result in similar:
            print(f"  Question: {result['question_text']}")
            print(f"  Similarity: {result['similarity']:.4f}")

        # Close connection
        db.close()
    else:
        print("Failed to connect to MongoDB")
