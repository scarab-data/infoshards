"""
Google Generative AI (Gemini) client for Vextra
Provides text generation capabilities using Google's Gemini models
"""

# pylint: disable=bare-except,broad-exception-caught
# pylint: disable=logging-fstring-interpolation
# pylint: disable=redefined-outer-name
# pylint: disable=line-too-long
# pylint: disable=protected-access
# pylint: disable=invalid-name

import os
import logging
from typing import Optional
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default model settings
DEFAULT_MODEL = "gemini-2.0-flash-001"  # or "gemini-1.0-pro" for older model
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


class GenAIClient:
    """Client for interacting with Google's Generative AI (Gemini) models"""

    def __init__(self, model_name=DEFAULT_MODEL, api_key=GEMINI_API_KEY):
        """Initialize the Gemini client"""
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        self.initialized = False

        # Initialize the client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Google Generative AI client"""
        if not self.api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            self.initialized = False
            return

        try:
            logger.info(
                f"Initializing Google Generative AI client with model: {self.model_name}"
            )

            # Configure the client with API key
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

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate content using Gemini model

        Args:
            prompt: The user prompt/question
            system_instruction: Optional system instruction to guide the model
            temperature: Controls randomness (0.0 to 1.0)
            max_output_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated text
        """
        if not self.initialized:
            logger.error("GenAI client not properly initialized")
            return ""

        try:
            # Combine system instruction and prompt if provided
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Generate content using the simplest possible approach
            response = self.client.models.generate_content(
                contents=full_prompt,
                model=self.model_name,
            )

            # Return the text response
            return response.text

        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return ""

    def analyze_data(
        self,
        data: str,
        question: str,
        context: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Analyze data with a specific question

        Args:
            data: The data to analyze (CSV, JSON, text)
            question: The question to answer about the data
            context: Additional context about the data
            temperature: Controls randomness (lower for more factual responses)

        Returns:
            Analysis result as text
        """
        prompt = f"I need to analyze the following data to answer a question.\n\nDATA:\n{data}\n\nQUESTION: {question}"

        if context:
            prompt += f"\n\nCONTEXT: {context}"

        system_instruction = "You are a data analysis assistant. Analyze the provided data to answer the question accurately. Include relevant statistics and insights. Be concise but thorough."

        try:
            response = self.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
            )
            
            if not response:
                return "I'm sorry, I couldn't analyze the data to answer your question."
                
            return response
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return "I'm sorry, I encountered an error while analyzing the data."


# Create a singleton instance
genai_client = GenAIClient()


# Function to get the GenAI client instance
def get_genai_client():
    """Get the GenAI client instance"""
    if not genai_client.initialized:
        genai_client._initialize_client()
    return genai_client


# Test the client if run directly
if __name__ == "__main__":
    client = get_genai_client()

    if client.initialized:
        print(f"GenAI client initialized with model: {client.model_name}")

        # Test basic generation
        prompt = "Write a short poem about data analysis"
        print("\nGenerating content for prompt:", prompt)
        response = client.generate_content(prompt)
        print("\nResponse:")
        print(response)

        # Test data analysis
        sample_data = """
        Year,Temperature,Rainfall
        2020,22.1,1200
        2021,22.4,1150
        2022,23.2,980
        2023,23.8,920
        """
        question = "What's the trend in temperature and rainfall over these years?"
        print("\nAnalyzing data with question:", question)
        analysis = client.analyze_data(sample_data, question)
        print("\nAnalysis:")
        print(analysis)
    else:
        print("GenAI client initialization failed")
