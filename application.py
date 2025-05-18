"""Vextra Flask App"""

# pylint: disable=bare-except,broad-exception-caught
# pylint: disable=logging-fstring-interpolation
# pylint: disable=redefined-outer-name
# pylint: disable=line-too-long
# pylint: disable=protected-access
# pylint: disable=invalid-name
# pylint: disable=global-statement
# pylint: disable=subprocess-run-check
# pylint: disable=unused-import

import os
import re
import sys
from datetime import datetime
from typing import Dict, Tuple
import json
import logging
import tempfile
import shutil
import subprocess
import threading
import time
import base64
from bson.objectid import ObjectId
import pandas as pd

from flask import (
    Flask,
    jsonify,
    request,
    render_template,
)

# Import our modules
from data import get_data_loader
from vectorizer import get_vectorizer
from mongo import get_vector_db
from gen import get_genai_client

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
template_dir = os.path.abspath("templates")
static_dir = os.path.abspath("static")
app = application = Flask(
    __name__, template_folder=template_dir, static_folder=static_dir
)

# Constants
SIMILARITY_THRESHOLD = 0.99  # Minimum similarity score to consider a match
REQUEST_TIMEOUT = 180  # 3 minutes timeout for the entire request
VISUALIZATION_TIMEOUT = 60  # 1 minute timeout for visualization generation


def generate_visualization(data_subset, question: str) -> Tuple[str, str, str, str]:
    """
    Generate a visualization for the data subset based on the question and save the processed data

    Args:
        data_subset: DataFrame subset to visualize
        question: The question being asked

    Returns:
        Tuple of (visualization_base64, visualization_type, visualization_code, processed_data_csv)
    """
    logger.info(f"Starting visualization generation for question: {question}")
    start_time = time.time()

    # Get GenAI client
    client = get_genai_client()

    # Clean the data before processing
    # Replace NaN values with appropriate defaults or drop them
    data_subset = data_subset.copy()

    # Log data types and NaN counts to help with debugging
    logger.info("Data types and NaN counts before cleaning:")
    for col in data_subset.columns:
        nan_count = data_subset[col].isna().sum()
        logger.info(
            f"Column '{col}': type={data_subset[col].dtype}, NaN count={nan_count}"
        )

    # Instead of converting to CSV string, we'll work directly with the DataFrame
    logger.info(
        f"Working with data subset of {len(data_subset)} rows and {len(data_subset.columns)} columns"
    )

    # Get column information for the model
    columns_info = "\n".join(
        [
            f"- {col}: {data_subset[col].dtype} (sample values: {', '.join(map(str, data_subset[col].head(3).tolist()))})"
            for col in data_subset.columns
        ]
    )

    # Get some basic statistics that might be helpful
    stats_info = ""
    try:
        numeric_cols = data_subset.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            stats_info += f"- {col}: min={data_subset[col].min()}, max={data_subset[col].max()}, mean={data_subset[col].mean():.2f}\n"
    except Exception as e:
        logger.warning(f"Could not generate statistics: {str(e)}")

    # Create a temporary directory for our files
    temp_dir = tempfile.mkdtemp()
    processed_data_csv = ""

    try:
        # Save the DataFrame to a CSV file with proper quoting to avoid issues
        csv_path = os.path.join(temp_dir, "data.csv")
        data_subset.to_csv(
            csv_path, index=False, quoting=1
        )  # QUOTE_ALL to ensure proper escaping
        logger.info(f"Saved data to CSV file: {csv_path}")

        # Create a small sample file for verification
        sample_path = os.path.join(temp_dir, "sample.csv")
        data_subset.head(10).to_csv(sample_path, index=False)

        # Create prompt for visualization generation with data export and base64 encoding
        # Note the enhanced instructions about NaN values
        prompt = f"""
        Create a Python visualization script using matplotlib or seaborn to answer the following question:
        
        QUESTION: {question}
        
        The data is available in a CSV file named 'data.csv'. Here's information about the dataset:
        
        Number of rows: {len(data_subset)}
        Columns:
        {columns_info}
        
        Key statistics:
        {stats_info}
        
        Your task is to:
        1. Load the data from 'data.csv' using pandas
        2. Process the data correctly to answer the question
        3. Create an appropriate visualization
        4. Save the plot as a PNG file
        5. IMPORTANT: Also save the processed data (after any filtering, grouping, or calculations) to a file named 'processed_data.csv'
        
        Important requirements:
        - Use pandas to read and process the data
        - CRITICAL: The data contains NaN values that must be handled properly before any type conversion
        - ALWAYS check for and handle NaN values BEFORE converting columns to integers or other types
        - For any column that needs to be converted to integers, first use dropna() or fillna() methods
        - Example of safe conversion: df = df.dropna(subset=['Year']); df['Year'] = df['Year'].fillna(0).astype(int)
        - Make sure to handle data types correctly (convert strings to numbers if needed, but check for NaN values first)
        - Create a clear, professional visualization with proper labels and title
        - Use plt.savefig('output.png', format='png', dpi=300, bbox_inches='tight')
        - Save the processed data that you actually visualize to 'processed_data.csv' using df.to_csv('processed_data.csv', index=False)
        - Include a comment at the end of your code that explains the key findings from the visualization in 3-5 bullet points
        - Do NOT use plt.show()
        
        Return ONLY the Python code without any explanation or markdown formatting.
        """

        # Generate visualization code
        logger.info("Generating visualization code using GenAI")
        system_instruction = """
        You are a data visualization expert specializing in Python data analysis.
        
        When creating visualizations:
        1. Always use pandas to read CSV files and handle data properly
        2. Pay careful attention to data types - convert strings to appropriate types when needed
        3. CRITICAL: ALWAYS check for and handle NaN values BEFORE converting data types
        4. For any column that needs to be converted to integers, first use dropna() or fillna() methods
        5. Example of safe conversion: df = df.dropna(subset=['Year']); df['Year'] = df['Year'].fillna(0).astype(int)
        6. For questions about counts or totals, make sure to use appropriate aggregation functions
        7. Create clear, professional visualizations with proper labels
        8. ALWAYS save the processed data that is actually visualized to 'processed_data.csv'
        9. Include a comment at the end with key findings from the visualization
        10. Return only executable Python code without any markdown formatting or explanations
        """

        gen_start_time = time.time()
        visualization_code_raw = client.generate_content(
            prompt=prompt, system_instruction=system_instruction, temperature=0.2
        )
        gen_time = time.time() - gen_start_time
        logger.info(
            f"Visualization code generation completed in {gen_time:.2f} seconds"
        )

        # Clean up the code - remove markdown formatting if present
        visualization_code = visualization_code_raw

        # Remove markdown code blocks if present
        if isinstance(visualization_code, str):
            if visualization_code.startswith("```python"):
                visualization_code = visualization_code.replace("```python", "", 1)
            if visualization_code.endswith("```"):
                visualization_code = visualization_code[:-3]
            visualization_code = visualization_code.strip()

        logger.info(f"Visualization code length: {len(visualization_code)} characters")

        # Execute the visualization code in the temporary directory
        try:
            # Save the code to a file
            code_file = os.path.join(temp_dir, "viz_script.py")
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(visualization_code)
            logger.info(f"Visualization code saved to file: {code_file}")

            # Add debugging to see what's in the data file
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_head = f.read(500)  # Read first 500 chars
            logger.debug(f"CSV file head: {csv_head}")

            # Execute the code with timeout
            logger.info(
                f"Executing visualization code with {VISUALIZATION_TIMEOUT}s timeout"
            )
            exec_start_time = time.time()

            # Create a more detailed environment for the subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(sys.path)

            result = subprocess.run(
                ["python", code_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=VISUALIZATION_TIMEOUT,
                env=env,
            )
            exec_time = time.time() - exec_start_time
            logger.info(
                f"Visualization script execution completed in {exec_time:.2f} seconds"
            )

            # Log stdout for debugging
            if result.stdout:
                logger.debug(f"Visualization script stdout: {result.stdout}")

            if result.returncode != 0:
                logger.error(f"Visualization script error: {result.stderr}")

                # Try to fix common issues and retry
                fixed_code = visualization_code

                # Fix for NaN values in Year column
                if (
                    "Cannot convert non-finite values (NA or inf) to integer"
                    in result.stderr
                ):
                    logger.info("Attempting to fix NaN values issue and retry")

                    # Add a more comprehensive fix at the beginning of the script
                    # This will handle NaN values for any column being converted to int
                    nan_fix = """
# Add comprehensive NaN handling at the beginning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to safely convert columns to int
def safe_int_convert(df, column):
    if column in df.columns:
        # First drop rows where the column is NaN if it's a critical column
        df = df.dropna(subset=[column])
        # Then convert to int
        df[column] = df[column].astype(int)
    return df

"""
                    # Insert the helper function at the beginning of the script
                    fixed_code = nan_fix + fixed_code

                    # Find patterns like df['Column'].astype(int) or data['Column'].astype(int)
                    int_conversions = re.findall(
                        r"(\w+\['[^']+'\])\.astype\(int\)", fixed_code
                    )

                    for match in int_conversions:
                        var_name = match.split("[")[0]  # Extract df or data
                        col_name = match.split("[")[1].split("]")[0]  # Extract 'Column'

                        # Replace with safe conversion
                        old_code = f"{match}.astype(int)"
                        new_code = f"# Safe conversion\n{var_name} = safe_int_convert({var_name}, {col_name})"

                        fixed_code = fixed_code.replace(old_code, new_code)

                    logger.info("Applied comprehensive NaN handling fix")

                # Fix path issue
                elif "No such file or directory: 'data.csv'" in result.stderr:
                    fixed_code = visualization_code.replace(
                        "'data.csv'", f"'{csv_path}'"
                    )
                    logger.info("Attempting to fix file path issue and retry")

                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(fixed_code)

                # Retry execution
                result = subprocess.run(
                    ["python", code_file],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=VISUALIZATION_TIMEOUT,
                    env=env,
                )

                if result.returncode != 0:
                    logger.error(f"Retry failed: {result.stderr}")

                    # If still failing, try a more aggressive approach
                    if (
                        "Cannot convert non-finite values" in result.stderr
                        or "Year" in result.stderr
                    ):
                        logger.info(
                            "Attempting more aggressive fix for numeric conversion issues"
                        )

                        # Even more aggressive fix - add at the beginning of the script
                        aggressive_fix = """
# Add very aggressive NaN handling at the beginning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('data.csv')

# Handle all numeric columns that might need conversion
for col in df.columns:
    # Try to convert to numeric first
    df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # Check if this column contains year-like values
    if 'year' in col.lower() or 'date' in col.lower():
        # For year columns, drop NaN values and convert to int
        df = df.dropna(subset=[col])
        # Convert to string first to handle any remaining issues
        df[col] = df[col].astype(str).replace('nan', '0').replace('NaN', '0').replace('None', '0')
        # Then convert to int if it looks like a year
        try:
            df[col] = df[col].astype(int)
        except:
            # If conversion fails, keep as string
            pass

"""
                        # Replace everything up to the first plot-related code
                        # Find the first occurrence of plt or sns
                        plot_index = fixed_code.find("plt.")
                        if plot_index == -1:
                            plot_index = fixed_code.find("sns.")

                        if plot_index != -1:
                            # Keep only the plotting part of the code
                            plotting_code = fixed_code[plot_index:]
                            # Create new script with aggressive fix + plotting code
                            fixed_code = (
                                aggressive_fix
                                + "\n# Original plotting code\n"
                                + plotting_code
                            )
                        else:
                            # If no plotting code found, just add the fix at the beginning
                            fixed_code = aggressive_fix + fixed_code

                        with open(code_file, "w", encoding="utf-8") as f:
                            f.write(fixed_code)

                        # Try one more time
                        result = subprocess.run(
                            ["python", code_file],
                            cwd=temp_dir,
                            capture_output=True,
                            text=True,
                            timeout=VISUALIZATION_TIMEOUT,
                            env=env,
                        )

                        if result.returncode != 0:
                            logger.error(f"All fixes failed: {result.stderr}")
                            return "", "image/png", visualization_code, ""
                    else:
                        return "", "image/png", visualization_code, ""

            # Check if output.png exists and convert to base64
            output_png = os.path.join(temp_dir, "output.png")
            processed_data_path = os.path.join(temp_dir, "processed_data.csv")

            visualization_base64 = ""
            visualization_type = "image/png"

            if os.path.exists(output_png):
                # Read the PNG file and convert to base64
                with open(output_png, "rb") as img_file:
                    visualization_base64 = base64.b64encode(img_file.read()).decode(
                        "utf-8"
                    )
                logger.info(
                    f"Visualization converted to base64 ({len(visualization_base64)} bytes)"
                )

                # Check if processed data was saved
                if os.path.exists(processed_data_path):
                    # Read the processed data as CSV string
                    with open(processed_data_path, "r", encoding="utf-8") as data_file:
                        processed_data_csv = data_file.read()
                    logger.info(
                        f"Processed data read ({len(processed_data_csv)} bytes)"
                    )
                else:
                    logger.warning("Processed data file was not generated")

                total_time = time.time() - start_time
                logger.info(
                    f"Total visualization generation time: {total_time:.2f} seconds"
                )
                return (
                    visualization_base64,
                    visualization_type,
                    visualization_code,
                    processed_data_csv,
                )
            else:
                logger.error("Visualization script did not generate output.png")
                return "", "image/png", visualization_code, ""

        except subprocess.TimeoutExpired:
            logger.error(
                f"Visualization script timed out after {VISUALIZATION_TIMEOUT} seconds"
            )
            return "", "image/png", visualization_code, ""
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return "", "image/png", visualization_code, ""

    finally:
        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory: {str(e)}")


def process_question(question: str) -> Dict:
    """
    Process a question about the data

    Args:
        question: The question to process

    Returns:
        Dict containing the response
    """
    logger.info(f"Processing question: {question}")
    start_time = time.time()

    # Initialize components
    logger.info("Initializing components (vectorizer, DB, data loader, GenAI)")
    vectorizer = get_vectorizer()
    vector_db = get_vector_db()
    data_loader = get_data_loader()
    genai_client = get_genai_client()

    # Load the data if not already loaded
    if not data_loader.loaded:
        logger.info("Data not loaded, loading now...")
        load_start = time.time()
        data_loader.load_data()
        load_time = time.time() - load_start
        logger.info(f"Data loaded in {load_time:.2f} seconds")

    # Vectorize the question
    logger.info("Vectorizing question")
    vec_start = time.time()
    question_vector = vectorizer.get_embedding(question)
    vec_time = time.time() - vec_start
    logger.info(f"Question vectorized in {vec_time:.2f} seconds")

    if not question_vector:
        logger.error("Failed to vectorize question")
        return {
            "success": False,
            "error": "Failed to vectorize question",
            "question": question,
        }

    # Search for similar questions in MongoDB
    logger.info(
        f"Searching for similar questions with threshold {SIMILARITY_THRESHOLD}"
    )
    search_start = time.time()
    similar_questions = vector_db.find_similar_question(
        query_vector=question_vector, similarity_threshold=SIMILARITY_THRESHOLD
    )
    search_time = time.time() - search_start
    logger.info(
        f"Similar question search completed in {search_time:.2f} seconds, found {len(similar_questions)} matches"
    )

    # Find related questions with a lower threshold (regardless of exact match)
    logger.info("Searching for related questions with lower threshold")
    related_start = time.time()

    # If we found an exact match, exclude it from related questions
    exclude_id = similar_questions[0]["_id"] if similar_questions else None

    related_questions = vector_db.find_related_questions(
        query_vector=question_vector,
        similarity_threshold=0.75,  # Lower threshold for related questions
        max_results=5,
        exclude_id=exclude_id,
    )
    related_time = time.time() - related_start
    logger.info(
        f"Related question search completed in {related_time:.2f} seconds, found {len(related_questions)} matches"
    )

    # If we found a similar question with good similarity, return its answer
    if similar_questions:
        best_match = similar_questions[0]
        logger.info(
            f"Found similar question: '{best_match['question_text']}' with similarity {best_match['similarity']:.4f}"
        )

        cache_response_time = time.time() - start_time
        logger.info(f"Returning cached answer in {cache_response_time:.2f} seconds")

        return {
            "success": True,
            "question": question,
            "answer": best_match["answer_text"],
            "visualization_id": best_match["_id"],
            "visualization_type": best_match.get("visualization_type", "image/png"),
            "similarity": best_match["similarity"],
            "similar_question": best_match["question_text"],
            "related_questions": related_questions,
            "source": "cache",
        }

    # No similar question found, generate a new answer
    logger.info(f"No similar question found, generating new answer for: {question}")

    # Get the data - use the full dataset
    data = data_loader.get_data()
    logger.info(
        f"Working with full dataset of {len(data)} rows and {len(data.columns)} columns"
    )

    # Use the full dataset for analysis
    data_subset = data

    # Generate visualization and processed data
    logger.info("Starting visualization generation")
    viz_start = time.time()
    visualization_base64, visualization_type, visualization_code, processed_data_csv = (
        generate_visualization(data_subset, question)
    )
    viz_time = time.time() - viz_start
    logger.info(
        f"Visualization generation completed in {viz_time:.2f} seconds, base64 length: {len(visualization_base64)}"
    )

    # Generate answer using GenAI with the processed data
    logger.info("Generating answer using GenAI with processed data")
    answer_start = time.time()

    # Prepare the prompt with the processed data
    if processed_data_csv:
        # Use the processed data for analysis
        prompt = f"""
        I need to analyze the following processed data to answer a question.
        
        QUESTION: {question}
        
        PROCESSED DATA:
        {processed_data_csv[:5000] if len(processed_data_csv) > 5000 else processed_data_csv}
        
        {f"[Note: Showing only first 5000 characters of {len(processed_data_csv)} total characters]" if len(processed_data_csv) > 5000 else ""}
        
        This data was specifically processed to answer the question. It represents the key information extracted from a larger dataset.
        
        Please provide a comprehensive answer to the question based on this processed data. Include specific numbers, trends, and insights.
        If there are any clear patterns, outliers, or notable findings in the data, be sure to highlight them.
        """

        system_instruction = """
        You are a data analysis expert. Analyze the provided processed data to answer the question accurately.
        Focus on the actual data values provided - do not make assumptions beyond what's shown in the data.
        Include relevant statistics, trends, and insights from the data.
        Be specific about numbers and patterns you observe.
        Organize your answer in a clear, structured way with appropriate paragraphs.
        """
    else:
        # Fallback to using a sample of the original data
        sample_size = min(200, len(data))  # Increased from 100 to 200
        csv_data = data_loader.to_csv_string(
            data_subset.sample(sample_size)
            if len(data_subset) > sample_size
            else data_subset
        )

        # Get metadata for context
        metadata = data_loader.get_metadata()
        context = f"""
        This dataset contains {metadata['num_rows']} rows and {metadata['num_columns']} columns.
        Column names: {', '.join(metadata['columns'])}.
        
        IMPORTANT: The full dataset is being analyzed, but I'm only showing a sample of {sample_size} rows in this prompt.
        Make sure to consider the entire dataset in your analysis, including checking for missing years or data.
        
        If asked about specific years, values, or trends, be sure to check if that data exists in the dataset before answering.
        """

        prompt = f"""
        I need to analyze the following data to answer a question.
        
        QUESTION: {question}
        
        DATA:
        {csv_data}
        
        CONTEXT:
        {context}
        """

        system_instruction = """
        You are a data analysis expert. Analyze the provided data to answer the question accurately.
        Focus on the actual data values provided - do not make assumptions beyond what's shown in the data.
        Include relevant statistics, trends, and insights from the data.
        Be specific about numbers and patterns you observe.
        Organize your answer in a clear, structured way with appropriate paragraphs.
        """

    # Generate the answer
    answer = genai_client.generate_content(
        prompt=prompt, system_instruction=system_instruction, temperature=0.2
    )

    answer_time = time.time() - answer_start
    logger.info(f"Answer generation completed in {answer_time:.2f} seconds")

    # Log a snippet of the answer for debugging
    answer_snippet = answer[:100] + "..." if len(answer) > 100 else answer
    logger.info(f"Answer snippet: {answer_snippet}")

    # Store the Q&A pair with vector in MongoDB for future use
    logger.info("Storing Q&A pair in vector database")
    store_start = time.time()
    doc_id = vector_db.store_qa_vector(
        question_text=question,
        question_vector=question_vector,
        answer_text=answer,
        visualization_base64=visualization_base64,
        visualization_type=visualization_type,
        processed_data=processed_data_csv,
        data_source=data_loader.data_path,
        metadata={
            "generated_at": datetime.now().isoformat(),
            "visualization_code": visualization_code if visualization_code else None,
        },
    )
    store_time = time.time() - store_start
    logger.info(f"Q&A storage completed in {store_time:.2f} seconds, doc_id: {doc_id}")

    if not doc_id:
        logger.warning("Failed to store Q&A vector in MongoDB")

    total_time = time.time() - start_time
    logger.info(f"Total question processing time: {total_time:.2f} seconds")

    return {
        "success": True,
        "question": question,
        "answer": answer,
        "visualization_id": doc_id,
        "visualization_type": visualization_type,
        "related_questions": related_questions,  # Add related questions to the response
        "source": "generated",
    }


@app.route("/", methods=["GET"])
def get_info():
    """Index route for Vextra"""
    logger.info(f"Index route accessed with args: {request.args}")
    if request.args.get("render", "html") == "json":
        data = {
            "status": "Welcome to Vextra",
        }
        return jsonify(data), 200
    return render_template("index.html"), 200


@app.route("/ask", methods=["POST"])
def ask_question():
    """API endpoint to ask a question with a longer timeout"""
    logger.info(f"Ask endpoint accessed with content-type: {request.content_type}")
    data = request.get_json()
    logger.info(f"Request data: {data}")

    if not data or "question" not in data:
        logger.warning("Missing question parameter in request")
        return jsonify({"success": False, "error": "Missing question parameter"}), 400

    question = data["question"].strip()
    if not question:
        logger.warning("Empty question in request")
        return jsonify({"success": False, "error": "Empty question"}), 400

    logger.info(f"Processing question with {REQUEST_TIMEOUT}s timeout: {question}")

    # Process the question with a longer timeout
    try:
        # Set a timeout for the entire process
        result = {}
        processing_error = None

        def process_with_timeout():
            nonlocal result, processing_error
            try:
                result = process_question(question)
            except Exception as e:
                processing_error = str(e)
                logger.error(f"Error in processing thread: {str(e)}")

        # Create and start the thread
        thread = threading.Thread(target=process_with_timeout)
        thread.daemon = True

        start_time = time.time()
        logger.info(f"Starting processing thread at {datetime.now().isoformat()}")
        thread.start()

        # Wait for the thread to complete with a much longer timeout
        thread.join(timeout=REQUEST_TIMEOUT)  # 3 minute timeout for the entire process

        elapsed_time = time.time() - start_time
        logger.info(f"Thread completed or timed out after {elapsed_time:.2f} seconds")

        if thread.is_alive():
            # Thread is still running after timeout
            logger.error(
                f"Processing question timed out after {REQUEST_TIMEOUT} seconds: {question}"
            )
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Request timed out after {REQUEST_TIMEOUT} seconds. The question might be too complex for the current dataset.",
                        "question": question,
                    }
                ),
                408,
            )

        if processing_error:
            # Thread encountered an error
            logger.error(f"Processing thread encountered an error: {processing_error}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"An error occurred while processing your question: {processing_error}",
                        "question": question,
                    }
                ),
                500,
            )

        if not result:
            # Thread completed but returned no result
            logger.error("Processing thread returned no result")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Failed to process question. No result was generated.",
                        "question": question,
                    }
                ),
                500,
            )

        # Thread completed successfully
        logger.info(f"Successfully processed question in {elapsed_time:.2f} seconds")
        return jsonify(result), 200

    except Exception as e:
        # Exception in the main thread
        logger.error(f"Exception in ask_question endpoint: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"An error occurred while handling your request: {str(e)}",
                    "question": question,
                }
            ),
            500,
        )


@app.route("/data/summary", methods=["GET"])
def get_data_summary():
    """Get a summary of the loaded dataset"""
    logger.info("Data summary endpoint accessed")
    data_loader = get_data_loader()

    if not data_loader.loaded:
        logger.info("Data not loaded, loading now...")
        data_loader.load_data()

    if not data_loader.loaded:
        logger.error("Failed to load dataset")
        return jsonify({"success": False, "error": "Failed to load dataset"}), 500

    logger.info("Generating data summary")
    summary = data_loader.get_summary()
    logger.info(
        f"Summary generated with {len(summary.get('statistics', {}))} statistics"
    )

    return jsonify({"success": True, "summary": summary}), 200


@app.route("/data/sample", methods=["GET"])
def get_data_sample():
    """Get a sample of the loaded dataset"""
    logger.info(f"Data sample endpoint accessed with args: {request.args}")
    data_loader = get_data_loader()

    if not data_loader.loaded:
        logger.info("Data not loaded, loading now...")
        data_loader.load_data()

    if not data_loader.loaded:
        logger.error("Failed to load dataset")
        return jsonify({"success": False, "error": "Failed to load dataset"}), 500

    # Get parameters
    rows = request.args.get("rows", 10, type=int)
    format_type = request.args.get("format", "json")
    logger.info(f"Requested {rows} rows in {format_type} format")

    # Get data sample
    data = data_loader.get_data()
    sample = data.head(min(rows, 100))  # Limit to max 100 rows
    logger.info(
        f"Returning sample with {len(sample)} rows and {len(sample.columns)} columns"
    )

    if format_type == "csv":
        csv_data = data_loader.to_csv_string(sample)
        logger.info(f"Returning CSV data ({len(csv_data)} bytes)")
        return csv_data, 200, {"Content-Type": "text/csv"}
    else:
        logger.info("Returning JSON data")
        return (
            jsonify(
                {
                    "success": True,
                    "sample": json.loads(sample.to_json(orient="records")),
                }
            ),
            200,
        )


@app.route("/visualizations/<path:doc_id>")
def get_visualization(doc_id):
    """Serve visualization from MongoDB by document ID"""
    logger.info(f"Visualization requested for document: {doc_id}")

    # Get the vector DB
    vector_db = get_vector_db()

    if not vector_db.connected and not vector_db.connect():
        logger.error("Cannot retrieve visualization: not connected to MongoDB")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # Convert string ID to ObjectId
        obj_id = ObjectId(doc_id)

        # Retrieve the document directly from the collection
        document = vector_db.db[vector_db.qa_collection].find_one({"_id": obj_id})

        if not document:
            logger.error(f"Document not found: {doc_id}")
            return jsonify({"error": "Visualization not found"}), 404

        # Get the base64 encoded visualization and its type
        visualization_base64 = document.get("visualization_base64", "")
        visualization_type = document.get("visualization_type", "image/png")

        if not visualization_base64:
            logger.error(f"No visualization found in document: {doc_id}")
            return jsonify({"error": "No visualization in document"}), 404

        # Decode the base64 data
        try:
            visualization_data = base64.b64decode(visualization_base64)

            # Return the visualization with the appropriate content type
            return visualization_data, 200, {"Content-Type": visualization_type}
        except Exception as e:
            logger.error(f"Error decoding base64 data: {str(e)}")
            return jsonify({"error": "Invalid visualization data"}), 500

    except Exception as e:
        logger.error(f"Error retrieving visualization: {str(e)}")
        return jsonify({"error": "Failed to retrieve visualization"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")

    # Check if components are initialized
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "data_loader": False,
            "vectorizer": False,
            "vector_db": False,
            "genai": False,
        },
    }

    try:
        # Check data loader
        data_loader = get_data_loader()
        health_status["components"]["data_loader"] = data_loader.loaded

        # Check vectorizer
        vectorizer = get_vectorizer()
        health_status["components"]["vectorizer"] = vectorizer.initialized

        # Check vector DB
        vector_db = get_vector_db()
        health_status["components"]["vector_db"] = vector_db.connected

        # Check GenAI
        genai_client = get_genai_client()
        health_status["components"]["genai"] = genai_client.initialized

        # Overall status
        if all(health_status["components"].values()):
            health_status["status"] = "healthy"
        else:
            health_status["status"] = "degraded"
            unhealthy_components = [
                k for k, v in health_status["components"].items() if not v
            ]
            health_status["message"] = (
                f"Unhealthy components: {', '.join(unhealthy_components)}"
            )

        logger.info(f"Health check result: {health_status['status']}")
        return jsonify(health_status), 200

    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        return jsonify(health_status), 500


if __name__ == "__main__":
    # Initialize components
    logger.info("Initializing components before starting server")

    try:
        # Initialize data loader
        logger.info("Initializing data loader")
        data_loader = get_data_loader()
        data_loader.load_data()

        # Initialize vectorizer
        logger.info("Initializing vectorizer")
        vectorizer = get_vectorizer()

        # Initialize vector DB
        logger.info("Initializing vector database")
        vector_db = get_vector_db()

        # Initialize GenAI
        logger.info("Initializing GenAI client")
        genai_client = get_genai_client()

        logger.info("All components initialized, starting server")
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        logger.warning("Starting server with uninitialized components")

    # Run the Flask app
    logger.info("Starting Flask application on 0.0.0.0:8000")
    application.run("0.0.0.0", 8000, debug=True)
