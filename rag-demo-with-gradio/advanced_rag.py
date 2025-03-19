import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datetime
import functools
import traceback
from typing import List, Optional, Any, Dict

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

# Other LangChain and community imports
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, Document
from langchain_core.runnables import RunnableParallel, RunnableLambda
from transformers.quantizers.auto import AutoQuantizationConfig
import gradio as gr
import requests
from pydantic import PrivateAttr
import pydantic

from langchain.llms.base import LLM
from typing import Any, Optional, List
import typing
import time
import requests
import re


print("Pydantic Version: ")
print(pydantic.__version__)
# Add Mistral imports with fallback handling

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
    debug_print = lambda msg: print(f"[{datetime.datetime.now().isoformat()}] {msg}")
    debug_print("Loaded latest Mistral client library")
except ImportError:
    MISTRAL_AVAILABLE = False
    debug_print = lambda msg: print(f"[{datetime.datetime.now().isoformat()}] {msg}")
    debug_print("Mistral client library not found. Install with: pip install mistralai")

def debug_print(message: str):
    print(f"[{datetime.datetime.now().isoformat()}] {message}", flush=True)

def word_count(text: str) -> int:
    return len(text.split())

# Initialize a tokenizer for token counting (using gpt2 as a generic fallback)
def initialize_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        debug_print("Failed to initialize tokenizer: " + str(e))
        return None

global_tokenizer = initialize_tokenizer()

def count_tokens(text: str) -> int:
    if global_tokenizer:
        try:
            return len(global_tokenizer.encode(text))
        except Exception as e:
            return len(text.split())
    return len(text.split())


# Add these imports at the top of your file
import uuid
import threading
import queue
from typing import Dict, Any, Tuple, Optional
import time

# Global storage for jobs and results
jobs = {}  # Stores job status and results
results_queue = queue.Queue()  # Thread-safe queue for completed jobs
processing_lock = threading.Lock()  # Prevent simultaneous processing of the same job

# Add a global variable to store the last job ID
last_job_id = None

# Add these missing async processing functions

def process_in_background(job_id, function, args):
    """Process a function in the background and store results"""
    try:
        debug_print(f"Processing job {job_id} in background")
        result = function(*args)
        results_queue.put((job_id, result))
        debug_print(f"Job {job_id} completed and added to results queue")
    except Exception as e:
        debug_print(f"Error in background job {job_id}: {str(e)}")
        error_result = (f"Error processing job: {str(e)}", "", "", "")
        results_queue.put((job_id, error_result))

def load_pdfs_async(file_links, model_choice, prompt_template, bm25_weight, temperature, top_p):
    """Asynchronous version of load_pdfs_updated to prevent timeouts"""
    global last_job_id
    if not file_links:
        return "Please enter non-empty URLs", "", "Model used: N/A", "", "", get_job_list()
    
    job_id = str(uuid.uuid4())
    debug_print(f"Starting async job {job_id} for file loading")
    
    # Start background thread
    threading.Thread(
        target=process_in_background,
        args=(job_id, load_pdfs_updated, [file_links, model_choice, prompt_template, bm25_weight, temperature, top_p])
    ).start()
    
    job_query = f"Loading files: {file_links.split()[0]}..." if file_links else "No files"
    jobs[job_id] = {
        "status": "processing", 
        "type": "load_files",
        "start_time": time.time(),
        "query": job_query
    }
    
    last_job_id = job_id
    
    return (
        f"Files submitted and processing in the background (Job ID: {job_id}).\n\n"
        f"Use 'Check Job Status' tab with this ID to get results.",
        f"Job ID: {job_id}",
        f"Model requested: {model_choice}",
        job_id,  # Return job_id to update the job_id_input component
        job_query,  # Return job_query to update the job_query_display component
        get_job_list()  # Return updated job list
    )

def submit_query_async(query, model_choice=None):
    """Asynchronous version of submit_query_updated to prevent timeouts"""
    global last_job_id
    if not query:
        return "Please enter a non-empty query", "", "Input tokens: 0", "Output tokens: 0", "", "", get_job_list()
    
    job_id = str(uuid.uuid4())
    debug_print(f"Starting async job {job_id} for query: {query}")
    
    # Update model if specified
    if model_choice and rag_chain and rag_chain.llm_choice != model_choice:
        debug_print(f"Updating model to {model_choice} for this query")
        rag_chain.update_llm_pipeline(model_choice, rag_chain.temperature, rag_chain.top_p,
                                     rag_chain.prompt_template, rag_chain.bm25_weight)
    
    # Start background thread
    threading.Thread(
        target=process_in_background,
        args=(job_id, submit_query_updated, [query])
    ).start()
    
    jobs[job_id] = {
        "status": "processing", 
        "type": "query",
        "start_time": time.time(),
        "query": query,
        "model": rag_chain.llm_choice if hasattr(rag_chain, 'llm_choice') else "Unknown"
    }
    
    last_job_id = job_id
    
    return (
        f"Query submitted and processing in the background (Job ID: {job_id}).\n\n"
        f"Use 'Check Job Status' tab with this ID to get results.",
        f"Job ID: {job_id}",
        f"Input tokens: {count_tokens(query)}",
        "Output tokens: pending",
        job_id,  # Return job_id to update the job_id_input component
        query,  # Return query to update the job_query_display component
        get_job_list()  # Return updated job list
    )

def update_ui_with_last_job_id():
    # This function doesn't need to do anything anymore
    # We'll update the UI directly in the functions that call this
    pass

# Function to display all jobs as a clickable list
def get_job_list():
    job_list_md = "### Submitted Jobs\n\n"
    
    if not jobs:
        return "No jobs found. Submit a query or load files to create jobs."
    
    # Sort jobs by start time (newest first)
    sorted_jobs = sorted(
        [(job_id, job_info) for job_id, job_info in jobs.items()],
        key=lambda x: x[1].get("start_time", 0),
        reverse=True
    )
    
    for job_id, job_info in sorted_jobs:
        status = job_info.get("status", "unknown")
        job_type = job_info.get("type", "unknown")
        query = job_info.get("query", "")
        start_time = job_info.get("start_time", 0)
        time_str = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a shortened query preview
        query_preview = query[:30] + "..." if query and len(query) > 30 else query or "N/A"
        
        # Add color and icons based on status
        if status == "processing":
            # Red color with processing icon for processing jobs
            status_formatted = f"<span style='color: red'>⏳ {status}</span>"
        elif status == "completed":
            # Green color with checkmark for completed jobs
            status_formatted = f"<span style='color: green'>✅ {status}</span>"
        else:
            # Default formatting for unknown status
            status_formatted = f"<span style='color: orange'>❓ {status}</span>"
        
        # Create clickable links using Markdown
        if job_type == "query":
            job_list_md += f"- [{job_id}](javascript:void) - {time_str} - {status_formatted} - Query: {query_preview}\n"
        else:
            job_list_md += f"- [{job_id}](javascript:void) - {time_str} - {status_formatted} - File Load Job\n"
    
    return job_list_md
    
# Function to handle job list clicks
def job_selected(job_id):
    if job_id in jobs:
        return job_id, jobs[job_id].get("query", "No query for this job")
    return job_id, "Job not found"

# Function to refresh the job list
def refresh_job_list():
    return get_job_list()

# Function to sync model dropdown boxes
def sync_model_dropdown(value):
    return value    

# Function to check job status
def check_job_status(job_id):
    if not job_id:
        return "Please enter a job ID", "", "", "", ""
    
    # Process any completed jobs in the queue
    try:
        while not results_queue.empty():
            completed_id, result = results_queue.get_nowait()
            if completed_id in jobs:
                jobs[completed_id]["status"] = "completed"
                jobs[completed_id]["result"] = result
                jobs[completed_id]["end_time"] = time.time()
                debug_print(f"Job {completed_id} completed and stored in jobs dictionary")
    except queue.Empty:
        pass
    
    # Check if the requested job exists
    if job_id not in jobs:
        return "Job not found. Please check the ID and try again.", "", "", "", ""
    
    job = jobs[job_id]
    job_query = job.get("query", "No query available for this job")
    
    # If job is still processing
    if job["status"] == "processing":
        elapsed_time = time.time() - job["start_time"]
        job_type = job.get("type", "unknown")
        
        if job_type == "load_files":
            return (
                f"Files are still being processed (elapsed: {elapsed_time:.1f}s).\n\n"
                f"Try checking again in a few seconds.",
                f"Job ID: {job_id}",
                f"Status: Processing",
                "",
                job_query
            )
        else:  # query job
            return (
                f"Query is still being processed (elapsed: {elapsed_time:.1f}s).\n\n"
                f"Try checking again in a few seconds.",
                f"Job ID: {job_id}",
                f"Input tokens: {count_tokens(job.get('query', ''))}",
                "Output tokens: pending",
                job_query
            )
    
    # If job is completed
    if job["status"] == "completed":
        result = job["result"]
        processing_time = job["end_time"] - job["start_time"]
        
        if job.get("type") == "load_files":
            return (
                f"{result[0]}\n\nProcessing time: {processing_time:.1f}s",
                result[1],
                result[2],
                "",
                job_query
            )
        else:  # query job
            return (
                f"{result[0]}\n\nProcessing time: {processing_time:.1f}s",
                result[1],
                result[2],
                result[3],
                job_query
            )
    
    # Fallback for unknown status
    return f"Job status: {job['status']}", "", "", "", job_query

# Function to clean up old jobs
def cleanup_old_jobs():
    current_time = time.time()
    to_delete = []
    
    for job_id, job in jobs.items():
        # Keep completed jobs for 24 hours, processing jobs for 48 hours
        if job["status"] == "completed" and (current_time - job.get("end_time", 0)) > 86400:
            to_delete.append(job_id)
        elif job["status"] == "processing" and (current_time - job.get("start_time", 0)) > 172800:
            to_delete.append(job_id)
    
    for job_id in to_delete:
        del jobs[job_id]
    
    debug_print(f"Cleaned up {len(to_delete)} old jobs. {len(jobs)} jobs remaining.")
    return f"Cleaned up {len(to_delete)} old jobs", "", ""

# Improve the truncate_prompt function to be more aggressive with limiting context
def truncate_prompt(prompt: str, max_tokens: int = 4096) -> str:
    """Truncate prompt to fit within token limit, preserving the most recent/relevant parts."""
    if not prompt:
        return ""
    
    if global_tokenizer:
        try:
            tokens = global_tokenizer.encode(prompt)
            if len(tokens) > max_tokens:
                # For prompts, we often want to keep the beginning instructions and the end context
                # So we'll keep the first 20% and the last 80% of the max tokens
                beginning_tokens = int(max_tokens * 0.2)
                ending_tokens = max_tokens - beginning_tokens
                
                new_tokens = tokens[:beginning_tokens] + tokens[-(ending_tokens):]
                return global_tokenizer.decode(new_tokens)
        except Exception as e:
            debug_print(f"Truncation error: {str(e)}")
    
    # Fallback to word-based truncation
    words = prompt.split()
    if len(words) > max_tokens:
        beginning_words = int(max_tokens * 0.2)
        ending_words = max_tokens - beginning_words
        
        return " ".join(words[:beginning_words] + words[-(ending_words):])
    
    return prompt



        
default_prompt = """\
{conversation_history}
Use the following context to provide a detailed technical answer to the user's question.
Do not include an introduction like "Based on the provided documents, ...". Just answer the question.

Context:
{context}

User's question:
{question}
"""

# #If you don't know the answer, please respond with "I don't know".

def load_txt_from_url(url: str) -> Document:
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text.strip()
        if not text:
            raise ValueError(f"TXT file at {url} is empty.")
        return Document(page_content=text, metadata={"source": url})
    else:
        raise Exception(f"Failed to load {url} with status {response.status_code}")
        
def load_txt_from_google_drive(link: str) -> Document:
    """
    Load text from a Google Drive shared link
    """
    # Extract the file ID from the Google Drive link
    file_id_match = re.search(r'\/d\/(.*?)\/view', link)
    if not file_id_match:
        raise ValueError(f"Could not extract file ID from Google Drive link: {link}")
    
    file_id = file_id_match.group(1)
    
    # Create direct download link
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Request the file content
    response = requests.get(download_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file from Google Drive. Status code: {response.status_code}")
    
    # Create a Document object
    content = response.text
    metadata = {"source": link}
    return Document(page_content=content, metadata=metadata)        

class ElevatedRagChain:
    def __init__(self, llm_choice: str = "Meta-Llama-3", prompt_template: str = default_prompt,
                 bm25_weight: float = 0.6, temperature: float = 0.5, top_p: float = 0.95) -> None:
        debug_print(f"Initializing ElevatedRagChain with model: {llm_choice}")
        self.embed_func = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.bm25_weight = bm25_weight
        self.faiss_weight = 1.0 - bm25_weight
        self.top_k = 5
        self.llm_choice = llm_choice
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_template = prompt_template
        self.context = ""
        self.conversation_history: List[Dict[str, str]] = []
        self.raw_data = None
        self.split_data = None
        self.elevated_rag_chain = None

    # Instance method to capture context and conversation history
    def capture_context(self, result):
        self.context = "\n".join([str(doc) for doc in result["context"]])
        result["context"] = self.context
        history_text = (
            "\n".join([f"Q: {conv['query']}\nA: {conv['response']}" for conv in self.conversation_history])
            if self.conversation_history else ""
        )
        result["conversation_history"] = history_text
        return result

    # Instance method to extract question from input data
    def extract_question(self, input_data):
        return input_data["question"]

    # Improve error handling in the ElevatedRagChain class
    def create_llm_pipeline(self):
        from langchain.llms.base import LLM  # Import LLM here so it's always defined
        normalized = self.llm_choice.lower()
        try:
            if "remote" in normalized:
                debug_print("Creating remote Meta-Llama-3 pipeline via Hugging Face Inference API...")
                from huggingface_hub import InferenceClient
                repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
                hf_api_token = os.environ.get("HF_API_TOKEN")
                if not hf_api_token:
                    raise ValueError("Please set the HF_API_TOKEN environment variable to use remote inference.")
                
                client = InferenceClient(token=hf_api_token, timeout=120)
                
                # We no longer use wait_for_model because it's unsupported
                def remote_generate(prompt: str) -> str:
                    max_retries = 3
                    backoff = 2  # start with 2 seconds
                    for attempt in range(max_retries):
                        try:
                            debug_print(f"Remote generation attempt {attempt+1}")
                            response = client.text_generation(
                                prompt,
                                model=repo_id,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                max_new_tokens=512  # Reduced token count for speed
                            )
                            return response
                        except Exception as e:
                            debug_print(f"Attempt {attempt+1} failed with error: {e}")
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(backoff)
                            backoff *= 2  # exponential backoff
                    return "Failed to generate response after multiple attempts."
                
                class RemoteLLM(LLM):
                    @property
                    def _llm_type(self) -> str:
                        return "remote_llm"
                    
                    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                        return remote_generate(prompt)
                    
                    @property
                    def _identifying_params(self) -> dict:
                        return {"model": repo_id}
                
                debug_print("Remote Meta-Llama-3 pipeline created successfully.")
                return RemoteLLM()
                
            elif "mistral-api" in normalized:
                debug_print("Creating Mistral API pipeline...")
                mistral_api_key = os.environ.get("MISTRAL_API_KEY")
                if not mistral_api_key:
                    raise ValueError("Please set the MISTRAL_API_KEY environment variable to use Mistral API.")
                try:
                    from mistralai import Mistral                    
                    debug_print("Mistral library imported successfully")
                except ImportError:
                    debug_print("Mistral client library not installed. Falling back to Llama pipeline.")
                    normalized = "llama"
                if normalized != "llama":
#                    from pydantic import PrivateAttr
#                    from langchain.llms.base import LLM
#                    from typing import Any, Optional, List
#                    import typing

                    class MistralLLM(LLM):
                        temperature: float = 0.7
                        top_p: float = 0.95
                        _client: Any = PrivateAttr(default=None)

                        def __init__(self, api_key: str, temperature: float = 0.7, top_p: float = 0.95, **kwargs: Any):
                            try:
                                super().__init__(**kwargs)
                                # Bypass Pydantic's __setattr__ to assign to _client
                                object.__setattr__(self, '_client', Mistral(api_key=api_key))
                                self.temperature = temperature
                                self.top_p = top_p
                            except Exception as e:
                                debug_print(f"Init Mistral failed with error: {e}")
                                                    
                        @property
                        def _llm_type(self) -> str:
                            return "mistral_llm"
                        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                            try:
                                debug_print("Calling Mistral API...")
                                response = self._client.chat.complete(
                                    model="mistral-small-latest",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=self.temperature,
                                    top_p=self.top_p
                                )
                                return response.choices[0].message.content
                            except Exception as e:
                                debug_print(f"Mistral API error: {str(e)}")
                                return f"Error generating response: {str(e)}"
                        @property
                        def _identifying_params(self) -> dict:
                            return {"model": "mistral-small-latest"}
                    debug_print("Creating Mistral LLM instance")
                    mistral_llm = MistralLLM(api_key=mistral_api_key, temperature=self.temperature, top_p=self.top_p)
                    debug_print("Mistral API pipeline created successfully.")
                    return mistral_llm
                
            else:
                # Default case - using a fallback model (or Llama)
                debug_print("Using local/fallback model pipeline")
                model_id = "facebook/opt-350m"  # Use a smaller model as fallback
                pipe = pipeline(
                    "text-generation",
                    model=model_id,
                    device=-1,  # CPU
                    max_length=1024
                )
                
                class LocalLLM(LLM):
                    @property
                    def _llm_type(self) -> str:
                        return "local_llm"
                    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                        # For this fallback, truncate prompt if it exceeds limits
                        reserved_gen = 128
                        max_total = 1024
                        max_prompt_tokens = max_total - reserved_gen
                        truncated_prompt = truncate_prompt(prompt, max_tokens=max_prompt_tokens)
                        generated = pipe(truncated_prompt, max_new_tokens=reserved_gen)[0]["generated_text"]
                        return generated
                    @property
                    def _identifying_params(self) -> dict:
                        return {"model": model_id, "max_length": 1024}
                
                debug_print("Local fallback pipeline created.")
                return LocalLLM()
                    
        except Exception as e:
            debug_print(f"Error creating LLM pipeline: {str(e)}")
            # Return a dummy LLM that explains the error
            class ErrorLLM(LLM):
                @property
                def _llm_type(self) -> str:
                    return "error_llm"
                
                def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                    return f"Error initializing LLM: \n\nPlease check your environment variables and try again."
                
                @property
                def _identifying_params(self) -> dict:
                    return {"model": "error"}
            
            return ErrorLLM()


    def update_llm_pipeline(self, new_model_choice: str, temperature: float, top_p: float, prompt_template: str, bm25_weight: float):
        debug_print(f"Updating chain with new model: {new_model_choice}")
        self.llm_choice = new_model_choice
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_template = prompt_template
        self.bm25_weight = bm25_weight
        self.faiss_weight = 1.0 - bm25_weight
        self.llm = self.create_llm_pipeline()
        def format_response(response: str) -> str:
            input_tokens = count_tokens(self.context + self.prompt_template)
            output_tokens = count_tokens(response)
            formatted = f"### Response\n\n{response}\n\n---\n"
            formatted += f"- **Input tokens:** {input_tokens}\n"
            formatted += f"- **Output tokens:** {output_tokens}\n"
            formatted += f"- **Generated using:** {self.llm_choice}\n"
            formatted += f"\n**Conversation History:** {len(self.conversation_history)} conversation(s) considered.\n"
            return formatted
        base_runnable = RunnableParallel({
            "context": RunnableLambda(self.extract_question) | self.ensemble_retriever,
            "question": RunnableLambda(self.extract_question)
        }) | self.capture_context
        self.elevated_rag_chain = base_runnable | self.rag_prompt | self.llm | format_response
        debug_print("Chain updated successfully with new LLM pipeline.")


    def add_pdfs_to_vectore_store(self, file_links: List[str]) -> None:
        debug_print(f"Processing files using {self.llm_choice}")
        self.raw_data = []
        for link in file_links:
            if link.lower().endswith(".pdf"):
                debug_print(f"Loading PDF: {link}")
                loaded_docs = OnlinePDFLoader(link).load()
                if loaded_docs:
                    self.raw_data.append(loaded_docs[0])
                else:
                    debug_print(f"No content found in PDF: {link}")
            elif link.lower().endswith(".txt") or link.lower().endswith(".utf-8"):
                debug_print(f"Loading TXT: {link}")
                try:
                    self.raw_data.append(load_txt_from_url(link))
                except Exception as e:
                    debug_print(f"Error loading TXT file {link}: {e}")
            elif "drive.google.com" in link and ("file/d" in link or "open?id=" in link):
                debug_print(f"Loading Google Drive file: {link}")
                try:
                    if ".pdf" in link.lower():
                        # Google Drive PDF handling
                        file_id = re.search(r'\/d\/(.*?)\/view', link).group(1)
                        direct_pdf_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                        loaded_docs = OnlinePDFLoader(direct_pdf_url).load()
                        if loaded_docs:
                            self.raw_data.append(loaded_docs[0])
                    else:
                        # Assuming it's a text file
                        self.raw_data.append(load_txt_from_google_drive(link))
                except Exception as e:
                    debug_print(f"Error loading Google Drive file {link}: {e}")
            else:
                debug_print(f"File type not supported for URL: {link}")
            
            
        debug_print("Files loaded successfully.")
        debug_print("Starting text splitting...")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        self.split_data = self.text_splitter.split_documents(self.raw_data)
        if not self.split_data:
            raise ValueError("Text splitting resulted in no chunks. Check the file contents.")
        debug_print(f"Text splitting completed. Number of chunks: {len(self.split_data)}")
        debug_print("Creating BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(self.split_data)
        self.bm25_retriever.k = self.top_k
        debug_print("BM25 retriever created.")
        debug_print("Embedding chunks and creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(self.split_data, self.embed_func)
        self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        debug_print("FAISS vector store created successfully.")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.faiss_weight]
        )
        
        base_runnable = RunnableParallel({
            "context": RunnableLambda(self.extract_question) | self.ensemble_retriever,
            "question": RunnableLambda(self.extract_question)
        }) | self.capture_context

        # Ensure the prompt template is set
        self.rag_prompt = ChatPromptTemplate.from_template(self.prompt_template)
        if self.rag_prompt is None:
            raise ValueError("Prompt template could not be created from the given template.")
        prompt_runnable = RunnableLambda(lambda vars: self.rag_prompt.format(**vars))
        
        self.str_output_parser = StrOutputParser()
        debug_print("Selecting LLM pipeline based on choice: " + self.llm_choice)
        self.llm = self.create_llm_pipeline()
        if self.llm is None:
            raise ValueError("LLM pipeline creation failed.")
        
        def format_response(response: str) -> str:
            input_tokens = count_tokens(self.context + self.prompt_template)
            output_tokens = count_tokens(response)
            formatted = f"### Response\n\n{response}\n\n---\n"
            formatted += f"- **Input tokens:** {input_tokens}\n"
            formatted += f"- **Output tokens:** {output_tokens}\n"
            formatted += f"- **Generated using:** {self.llm_choice}\n"
            formatted += f"\n**Conversation History:** {len(self.conversation_history)} conversation(s) considered.\n"
            return formatted
        
        self.elevated_rag_chain = base_runnable | prompt_runnable | self.llm | format_response
        debug_print("Elevated RAG chain successfully built and ready to use.")



    def get_current_context(self) -> str:
        base_context = "\n".join([str(doc) for doc in self.split_data[:3]]) if self.split_data else "No context available."
        history_summary = "\n\n---\n**Recent Conversations (last 3):**\n"
        recent = self.conversation_history[-3:]
        if recent:
            for i, conv in enumerate(recent, 1):
                history_summary += f"**Conversation {i}:**\n- Query: {conv['query']}\n- Response: {conv['response']}\n"
        else:
            history_summary += "No conversation history."
        return base_context + history_summary

# ----------------------------
# Gradio Interface Functions
# ----------------------------
global rag_chain
rag_chain = ElevatedRagChain()

def load_pdfs_updated(file_links, model_choice, prompt_template, bm25_weight, temperature, top_p):
    debug_print("Inside load_pdfs function.")
    if not file_links:
        debug_print("Please enter non-empty URLs")
        return "Please enter non-empty URLs", "Word count: N/A", "Model used: N/A", "Context: N/A"
    try:
        links = [link.strip() for link in file_links.split("\n") if link.strip()]
        global rag_chain
        if rag_chain.raw_data:
            rag_chain.update_llm_pipeline(model_choice, temperature, top_p, prompt_template, bm25_weight)
            context_display = rag_chain.get_current_context()
            response_msg = f"Files already loaded. Chain updated with model: {model_choice}"
            return (
                response_msg,
                f"Word count: {word_count(rag_chain.context)}",
                f"Model used: {rag_chain.llm_choice}",
                f"Context:\n{context_display}"
            )
        else:
            rag_chain = ElevatedRagChain(
                llm_choice=model_choice,
                prompt_template=prompt_template,
                bm25_weight=bm25_weight,
                temperature=temperature,
                top_p=top_p
            )
            rag_chain.add_pdfs_to_vectore_store(links)
            context_display = rag_chain.get_current_context()
            response_msg = f"Files loaded successfully. Using model: {model_choice}"
            return (
                response_msg,
                f"Word count: {word_count(rag_chain.context)}",
                f"Model used: {rag_chain.llm_choice}",
                f"Context:\n{context_display}"
            )
    except Exception as e:
        error_msg = traceback.format_exc()
        debug_print("Could not load files. Error: " + error_msg)
        return (
            "Error loading files: " + str(e),
            f"Word count: {word_count('')}",
            f"Model used: {rag_chain.llm_choice}",
            "Context: N/A"
        )

def update_model(new_model: str):
    global rag_chain
    if rag_chain and rag_chain.raw_data:
        rag_chain.update_llm_pipeline(new_model, rag_chain.temperature, rag_chain.top_p,
                                      rag_chain.prompt_template, rag_chain.bm25_weight)
        debug_print(f"Model updated to {rag_chain.llm_choice}")
        return f"Model updated to: {rag_chain.llm_choice}"
    else:
        return "No files loaded; please load files first."


# Update submit_query_updated to better handle context limitation
def submit_query_updated(query):
    debug_print(f"Processing query: {query}")
    if not query:
        debug_print("Empty query received")
        return "Please enter a non-empty query", "", "Input tokens: 0", "Output tokens: 0"
    
    if not hasattr(rag_chain, 'elevated_rag_chain') or not rag_chain.raw_data:
        debug_print("RAG chain not initialized")
        return "Please load files first.", "", "Input tokens: 0", "Output tokens: 0"
    
    try:
        # Determine max context size based on model
        model_name = rag_chain.llm_choice.lower()
        max_context_tokens = 32000 if "mistral" in model_name else 4096
        
        # Reserve 20% of tokens for the question and response generation
        reserved_tokens = int(max_context_tokens * 0.2)
        max_context_tokens -= reserved_tokens
        
        # Collect conversation history (last 2 only to save tokens)
        if rag_chain.conversation_history:
            recent_history = rag_chain.conversation_history[-2:]
            history_text = "\n".join([f"Q: {conv['query']}\nA: {conv['response'][:300]}..." 
                                     for conv in recent_history])
        else:
            history_text = ""
        
        # Get history token count
        history_tokens = count_tokens(history_text)
        
        # Adjust context tokens based on history size
        context_tokens = max_context_tokens - history_tokens
        
        # Ensure we have some minimum context
        context_tokens = max(context_tokens, 1000)
        
        # Truncate context if needed
        context = truncate_prompt(rag_chain.context, max_tokens=context_tokens)
        
        debug_print(f"Using model: {model_name}, context tokens: {count_tokens(context)}, history tokens: {history_tokens}")
        
        prompt_variables = {
            "conversation_history": history_text,
            "context": context,
            "question": query
        }
        
        debug_print("Invoking RAG chain")
        response = rag_chain.elevated_rag_chain.invoke({"question": query})
        
        # Store only a reasonable amount of the response in history
        trimmed_response = response[:1000] + ("..." if len(response) > 1000 else "")
        rag_chain.conversation_history.append({"query": query, "response": trimmed_response})
        
        input_token_count = count_tokens(query)
        output_token_count = count_tokens(response)
        
        debug_print(f"Query processed successfully. Output tokens: {output_token_count}")
        
        return (
            response,
            rag_chain.get_current_context(),
            f"Input tokens: {input_token_count}",
            f"Output tokens: {output_token_count}"
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        debug_print(f"LLM error: {error_msg}")
        return (
            f"Query error: {str(e)}\n\nTry using a smaller document or simplifying your query.",
            "",
            "Input tokens: 0",
            "Output tokens: 0"
        )

def reset_app_updated():
    global rag_chain
    rag_chain = ElevatedRagChain()
    debug_print("App reset successfully.")
    return (
        "App reset successfully. You can now load new files",
        "",
        "Model used: Not selected"
    )

# ----------------------------
# Gradio Interface Setup
# ----------------------------
custom_css = """
textarea {
  overflow-y: scroll !important;
  max-height: 200px;
}
"""

# Function to add dots and reset
def add_dots_and_reset():
    if not hasattr(add_dots_and_reset, "dots"):
        add_dots_and_reset.dots = ""  # Initialize the attribute

    # Add a dot
    add_dots_and_reset.dots += "."
    
    # Reset after 5 dots
    if len(add_dots_and_reset.dots) > 5:
        add_dots_and_reset.dots = ""
    
    print(f"Current dots: {add_dots_and_reset.dots}")  # Debugging print
    return add_dots_and_reset.dots

# Define a dummy function to simulate data retrieval
def run_query(max_value):
    # Simulate a data retrieval or processing function
    return [[i, i**2] for i in range(1, max_value + 1)]

# Function to call both refresh_job_list and check_job_status using the last job ID
def periodic_update(is_checked):
    interval = 2 if is_checked else None
    debug_print(f"Auto-refresh checkbox is {'checked' if is_checked else 'unchecked'}, every={interval}")
    if is_checked:
        global last_job_id
        job_list_md = refresh_job_list()
        job_status = check_job_status(last_job_id) if last_job_id else ("No job ID available", "", "", "", "")
        query_results = run_query(10)  # Use a fixed value or another logic if needed
        context_info = rag_chain.get_current_context() if rag_chain else "No context available."
        return job_list_md, job_status[0], query_results, context_info
    else:
        # Return empty values to stop updates
        return "", "", [], ""

# Define a function to determine the interval based on the checkbox state
def get_interval(is_checked):
    return 2 if is_checked else None

# Update the Gradio interface to include job status checking
with gr.Blocks(css=custom_css, js="""
document.addEventListener('DOMContentLoaded', function() {
    // Add event listener for job list clicks
    const jobListInterval = setInterval(() => {
        const jobLinks = document.querySelectorAll('.job-list-container a');
        if (jobLinks.length > 0) {
            jobLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const jobId = this.textContent.split(' ')[0];
                    // Find the job ID input textbox and set its value
                    const jobIdInput = document.querySelector('.job-id-input input');
                    if (jobIdInput) {
                        jobIdInput.value = jobId;
                        // Trigger the input event to update Gradio's state
                        jobIdInput.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                });
            });
            clearInterval(jobListInterval);
        }
    }, 500);
});
""") as app:
    gr.Markdown('''# PhiRAG - Async Version  
**PhiRAG** Query Your Data with Advanced RAG Techniques

**Model Selection & Parameters:** Choose from the following options:
- 🇺🇸 Remote Meta-Llama-3 - has context windows of 8000 tokens
- 🇪🇺 Mistral-API - has context windows of 32000 tokens

**🔥 Randomness (Temperature):** Adjusts output predictability. 
- Example: 0.2 makes the output very deterministic (less creative), while 0.8 introduces more variety and spontaneity.

**🎯 Word Variety (Top‑p):** Limits word choices to a set probability percentage.
- Example: 0.5 restricts output to the most likely 50% of token choices for a focused answer; 0.95 allows almost all possibilities for more diverse responses.

**⚖️ BM25 Weight:** Adjust Lexical vs Semantics.
- Example: A value of 0.8 puts more emphasis on exact keyword (lexical) matching, while 0.3 shifts emphasis toward semantic similarity.

**✏️ Prompt Template:** Edit as desired.

**🔗 File URLs:** Enter one URL per line (.pdf or .txt).\
- Example: Provide one URL per line, such as
https://www.gutenberg.org/ebooks/8438.txt.utf-8

**🔍 Query:** Enter your query below.

**⚠️ IMPORTANT: This app now uses asynchronous processing to avoid timeout issues**
- When you load files or submit a query, you'll receive a Job ID
- Use the "Check Job Status" tab to monitor and retrieve your results
''')

    with gr.Tabs() as tabs:
        with gr.TabItem("Setup & Load Files"):
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=["🇺🇸 Remote Meta-Llama-3", "🇪🇺 Mistral-API"],
                        value="🇺🇸 Remote Meta-Llama-3",
                        label="Select Model"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                        label="Randomness (Temperature)"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=0.99, value=0.95, step=0.05,
                        label="Word Variety (Top-p)"
                    )
                with gr.Column():
                    pdf_input = gr.Textbox(
                        label="Enter your file URLs (one per line)",
                        placeholder="Enter one URL per line (.pdf or .txt)",
                        lines=4
                    )
                    prompt_input = gr.Textbox(
                        label="Custom Prompt Template",
                        placeholder="Enter your custom prompt template here",
                        lines=8,
                        value=default_prompt
                    )
                with gr.Column():
                    bm25_weight_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.6, step=0.1,
                        label="Lexical vs Semantics (BM25 Weight)"
                    )
                    load_button = gr.Button("Load Files (Async)")
                    load_status = gr.Markdown("Status: Waiting for files")
            
            with gr.Row():
                load_response = gr.Textbox(
                    label="Load Response",
                    placeholder="Response will appear here",
                    lines=4
                )
                load_context = gr.Textbox(
                    label="Context Info",
                    placeholder="Context info will appear here",
                    lines=4
                )
            
            with gr.Row():
                model_output = gr.Markdown("**Current Model**: Not selected")
        
        with gr.TabItem("Submit Query"):
            with gr.Row():
                # Add this line to define the query_model_dropdown
                query_model_dropdown = gr.Dropdown(
                    choices=["🇺🇸 Remote Meta-Llama-3", "🇪🇺 Mistral-API"],
                    value="🇺🇸 Remote Meta-Llama-3",
                    label="Query Model"
                )

                query_input = gr.Textbox(
                    label="Enter your query here",
                    placeholder="Type your query",
                    lines=4
                )
                submit_button = gr.Button("Submit Query (Async)")
            
            with gr.Row():
                query_response = gr.Textbox(
                    label="Query Response",
                    placeholder="Response will appear here (formatted as Markdown)",
                    lines=6
                )
                query_context = gr.Textbox(
                    label="Context Information",
                    placeholder="Retrieved context and conversation history will appear here",
                    lines=6
                )
            
            with gr.Row():
                input_tokens = gr.Markdown("Input tokens: 0")
                output_tokens = gr.Markdown("Output tokens: 0")
        
            with gr.TabItem("Check Job Status"):
                with gr.Row():
                    with gr.Column(scale=1):
                        job_list = gr.Markdown(
                            value="No jobs yet",
                            label="Job List (Click to select)"
                        )
                        # Add the Refresh Job List button
                        refresh_button = gr.Button("Refresh Job List")
                        
                        # Use a Checkbox to control the periodic updates
                        auto_refresh_checkbox = gr.Checkbox(
                            label="Enable Auto Refresh",
                            value=False  # Default to unchecked
                        )
                        
                        # Use a DataFrame to display results
                        df = gr.DataFrame(
                            value=run_query(10),  # Initial value
                            headers=["Number", "Square"],
                            label="Query Results",
                            visible=False  # Set the DataFrame to be invisible
                        )
                    
                    with gr.Column(scale=2):
                        job_id_input = gr.Textbox(
                            label="Job ID",
                            placeholder="Job ID will appear here when selected from the list",
                            lines=1
                        )
                        job_query_display = gr.Textbox(
                            label="Job Query",
                            placeholder="The query associated with this job will appear here",
                            lines=2,
                            interactive=False
                        )
                        check_button = gr.Button("Check Status")
                        cleanup_button = gr.Button("Cleanup Old Jobs")
                
                with gr.Row():
                    status_response = gr.Textbox(
                        label="Job Result",
                        placeholder="Job result will appear here",
                        lines=6
                    )
                    status_context = gr.Textbox(
                        label="Context Information",
                        placeholder="Context information will appear here",
                        lines=6
                    )
                
                with gr.Row():
                    status_tokens1 = gr.Markdown("")
                    status_tokens2 = gr.Markdown("")
        
        with gr.TabItem("App Management"):
            with gr.Row():
                reset_button = gr.Button("Reset App")
            
            with gr.Row():
                reset_response = gr.Textbox(
                    label="Reset Response",
                    placeholder="Reset confirmation will appear here",
                    lines=2
                )
                reset_context = gr.Textbox(
                    label="",
                    placeholder="",
                    lines=2,
                    visible=False
                )
            
            with gr.Row():
                reset_model = gr.Markdown("")
    
    # Connect the buttons to their respective functions
    load_button.click(
        load_pdfs_async, 
        inputs=[pdf_input, model_dropdown, prompt_input, bm25_weight_slider, temperature_slider, top_p_slider],
        outputs=[load_response, load_context, model_output, job_id_input, job_query_display, job_list]
    )

    # Also sync in the other direction
    query_model_dropdown.change(
        fn=sync_model_dropdown,
        inputs=query_model_dropdown,
        outputs=model_dropdown
    )

    submit_button.click(
        submit_query_async, 
        inputs=[query_input, query_model_dropdown],
        outputs=[query_response, query_context, input_tokens, output_tokens, job_id_input, job_query_display, job_list]
    )

    check_button.click(
        check_job_status,
        inputs=[job_id_input],
        outputs=[status_response, status_context, status_tokens1, status_tokens2, job_query_display]
    )

    # Connect the refresh button to the refresh_job_list function
    refresh_button.click(
        refresh_job_list,
        inputs=[],
        outputs=[job_list]
    )

    # Connect the job list selection event (this is handled by JavaScript)
    job_id_input.change(
        job_selected,
        inputs=[job_id_input],
        outputs=[job_id_input, job_query_display]
    )

    cleanup_button.click(
        cleanup_old_jobs,
        inputs=[],
        outputs=[status_response, status_context, status_tokens1]
    )

    reset_button.click(
        reset_app_updated, 
        inputs=[], 
        outputs=[reset_response, reset_context, reset_model]
    )

    model_dropdown.change(
        fn=sync_model_dropdown,
        inputs=model_dropdown,
        outputs=query_model_dropdown
    )

    # Add an event to refresh the job list on page load
    app.load(
        fn=refresh_job_list,
        inputs=None,
        outputs=job_list
    )

    # Use the Checkbox to control the periodic updates
    auto_refresh_checkbox.change(
        fn=periodic_update,
        inputs=[auto_refresh_checkbox],
        outputs=[job_list, status_response, df, status_context],
        every=2 #if auto_refresh_checkbox.value else None  # Directly set `every` based on the checkbox state
    )

if __name__ == "__main__":
    debug_print("Launching Gradio interface.")
    app.queue().launch(share=False)
