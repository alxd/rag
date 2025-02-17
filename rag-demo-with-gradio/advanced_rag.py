import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datetime
import functools
import traceback
from typing import List, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

# Other LangChain and community imports
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from transformers.quantizers.auto import AutoQuantizationConfig
import gradio as gr

# ----------------------------
# Globals for Debugging & Token Count
# ----------------------------
def debug_print(message: str):
    print(f"[{datetime.datetime.now().isoformat()}] {message}")

last_token_count = 0

def debug_token_count_function(prompt: str, tokenizer) -> str:
    global last_token_count
    if tokenizer is None:
        print("Can't count tokens. Tokenizer failed to initialize. Please check the model name and dependencies.")
        return prompt
    tokens = tokenizer.encode(prompt)
    last_token_count = len(tokens)
    debug_print(f"Token count for prompt: {last_token_count}")
    return prompt

# ----------------------------
# Remote LLM Pipeline Creator using Hugging Face Inference API
# ----------------------------
def create_llama3_pipeline_remote():
    debug_print("Creating remote Meta-Llama-3 pipeline via Hugging Face Inference API...")
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_api_token = os.environ.get("HF_API_TOKEN")
    if hf_api_token is None:
        raise ValueError("Please set the HF_API_TOKEN environment variable to use remote inference.")
    
    # Import and create the client
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=hf_api_token)
    
    def remote_generate(prompt: str) -> str:
        # Use the client's text-generation method
        response = client.text_generation(
            prompt,
            model=repo_id,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        return response
    
    # Wrap remote_generate in a LangChain LLM interface
    from langchain.llms.base import LLM
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
# ----------------------------
# Local LLM Pipeline Creators
# ----------------------------
def create_deepseek_pipeline() -> HuggingFacePipeline:
    debug_print("Creating DeepSeek pipeline: starting quantization config setup.")
    quant_config = AutoQuantizationConfig.from_dict({
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "bitsandbytes_4bit",
        "weight_block_size": [128, 128]
    })
    debug_print("Quantization config created successfully.")
    
    debug_print("Loading DeepSeek-R1 model with quantization config...")
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1", 
        trust_remote_code=True,
        quantization_config=quant_config
    )
    debug_print("DeepSeek-R1 model loaded successfully.")
    
    debug_print("Loading DeepSeek-R1 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
    debug_print("Tokenizer loaded successfully.")
    
    debug_print("Creating text-generation pipeline for DeepSeek-R1...")
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        trust_remote_code=True,
        max_length=2048,
        do_sample=True,
        temperature=0.5,
        top_p=1,
#        device=0 if torch.cuda.is_available() else -1
        device=-1 
    )
    debug_print("DeepSeek pipeline created successfully.")
    return HuggingFacePipeline(pipeline=pipe)

def create_llama3_pipeline() -> HuggingFacePipeline:
    debug_print("Creating local Meta-Llama-3 pipeline...")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        max_length=4096,
        do_sample=True,
        temperature=0.5,
        top_p=1,
        device=-1
        #device=0 if torch.cuda.is_available() else -1,        
    )
    debug_print("Local Meta-Llama-3 pipeline created successfully.")
    return HuggingFacePipeline(pipeline=pipe)

def create_gemini_pipeline() -> HuggingFacePipeline:
    debug_print("Creating Gemini Flash 1.5 pipeline...")
    model_id = "gemini/flash-1.5"  # Placeholder model id; replace with actual id if available.
    pipe = pipeline(
        "text-generation",
        model=model_id,
        max_length=4096,
        do_sample=True,
        temperature=0.5,
        top_p=1,
        device=-1
        #device=0 if torch.cuda.is_available() else -1,        
    )
    debug_print("Gemini Flash 1.5 pipeline created successfully.")
    return HuggingFacePipeline(pipeline=pipe)

def create_llm_pipeline(choice: str = "Meta-Llama-3") -> HuggingFacePipeline:
    lc_choice = choice.lower()
    if lc_choice in ["meta-llama-3", "llama", "llama3"]:
        debug_print("Using local Meta-Llama-3 pipeline.")
        return create_llama3_pipeline()        
    elif lc_choice in ["remote meta-llama-3", "remote llama"]:
        debug_print("Using remote Meta-Llama-3 pipeline via Hugging Face Inference API.")
        return create_llama3_pipeline_remote()
    elif lc_choice in ["deepseek", "deepseek-r1"]:
        debug_print("Using DeepSeek-R1 pipeline.")
        return create_deepseek_pipeline()
    elif lc_choice in ["gemini flash 1.5", "gemini"]:
        debug_print("Using Gemini Flash 1.5 pipeline.")
        return create_gemini_pipeline()
    else:
        debug_print("Invalid model choice, defaulting to local Meta-Llama-3.")
        return create_llama3_pipeline()

# Default prompt template
default_prompt = """\
Use the following context to provide a detailed technical answer to the user's question.
Do not include an introduction like "Based on the provided documents, ...". Just answer the question.
If you don't know the answer, please respond with "I don't know".

Context:
{context}

User's question:
{question}
"""

# ----------------------------
# ElevatedRagChain Class
# ----------------------------
class ElevatedRagChain:
    def __init__(self, llm_choice: str = "Meta-Llama-3", prompt_template: str = default_prompt) -> None:
        debug_print("Initializing ElevatedRagChain with llm_choice: " + llm_choice)
        self.embed_func = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
            )
        #self.embed_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.bm25_weight  = 0.6
        self.faiss_weight = 0.4
        self.top_k        = 5
        self.llm_choice   = llm_choice
        self.prompt_template = prompt_template

    def add_pdfs_to_vectore_store(self, pdf_links: List, chunk_size: int = 1500) -> None:
        debug_print("Starting PDF processing for vector store.")
        self.raw_data = [OnlinePDFLoader(doc).load()[0] for doc in pdf_links]
        debug_print("PDFs loaded successfully.")
        
        debug_print("Starting text splitting...")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        self.split_data = self.text_splitter.split_documents(self.raw_data)
        debug_print(f"Text splitting completed. Number of chunks: {len(self.split_data)}")
        
        debug_print("Creating BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(self.split_data)
        self.bm25_retriever.k = self.top_k
        debug_print("BM25 retriever created.")
        
        debug_print("Embedding chunks and creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(self.split_data, self.embed_func)
        self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        debug_print("FAISS vector store created successfully.")
        
        debug_print("All PDFs processed and added to vector store.")
        self.build_elevated_rag_system()
        debug_print("RAG system is built successfully.")

    def build_elevated_rag_system(self) -> None:
        debug_print("Building elevated RAG system.")
        ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.faiss_weight]
        )
        base_runnable = RunnableParallel({
            "context": ensemble,
            "question": RunnablePassthrough()
        })
        self.rag_prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.str_output_parser = StrOutputParser()
        debug_print("Selecting LLM pipeline based on choice: " + self.llm_choice)
        self.llm = create_llm_pipeline(choice=self.llm_choice)
        debug_token_fn = functools.partial(debug_token_count_function, tokenizer=self.llm.pipeline.tokenizer if hasattr(self.llm, "pipeline") else None)
        self.elevated_rag_chain = base_runnable | self.rag_prompt | debug_token_fn | self.llm
        debug_print("Elevated RAG chain successfully built and ready to use.")

# ----------------------------
# Gradio Interface Functions
# ----------------------------
os.environ["HF_API_TOKEN"] = "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" 
rag_chain = ElevatedRagChain()

def load_pdfs(pdf_links, model_choice, prompt_template):
    debug_print("Inside load_pdfs function.")
    if not pdf_links:
        debug_print("Please enter non-empty URLs")
        return "Please enter non-empty URLs", f"Token count: {last_token_count}"
    try:
        links = [link.strip() for link in pdf_links.split("\n") if link.strip()]
        debug_print(f"PDF links received: {links}")
        global rag_chain
        rag_chain = ElevatedRagChain(llm_choice=model_choice, prompt_template=prompt_template)
        rag_chain.add_pdfs_to_vectore_store(links)
        debug_print("PDFs loaded successfully into a new vector store. If you had an old one, it was overwritten.")
        return ("PDFs loaded successfully into a new vector store. If you had an old one, it was overwritten.",
                f"Token count: {last_token_count}")
    except Exception as e:
        error_msg = traceback.format_exc()
        debug_print("Could not load PDFs. Are URLs valid? Error: " + error_msg)
        return ("Could not load PDFs. Are URLs valid?\n" + error_msg,
                f"Token count: {last_token_count}")

def submit_query(query):
    debug_print("Inside submit_query function.")
    if not query:
        debug_print("Please enter a non-empty query")
        return "Please enter a non-empty query", f"Token count: {last_token_count}"
    if hasattr(rag_chain, 'elevated_rag_chain'):
        try:
            response = rag_chain.elevated_rag_chain.invoke(query)
            debug_print("Query processed successfully.")
            return response, f"Token count: {last_token_count}"
        except Exception as e:
            error_msg = traceback.format_exc()
            debug_print("LLM error. Please re-submit your query. Error: " + error_msg)
            return ("LLM error. Please re-submit your query.\n" + error_msg,
                    f"Token count: {last_token_count}")
    else:
        debug_print("Please load PDFs before submitting a query")
        return "Please load PDFs before submitting a query", f"Token count: {last_token_count}"

def reset_app():
    global rag_chain
    rag_chain = ElevatedRagChain()  # Reinitialize with default values
    debug_print("App reset successfully. You can now load new PDFs")
    return "App reset successfully. You can now load new PDFs", f"Token count: {last_token_count}"

# ----------------------------
# Gradio Interface Setup
# ----------------------------
custom_css = """
/* Customize button style */
button {
    background-color: grey !important;
    font-family: Arial !important;
    font-weight: bold !important;
    color: blue !important;
}

/* Example for a custom background color for the container */
/* .gradio-container {background-color: #E0F7FA} */
"""

with gr.Blocks(css=custom_css) as app:
    gr.Markdown('''# Query Your Own Data
## Llama 2 RAG
- Enter one or more URLs for PDF files (one per line), select the model from the dropdown (note: choose "Remote Meta-Llama-3" to run inference on Hugging Face’s servers), and modify the prompt template if desired.
- Click **Load PDF** and wait until the RAG system is built.
- Enter your query and click **Submit**. The LLM's response, token count, and any error messages will appear.
- Click **Reset App** to clear/reset the RAG system.
    ''')
    with gr.Row():
        with gr.Column():
            pdf_input = gr.Textbox(
                label="Enter your PDF URLs (one per line)",
                placeholder="Enter one URL per line",
                lines=4
            )
            model_dropdown = gr.Dropdown(
                choices=["Meta-Llama-3", "Remote Meta-Llama-3", "DeepSeek-R1", "Gemini Flash 1.5"],
                value="Meta-Llama-3",
                label="Select Model"
            )
            prompt_input = gr.Textbox(
                label="Custom Prompt Template",
                placeholder="Enter your custom prompt template here",
                lines=8,
                value=default_prompt
            )
            load_button = gr.Button("Load PDF")
        with gr.Column():
            query_input = gr.Textbox(
                label="Enter your query here",
                placeholder="Type your query",
                lines=4
            )
            submit_button = gr.Button("Submit")
    
    with gr.Row():
        response_output = gr.Textbox(
            label="Response",
            placeholder="Response will appear here",
            lines=6
        )
    with gr.Row():
        token_output = gr.Textbox(
            label="Token Count",
            placeholder="Token count will be displayed here",
            lines=1
        )
    reset_button = gr.Button("Reset App")
    
    load_button.click(load_pdfs, inputs=[pdf_input, model_dropdown, prompt_input], outputs=[response_output, token_output])
    submit_button.click(submit_query, inputs=query_input, outputs=[response_output, token_output])
    reset_button.click(reset_app, inputs=[], outputs=[response_output, token_output])

if __name__ == "__main__":
    debug_print("Launching Gradio interface.")
    app.launch()
