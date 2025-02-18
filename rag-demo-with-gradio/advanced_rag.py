import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datetime
import functools
import traceback
from typing import List, Optional

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
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from transformers.quantizers.auto import AutoQuantizationConfig
import gradio as gr

os.environ["HF_API_TOKEN"] = "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"

# Debug print function
def debug_print(message: str):
    print(f"[{datetime.datetime.now().isoformat()}] {message}")

def word_count(text: str) -> int:
    return len(text.split())

# Initialize tokenizer for counting
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

    def create_llm_pipeline(self):
        if "remote" in self.llm_choice.lower():
            debug_print("Creating remote Meta-Llama-3 pipeline via Hugging Face Inference API...")
            from huggingface_hub import InferenceClient
            repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            hf_api_token = os.environ.get("HF_API_TOKEN")
            if not hf_api_token:
                raise ValueError("Please set the HF_API_TOKEN environment variable to use remote inference.")
            client = InferenceClient(token=hf_api_token)
            
            def remote_generate(prompt: str) -> str:
                response = client.text_generation(
                    prompt,
                    model=repo_id,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=1.1
                )
                return response
            
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
        else:
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            if "deepseek" in self.llm_choice.lower():
                model_id = "deepseek-ai/DeepSeek-R1"
            elif "gemini" in self.llm_choice.lower():
                model_id = "gemini/flash-1.5"

            pipe = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                max_length=4096,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                device=-1
            )
            return HuggingFacePipeline(pipeline=pipe)

    def add_pdfs_to_vectore_store(self, pdf_links: List) -> None:
        debug_print(f"Processing PDFs using {self.llm_choice}")
        self.raw_data = [OnlinePDFLoader(doc).load()[0] for doc in pdf_links]
        debug_print("PDFs loaded successfully.")
        
        debug_print("Starting text splitting...")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
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
        
        ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.faiss_weight]
        )
        
        def capture_context(result):
            # Revert: extract the page_content of each Document.
            self.context = "\n".join([doc.page_content for doc in result["context"] if hasattr(doc, "page_content")])
            return result

        base_runnable = RunnableParallel({
            "context": ensemble,
            "question": RunnablePassthrough()
        }) | capture_context

        self.rag_prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.str_output_parser = StrOutputParser()
        debug_print("Selecting LLM pipeline based on choice: " + self.llm_choice)
        self.llm = self.create_llm_pipeline()
        
        def format_response(response: str) -> str:
            input_tokens = count_tokens(self.context + self.prompt_template)
            output_tokens = count_tokens(response)
            token_info = f"\nInput tokens: {input_tokens} | Output tokens: {output_tokens}"
            model_info = f"\nGenerated using: {self.llm_choice}"
            return response + token_info + model_info

        self.elevated_rag_chain = base_runnable | self.rag_prompt | self.llm | format_response
        debug_print("Elevated RAG chain successfully built and ready to use.")
        
    def get_current_context(self) -> str:
        # Revert: return the page_content of the first three documents
        if hasattr(self, "split_data") and self.split_data:
            return "\n".join([doc.page_content for doc in self.split_data[:3] if hasattr(doc, "page_content")])
        return "No context available."

# ----------------------------
# Gradio Interface Functions
# ----------------------------
global rag_chain
rag_chain = ElevatedRagChain()

def load_pdfs_updated(pdf_links, model_choice, prompt_template, bm25_weight, temperature, top_p):
    debug_print("Inside load_pdfs function.")
    if not pdf_links:
        debug_print("Please enter non-empty URLs")
        return "Please enter non-empty URLs", "Word count: N/A", "Model used: N/A", "Context: N/A"
    try:
        links = [link.strip() for link in pdf_links.split("\n") if link.strip()]
        global rag_chain
        rag_chain = ElevatedRagChain(
            llm_choice=model_choice,
            prompt_template=prompt_template,
            bm25_weight=bm25_weight,
            temperature=temperature,
            top_p=top_p
        )
        rag_chain.add_pdfs_to_vectore_store(links)
        context_display = rag_chain.get_current_context()
        response_msg = f"PDFs loaded successfully. Using model: {model_choice}"
        debug_print(response_msg)
        return (
            response_msg,
            f"Word count: {word_count(rag_chain.context)}",
            f"Model used: {rag_chain.llm_choice}",
            f"Context:\n{context_display}"
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        debug_print("Could not load PDFs. Error: " + error_msg)
        return (
            "Error loading PDFs: " + str(e),
            f"Word count: {word_count('')}",
            f"Model used: {rag_chain.llm_choice}",
            "Context: N/A"
        )

def submit_query_updated(query):
    debug_print("Inside submit_query function.")
    if not query:
        debug_print("Please enter a non-empty query")
        return "Please enter a non-empty query", "Word count: 0", f"Model used: {rag_chain.llm_choice}", ""
    if hasattr(rag_chain, 'elevated_rag_chain'):
        try:
            response = rag_chain.elevated_rag_chain.invoke(query)
            input_token_count = count_tokens(query)
            output_token_count = count_tokens(response)
            return (
                response,
                rag_chain.context,
                f"Input tokens: {input_token_count}",
                f"Output tokens: {output_token_count}"
            )
        except Exception as e:
            error_msg = traceback.format_exc()
            debug_print("LLM error. Error: " + error_msg)
            return (
                "Query error: " + str(e),
                "",
                "Input tokens: 0",
                "Output tokens: 0"
            )
    return (
        "Please load PDFs first.",
        "",
        "Input tokens: 0",
        "Output tokens: 0"
    )

def reset_app_updated():
    global rag_chain
    rag_chain = ElevatedRagChain()
    debug_print("App reset successfully.")
    return (
        "App reset successfully. You can now load new PDFs",
        "",
        "Model used: Not selected"
    )

# ----------------------------
# Gradio Interface Setup
# ----------------------------
custom_css = """
button {
    background-color: grey !important;
    font-family: Arial !important;
    font-weight: bold !important;
    color: blue !important;
}
"""

with gr.Blocks(css=custom_css) as app:
    gr.Markdown('''# PhiRAG  
**PhiRAG** enables you to query your own data with advanced RAG techniques.  
- **Model Selection & Parameters:** Choose from local or remote models (Meta-Llama-3, DeepSeek-R1, Gemini Flash 1.5) and adjust temperature & top-p via the sliders.  
- **Weight Controls:** Adjust BM25 weight (the complement is used for FAISS similarity).  
- **Prompt Template:** Edit the prompt template if desired.  
- **PDF URLs:** Enter one or more PDF URLs (one per line).  
- **Query:** Enter your query below.  
The response displays the model used, word count, and the current context.
    ''')
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=["Meta-Llama-3", "Remote Meta-Llama-3", "DeepSeek-R1", "Gemini Flash 1.5"],
                value="Remote Meta-Llama-3",
                label="Select Model"
            )
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                label="Temperature"
            )
            top_p_slider = gr.Slider(
                minimum=0.1, maximum=0.99, value=0.95, step=0.05,
                label="Top-p"
            )
        with gr.Column():
            pdf_input = gr.Textbox(
                label="Enter your PDF URLs (one per line)",
                placeholder="Enter one URL per line",
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
                label="BM25 Weight"
            )
            load_button = gr.Button("Load PDF")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Enter your query here",
                placeholder="Type your query",
                lines=4
            )
            submit_button = gr.Button("Submit")
        with gr.Column():
            reset_button = gr.Button("Reset App")
    
    with gr.Row():
        response_output = gr.Textbox(
            label="Response",
            placeholder="Response will appear here",
            lines=6
        )
        context_output = gr.Textbox(
            label="Current Context",
            placeholder="Retrieved context will appear here",
            lines=6
        )
    
    with gr.Row():
        input_tokens = gr.Markdown("Input tokens: 0")
        output_tokens = gr.Markdown("Output tokens: 0")
        model_output = gr.Markdown("**Current Model**: Not selected")
    
    load_button.click(
        load_pdfs_updated, 
        inputs=[pdf_input, model_dropdown, prompt_input, bm25_weight_slider, temperature_slider, top_p_slider],
        outputs=[response_output, context_output, model_output]
    )
    
    submit_button.click(
        submit_query_updated, 
        inputs=[query_input],
        outputs=[response_output, context_output, input_tokens, output_tokens]
    )
    
    reset_button.click(
        reset_app_updated, 
        inputs=[], 
        outputs=[response_output, context_output, model_output]
    )

if __name__ == "__main__":
    debug_print("Launching Gradio interface.")
    app.launch(share=True)
