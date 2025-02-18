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

# Debug print function
def debug_print(message: str):
    print(f"[{datetime.datetime.now().isoformat()}] {message}")

def word_count(text: str) -> int:
    return len(text.split())

# Default prompt template - Defined before the class
default_prompt = """\
Use the following context to provide a detailed technical answer to the user's question.
Do not include an introduction like "Based on the provided documents, ...". Just answer the question.
If you don't know the answer, please respond with "I don't know".

Context:
{context}

User's question:
{question}
"""

global_tokenizer = None

def initialize_tokenizer():
    global global_tokenizer
    try:
        global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    except:
        debug_print("Failed to initialize main tokenizer, falling back to GPT2 tokenizer")
        global_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    if global_tokenizer is None:
        initialize_tokenizer()
    try:
        return len(global_tokenizer.encode(text))
    except:
        return len(text.split())  # Fallback to word count



class ElevatedRagChain:
    def __init__(self, llm_choice: str = "Meta-Llama-3", prompt_template: str = default_prompt,
                 bm25_weight: float = 0.6, temperature: float = 0.5, top_p: float = 1.0) -> None:
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
        self.is_remote = "remote" in llm_choice.lower()

    def create_remote_llm(self):
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        hf_api_token = os.environ.get("HF_API_TOKEN")
        if not hf_api_token:
            raise ValueError("HF_API_TOKEN not set for remote inference")
        
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
            def _llm_type(self) -> str:
                return "remote_llm"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                return remote_generate(prompt)
            
            @property
            def _identifying_params(self) -> dict:
                return {"model": repo_id}
        
        return RemoteLLM()

    def create_llm_pipeline(self):
        if self.is_remote:
            return self.create_remote_llm()
            
        if self.llm_choice.lower() in ["meta-llama-3", "llama", "llama3"]:
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif self.llm_choice.lower() in ["deepseek", "deepseek-r1"]:
            model_id = "deepseek-ai/DeepSeek-R1"
        elif self.llm_choice.lower() in ["gemini flash 1.5", "gemini"]:
            model_id = "gemini/flash-1.5"
        else:
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

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
        
        # Modify base_runnable to capture context
        def capture_context(result):
            self.context = "\n".join([doc.page_content for doc in result["context"]])
            return result

        base_runnable = RunnableParallel({
            "context": ensemble,
            "question": RunnablePassthrough()
        }) | capture_context

        self.rag_prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.str_output_parser = StrOutputParser()
        self.llm = self.create_llm_pipeline()
        
        def format_response(self, response: str) -> str:
            input_tokens = count_tokens(self.context + self.prompt_template)
            output_tokens = count_tokens(response)
            token_info = f"\nInput tokens: {input_tokens} | Output tokens: {output_tokens}"
            model_info = f"\nGenerated using: {self.llm_choice}"
            return response + token_info + model_info
        
        self.elevated_rag_chain = base_runnable | self.rag_prompt | self.llm | format_response

# Gradio interface setup
custom_css = """
/* Customize button style */
button {
    background-color: grey !important;
    font-family: Arial !important;
    font-weight: bold !important;
    color: blue !important;
}
"""

# Gradio interface updates
with gr.Blocks(css=custom_css) as app:
    # Add current model indicator
    current_model = gr.Markdown("**Current Model**: Not selected")
    
    gr.Markdown('''# Query Your Own Data
## PhiRAG
- Select your model and adjust the parameters:
  - BM25 Weight: Controls the importance of keyword matching (complement is used for FAISS vector similarity)
  - Temperature: Controls randomness in generation (0.0-1.0)
  - Top-p: Controls diversity of generated text (0.0-1.0)
- Enter PDF URLs (one per line) and click Load PDF
- Enter your query and click Submit
- The response will show the model used and word count
- Context window shows the retrieved text used for the response
    ''')
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=["Meta-Llama-3", "Remote Meta-Llama-3", "DeepSeek-R1", "Gemini Flash 1.5"],
                value="Meta-Llama-3",
                label="Select Model"
            )
            temperature_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                label="Temperature"
            )
            top_p_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=1.0, step=0.1,
                label="Top-p"
            )
            bm25_weight_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.6, step=0.1,
                label="BM25 Weight"
            )
            
    with gr.Row():
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
            load_button = gr.Button("Load PDF")

    with gr.Row():
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
        context_output = gr.Textbox(
            label="Current Context",
            placeholder="Retrieved context will appear here",
            lines=6
        )
    
    reset_button = gr.Button("Reset App")


    def reset_app_updated():
        global rag_chain
        rag_chain = ElevatedRagChain()
        return "App reset successfully.", ""

    # Add token counters
    with gr.Row():
        input_tokens = gr.Markdown("Input tokens: 0")
        output_tokens = gr.Markdown("Output tokens: 0")
    
    def update_token_count(text):
        count = count_tokens(text)
        return f"Input tokens: {count}"
    
    def load_pdfs_updated(pdf_links, model_choice, prompt_template, bm25_weight, temperature, top_p):
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
            return (
                f"PDFs loaded successfully. Using model: {model_choice}",
                rag_chain.context,
                f"**Current Model**: {model_choice}"
            )
        except Exception as e:
            return (
                f"Error loading PDFs: {str(e)}",
                "",
                "**Current Model**: Error occurred"
            )

    def submit_query_updated(query):
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
                return (
                    f"Query error: {str(e)}",
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

    # Update event handlers
    query_input.change(
        update_token_count,
        inputs=[query_input],
        outputs=[input_tokens]
    )
    
    load_button.click(
        load_pdfs_updated,
        inputs=[pdf_input, model_dropdown, prompt_input, bm25_weight_slider, temperature_slider, top_p_slider],
        outputs=[response_output, context_output, current_model]
    )
    
    submit_button.click(
        submit_query_updated,
        inputs=[query_input],
        outputs=[response_output, context_output, input_tokens, output_tokens]
    )

if __name__ == "__main__":
    debug_print("Launching Gradio interface.")
    initialize_tokenizer()
    app.launch(share=True)
