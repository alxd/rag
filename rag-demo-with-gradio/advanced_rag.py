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
    print(f"[{datetime.datetime.now().isoformat()}] {message}")

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

def truncate_prompt(prompt: str, max_tokens: int = 4096) -> str:
    if global_tokenizer:
        try:
            tokens = global_tokenizer.encode(prompt)
            if len(tokens) > max_tokens:
                tokens = tokens[-max_tokens:]  # keep the last max_tokens tokens
                return global_tokenizer.decode(tokens)
        except Exception as e:
            debug_print("Truncation error: " + str(e))
    words = prompt.split()
    if len(words) > max_tokens:
        return " ".join(words[-max_tokens:])
    return prompt

default_prompt = """\
{conversation_history}
Use the following context to provide a detailed technical answer to the user's question.
Do not include an introduction like "Based on the provided documents, ...". Just answer the question.
If you don't know the answer, please respond with "I don't know".

Context:
{context}

User's question:
{question}
"""

def load_txt_from_url(url: str) -> Document:
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text.strip()
        if not text:
            raise ValueError(f"TXT file at {url} is empty.")
        return Document(page_content=text, metadata={"source": url})
    else:
        raise Exception(f"Failed to load {url} with status {response.status_code}")

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

    def create_llm_pipeline(self):
        normalized = self.llm_choice.lower()
        if "remote" in normalized:
            debug_print("Creating remote Meta-Llama-3 pipeline via Hugging Face Inference API...")
            from huggingface_hub import InferenceClient
            repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            hf_api_token = os.environ.get("HF_API_TOKEN")
            if not hf_api_token:
                raise ValueError("Please set the HF_API_TOKEN environment variable to use remote inference.")
            client = InferenceClient(token=hf_api_token, timeout=180)
            def remote_generate(prompt: str) -> str:
                response = client.text_generation(
                    prompt,
                    model=repo_id,
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
        elif "mistral-api" in normalized:
            debug_print("Creating Mistral API pipeline...")
            mistral_api_key = os.environ.get("MISTRAL_API_KEY")
            if not mistral_api_key:
                raise ValueError("Please set the MISTRAL_API_KEY environment variable to use Mistral API.")
            if not MISTRAL_AVAILABLE:
                raise ImportError("Mistral client library not installed. Install with: pip install mistralai")
            from langchain.llms.base import LLM
            class MistralLLM(LLM):
                temperature: float = 0.7
                top_p: float = 0.95
                _client: Any = PrivateAttr()  # Declare _client as a private attribute
                def __init__(self, api_key: str, temperature: float = 0.7, top_p: float = 0.95):
                    super().__init__()
                    self._client = Mistral(api_key=api_key)
                    self.temperature = temperature
                    self.top_p = top_p
                @property
                def _llm_type(self) -> str:
                    return "mistral_llm"
                def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                    response = self._client.chat.complete( 
                        model="mistral-small-latest",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=32000
                    )
                    return response.choices[0].message.content
                @property
                def _identifying_params(self) -> dict:
                    return {"model": "mistral-small-latest"}
            mistral_llm = MistralLLM(api_key=mistral_api_key, temperature=self.temperature, top_p=self.top_p)
            debug_print("Mistral API pipeline created successfully.")
            return mistral_llm
        else:
            # Default branch: assume Llama
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            extra_kwargs = {}
            if "llama" in normalized or model_id.startswith("meta-llama"):
                extra_kwargs["max_length"] = 4096
            pipe = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                device=-1,
                **extra_kwargs
            )
            from langchain.llms.base import LLM
            class LocalLLM(LLM):
                @property
                def _llm_type(self) -> str:
                    return "local_llm"
                def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                    # Reserve tokens for generation (e.g., 512 tokens)
                    reserved_gen = 512
                    max_total = 8192
                    max_prompt_tokens = max_total - reserved_gen
                    truncated_prompt = truncate_prompt(prompt, max_tokens=max_prompt_tokens)
                    generated = pipe(truncated_prompt, max_new_tokens=reserved_gen)[0]["generated_text"]
                    return generated
                @property
                def _identifying_params(self) -> dict:
                    return {"model": model_id, "max_length": extra_kwargs.get("max_length")}
            debug_print("Local Llama pipeline created successfully with max_length=4096.")
            return LocalLLM()

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
            else:
                debug_print(f"File type not supported for URL: {link}")
        if not self.raw_data:
            raise ValueError("No files were successfully loaded. Please check the URLs and file formats.")
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
        self.rag_prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.str_output_parser = StrOutputParser()
        debug_print("Selecting LLM pipeline based on choice: " + self.llm_choice)
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
        self.elevated_rag_chain = base_runnable | self.rag_prompt | self.llm | format_response
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

def submit_query_updated(query):
    debug_print("Inside submit_query function.")
    if not query:
        debug_print("Please enter a non-empty query")
        return "Please enter a non-empty query", "Word count: 0", f"Model used: {rag_chain.llm_choice}", ""
    if hasattr(rag_chain, 'elevated_rag_chain'):
        try:
            history_text = "\n".join([f"Q: {conv['query']}\nA: {conv['response']}" for conv in rag_chain.conversation_history]) if rag_chain.conversation_history else ""
            prompt_variables = {
                "conversation_history": history_text,
                "context": rag_chain.context,
                "question": query
            }
            if "llama" in rag_chain.llm_choice.lower():
                prompt_variables["context"] = truncate_prompt(prompt_variables["context"], max_tokens=4092)
            response = rag_chain.elevated_rag_chain.invoke(prompt_variables)
            rag_chain.conversation_history.append({"query": query, "response": response})
            input_token_count = count_tokens(query)
            output_token_count = count_tokens(response)
            return (
                response,
                rag_chain.get_current_context(),
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
        "Please load files first.",
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

with gr.Blocks(css=custom_css) as app:
    gr.Markdown('''# PhiRAG  
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
- Example: Select all parts in each book focusing on moral responsibility in Aristotle philosophy and discuss in a comprehensible way and link the topics to other top world philosophers. Use a structure and bullet points

The response displays the model used, word count, and current context (with conversation history).
''')
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
            load_button = gr.Button("Load Files")
    
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
            placeholder="Response will appear here (formatted as Markdown)",
            lines=6
        )
        context_output = gr.Textbox(
            label="Current Context",
            placeholder="Retrieved context and conversation history will appear here",
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
    
    model_dropdown.change(
        fn=update_model,
        inputs=model_dropdown,
        outputs=model_output
    )

if __name__ == "__main__":
    debug_print("Launching Gradio interface.")
    app.launch(share=False)
