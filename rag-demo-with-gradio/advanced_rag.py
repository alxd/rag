import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List

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

# ----------------------------
# LLM Pipeline Creators
# ----------------------------

def create_deepseek_pipeline() -> HuggingFacePipeline:
    """
    Create a HuggingFace pipeline using the DeepSeek-R1 model and wrap it as a LangChain LLM.
    ERROR: https://github.com/deepseek-ai/DeepSeek-V3/issues/558
    """
    quant_config = AutoQuantizationConfig.from_dict({
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "bitsandbytes_4bit",
    "weight_block_size": [128, 128]
})
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1", 
        trust_remote_code=True,
        quantization_config=quant_config #  supported types are: ['awq', 'bitsandbytes_4bit', 'bitsandbytes_8bit', 'gptq', 'aqlm', 'quanto', 'eetq', 'higgs', 'hqq', 'compressed-tensors', 'fbgemm_fp8', 'torchao', 'bitnet', 'vptq']
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        trust_remote_code=True,
        max_length=2048,
        do_sample=True,
        temperature=0.5,
        top_p=1,
        device=0 if torch.cuda.is_available() else -1
    )
    return HuggingFacePipeline(pipeline=pipe)

def create_llama3_pipeline() -> HuggingFacePipeline:
    """
    Create a HuggingFace pipeline using Meta-Llama-3-8B-Instruct and wrap it as a LangChain LLM.
    To use this, first download the model with:
    
    ```
    huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct
    ```
    ACCESS ISSUE - Agree with license: https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/172 
    ACCESS ISSUE - Pass huggingface token: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/68 
    ACCESS ISSUE - WRITE token required: https://discuss.huggingface.co/t/loading-llama-3/90492
    Adjust device settings as needed.
    """
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #model_id = 'meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48'
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=0 if torch.cuda.is_available() else -1,
        max_length=2048,
        do_sample=True,
        temperature=0.5,
        top_p=1
    )
    return HuggingFacePipeline(pipeline=pipe)

def create_llm_pipeline(choice: str = "llama3") -> HuggingFacePipeline:
    """
    Wrapper to choose between 'deepseek' and 'llama3' pipelines.
    """
    if choice.lower() == "llama3":
        print ("Using Meta Llama-3")
        return create_llama3_pipeline()        
    else:
        print ("Using DeepSeek R1")
        return create_deepseek_pipeline()

# ----------------------------
# ElevatedRagChain Class
# ----------------------------

class ElevatedRagChain:
    """
    ElevatedRagChain builds an advanced retrieval-augmented generation (RAG) system.
    It processes PDFs by loading, chunking, embedding, and indexing in a FAISS vector store.
    It uses an ensemble retriever (BM25 + FAISS) and a configurable LLM (DeepSeek or Meta-Llama-3)
    to generate detailed technical answers.
    """
    def __init__(self, llm_choice: str = "llama3") -> None:
        """
        Initialize with a HuggingFaceEmbeddings instance (using an open source SentenceTransformer),
        retriever weights, top_k value, and select the LLM pipeline based on llm_choice.
        
        llm_choice: "deepseek" (default) or "llama3"
        """
        self.embed_func   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.bm25_weight  = 0.6
        self.faiss_weight = 0.4
        self.top_k        = 5
        self.llm_choice   = llm_choice  # Save the LLM choice for later use

    def add_pdfs_to_vectore_store(
            self,
            pdf_links: List,
            chunk_size: int = 1500,
        ) -> None:
        """
        Processes PDF documents by loading, chunking, embedding, and adding them to a FAISS vector store.
        """
        self.raw_data = [OnlinePDFLoader(doc).load()[0] for doc in pdf_links]
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        self.split_data    = self.text_splitter.split_documents(self.raw_data)
        self.bm25_retriever = BM25Retriever.from_documents(self.split_data)
        self.bm25_retriever.k = self.top_k
        self.vector_store    = FAISS.from_documents(self.split_data, self.embed_func)
        self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        print("All PDFs processed and added to vector store.")
        self.build_elevated_rag_system()
        print("RAG system is built successfully.")

    def build_elevated_rag_system(self) -> None:
        """
        Build the RAG system by chaining:
          - An ensemble retriever (BM25 + FAISS)
          - A prompt template
          - The selected LLM (DeepSeek or Meta-Llama-3) via a HuggingFace pipeline
        """
        ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.faiss_weight]
        )
        base_runnable = RunnableParallel(
            {
                "context": ensemble,
                "question": RunnablePassthrough()
            }
        )
        self.rag_prompt = ChatPromptTemplate.from_template(
            """\
Use the following context to provide a detailed technical answer to the user's question.
Do not include an introduction like "Based on the provided documents, ...". Just answer the question.
If you don't know the answer, please respond with "I don't know".

Context:
{context}

User's question:
{question}
"""
        )
        self.str_output_parser = StrOutputParser()
        # Select the LLM pipeline based on the llm_choice provided during initialization
        self.llm = create_llm_pipeline(choice=self.llm_choice)
        self.elevated_rag_chain = base_runnable | self.rag_prompt | self.llm
