import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List

# Imports for our DeepSeek model pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

# Other LangChain and community imports
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# Replace CohereEmbeddings with HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def create_deepseek_pipeline() -> HuggingFacePipeline:
    """
    Create a HuggingFace pipeline using the DeepSeek-R1 model and wrap it as a LangChain LLM.
    """
    # Load the DeepSeek model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
    
    # Create a text-generation pipeline.
    # Adjust parameters like max_length, temperature, and top_p as needed.
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        trust_remote_code=True,
        max_length=2048,
        do_sample=True,
        temperature=0.5,
        top_p=1
    )
    
    # Wrap the pipeline with HuggingFacePipeline for LangChain compatibility
    return HuggingFacePipeline(pipeline=pipe)

class ElevatedRagChain:
    """
    ElevatedRagChain integrates various components from LangChain to build an advanced
    retrieval-augmented generation (RAG) system. It processes PDF documents by loading,
    chunking, embedding, and adding their embeddings to a FAISS vector store for efficient
    retrieval. It then uses an ensemble retriever (BM25 + FAISS) and a DeepSeek model (via a
    HuggingFace pipeline) for generating detailed technical answers.
    """
    def __init__(self) -> None:
        """
        Initialize the class with a predefined embedding function, weights, and top_k value.
        """
        # Use HuggingFaceEmbeddings with a model that doesn't require an API key.
        self.embed_func   = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.bm25_weight  = 0.6
        self.faiss_weight = 0.4
        self.top_k        = 5

    def add_pdfs_to_vectore_store(
            self,
            pdf_links: List,
            chunk_size: int = 1500,
        ) -> None:
        """
        Processes PDF documents by loading, chunking, embedding, and adding them to a FAISS vector store.
        
        Args:
            pdf_links (List): List of URLs pointing to the PDF documents to be processed.
            chunk_size (int, optional): Size of text chunks to split the documents into (default: 1500).
        """        
        # Load PDFs
        self.raw_data = [OnlinePDFLoader(doc).load()[0] for doc in pdf_links]

        # Chunk text
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        self.split_data    = self.text_splitter.split_documents(self.raw_data)

        # Create BM25 retriever from the split documents
        self.bm25_retriever = BM25Retriever.from_documents(self.split_data)
        self.bm25_retriever.k = self.top_k

        # Embed and add chunks to FAISS vector store
        self.vector_store    = FAISS.from_documents(self.split_data, self.embed_func)
        self.faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        print("All PDFs processed and added to vector store.")
        
        # Build the advanced RAG system
        self.build_elevated_rag_system()
        print("RAG system is built successfully.")

    def build_elevated_rag_system(self) -> None:
        """
        Build an advanced RAG system by combining:
         - BM25 retriever
         - FAISS vector store retriever
         - A DeepSeek model (via a HuggingFace pipeline)
        
        Note: The retrieval is performed using an ensemble of BM25 and FAISS retrievers
        without applying any additional reranking.
        """
        # Combine BM25 and FAISS retrievers into an ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.faiss_weight]
        )

        # Define the prompt template for the language model
        RAG_PROMPT_TEMPLATE = """\
Use the following context to provide a detailed technical answer to the user's question.
Do not include an introduction like "Based on the provided documents, ...". Just answer the question.
If you don't know the answer, please respond with "I don't know".

Context:
{context}

User's question:
{question}
"""
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.str_output_parser = StrOutputParser()

        # Prepare parallel execution of context retrieval and question processing
        # Use the ensemble retriever directly (without reranking)
        self.entry_point_and_elevated_retriever = RunnableParallel(
            {
                "context": self.ensemble_retriever,
                "question": RunnablePassthrough()
            }
        )

        # Initialize the DeepSeek model using a HuggingFace pipeline as our LLM
        self.llm = create_deepseek_pipeline()

        # Chain the components to form the final elevated RAG system.
        # Optionally, you can append self.str_output_parser if output parsing is needed.
        self.elevated_rag_chain = self.entry_point_and_elevated_retriever | self.rag_prompt | self.llm
