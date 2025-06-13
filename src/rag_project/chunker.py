import re
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from rag_project.utils import preprocess_text
from rag_project.exceptions import EmbeddingError
from typing import List
from rag_project.utils import load_config, model_loading
from rag_project.logger import logger
from rag_project.utils import model_loading

nlp = model_loading()

def chunk_and_preprocess_docs(docs: List[Document],lang_model=nlp) -> List[Document]:
    
    try:
        config = load_config()
        chunk_size = config['chunking']['chunk_size']
        chunk_overlap = config['chunking']['chunk_overlap']
        encoding = config['chunking']['encoding']

        

        cleaned_data = []

        for item in docs:
            cleaned_entry = {
            "id": item.get("id"),
            "title": preprocess_text(item.get("title", ""),nlp),
            "content": preprocess_text(item.get("publication_description", ""),nlp),
            "username": preprocess_text(item.get("username", ""),nlp)
            }
        cleaned_data.append(cleaned_entry)

        documents_with_metadata = [
            Document(page_content=item["content"], metadata={
                        "id": item["id"],
                        "title": item["title"],
                        "username": item["username"]
                    })
                    for item in cleaned_data
        ]
        logger.info("converted langchain document type.")

        logger.info(f"Chunking config: size={chunk_size}, overlap={chunk_overlap}, encoding={encoding}")

        token_splitter = TokenTextSplitter(
            encoding_name=encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunked_documents = []

        for doc in documents_with_metadata:
            chunks = token_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks, 1):
                chunked_doc = {
                    "id": doc.metadata.get("id", None),
                    "title": doc.metadata.get("title", None),
                    "username": doc.metadata.get("username", None),
                    "chunk_number": i,
                    "content": chunk
                }
                chunked_documents.append(chunked_doc)
        
        logger.info("Chunking finished.")

        langchain_chunks = [
            Document(page_content=doc["content"], metadata={
                "id": doc["id"],
                "title": doc["title"],
                "username": doc["username"],
                "chunk_number": doc["chunk_number"]
            })
            for doc in chunked_documents
        ]
        logger.info(f"Number of chunks: {len(langchain_chunks)}")
        return langchain_chunks

    except Exception as e:
        logger.exception("Error during chunking")
        raise EmbeddingError("Chunking failed") from e