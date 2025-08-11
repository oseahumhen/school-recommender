import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

with open("env.txt") as fi:
    os.environ["OPENAI_API_KEY"] = fi.readline()
    os.environ["CHROMA_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

CHROMA_DB_CLIENT = chromadb.Client()
COLLECTION_NAME = "school_recommender"
COLLECTION = CHROMA_DB_CLIENT.get_or_create_collection(COLLECTION_NAME,
                                                       embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small"),
                                                       configuration={
                                                           "hnsw": {
                                                               "space": "ip",
                                                           }
                                                       }
                                                       )

def document_uploader(school_tag):
    uploaded_files = st.file_uploader(
        f"Upload {school_tag} documents", accept_multiple_files=True
    )
    docs = []
    for uploaded_file in uploaded_files:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        docs.append(text)
    return docs

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "? ", "! "],  # List of characters to split on
        chunk_size=600,  # The maximum size of your chunks
        chunk_overlap=50,  # The maximum overlap between chunks
    )
    return text_splitter.create_documents(docs)

def add_to_collection(chunks, school_tag):
    for index, chunk in enumerate(chunks):
        chunk_idx = f"{school_tag}_{index}"
        COLLECTION.add(
            ids=[f"chunk_{chunk_idx}"],
            documents=[chunk.page_content],
            metadatas=[{"source": f"{school_tag}: {chunk.page_content[:60]}...",
                        "chunk_index": chunk_idx, "school_tag":school_tag}]
        )

def retrieve_context(query, school_tag):
    return COLLECTION.query(query_texts=[query], where={"school_tag": school_tag}, n_results=3)

def create_prompt():
    # to do
    pass

def main():
    st.title("School Recommender")
    st.write("Application that recommends from the choice of 2 schools")
    school_1_docs = document_uploader("first school")
    school_2_docs = document_uploader("second school")
    add_to_collection(split_docs(school_1_docs), "school_one")
    add_to_collection(split_docs(school_2_docs), "school_two")
    st.write(retrieve_context("i want a school that supports sporting activities", "school_one"))
    st.write(retrieve_context("i want a school that supports sporting activities", "school_two"))


if __name__ == "__main__":
    main()