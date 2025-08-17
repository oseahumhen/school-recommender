import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

with open("env.txt") as fi:
    os.environ["OPENAI_API_KEY"] = fi.readline()
    os.environ["CHROMA_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

CHROMA_DB_CLIENT = chromadb.Client()
COLLECTION_NAME = "school_recommender"
COLLECTION = CHROMA_DB_CLIENT.get_or_create_collection(COLLECTION_NAME,
                                                       embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small"),
                                                       configuration={
                                                           "hnsw": {
                                                               "space": "l2",
                                                           }
                                                       }
                                                       )

CLIENT = OpenAI()

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
            metadatas=[{"source": f"{chunk.page_content[:100]}...",
                        "chunk_index": chunk_idx, "school_tag":school_tag}]
        )

def retrieve_context(query, school_tag):
    return COLLECTION.query(query_texts=[query], where={"school_tag": school_tag}, n_results=1)

def create_prompt(user_query, school_1_results, school_2_results):
    # to do
    return f"""
    <INSTRUCTION>
    Your task is to recommend a school from 2 school options based on list of desirable attributes found in USER_QUERY section
    Use ONLY relevant search results have been provided in SEARCH_RESULTS section.
    If unable to give a recommendation based on what relevant search results contain say so.
    SEARCH_RESULTS is split into two sections. SCHOOL_1 corresponds to school 1 while 
    SCHOOL_2 corresponds to school 2.
    Answer should be in tabular format
    columns should be the following:
    - column one is the list of attributes user wants in a school
    - column two should give school 1's ranking on corresponding attributes
    - column three should give school 2's ranking on corresponding attributes Use the text extract string associated with
      a search result in SEARCH_RESULTS
    - use a ranking scale of ["unable to determine", "low", "medium", "high"]
    - column four is a reference column. use metadata for the reference column. Column name should  be "Reference from docs uploaded"
      For example if source of information for SCHOOL_1 was "metadata 2: This is a ref text..."
      and SCHOOL_2 was "metadata 1: Another ref text.." then column four should be as below(each shool in a new line):
      1. school 1: This is a ref text...
      2. school 2: Another ref text..
      (a school which information was not found can be excluded from column four)
    - ranking weights are as follow:
      * "unable to determine" has a weight of 0
      * "low" has a weight of 1
      * "medium" has a weight of 2
      * "high" has a weight of 3
    - Sum the ranks for column two and column three
    - The school with higher rank sum should be recommended. Give a brief summary about recommendation
    If unable to rank any desired attributes due to insufficient information in SEARCH_RESULTS section assign the corresponding ranking column as "unable to determine".
    </INSTRUCTION>
    <USER_QUERY>
    {user_query}
    </USER_QUERY>
    <SEARCH_RESULTS>
    <SCHOOL_1>
    {school_1_results}
    </SCHOOL_1> 
    <SCHOOL_2>
    {school_2_results}
    </SCHOOL_2>
    </SEARCH_RESULTS>
    """
def format_search_results(retrival_results):
    final_str = ""
    for i in range(len(retrival_results)):
        doc = retrival_results[i]["documents"][0][0]
        metadata = retrival_results[i]["metadatas"][0][0]
        final_str += (f"search result {i+1}: {doc}\n "
                      f"metadata {i+1}: {metadata["source"]}\n")
    return final_str


def get_completion(user_prompt, model="gpt-4"):
    system_prompt = "You are a RAG assistant who uses results from a search strings to recommend a school from a list of 2 schools"
    completion = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    return completion.choices[0].message.content


system_prompt = "You are a helpful RAG search assistant who uses results from a search engine to answer user queries."

def main():
    st.title("School Recommender")
    st.write("Application that recommends a suitable school from a choice of 2 schools based on desired school "
             "attributes given.")

    school_1_docs = document_uploader("first school")
    school_2_docs = document_uploader("second school")

    add_to_collection(split_docs(school_1_docs), "school_one")
    add_to_collection(split_docs(school_2_docs), "school_two")

    # user_query = "i want a school that supports sporting activities"
    user_query = st.text_area("Write FIVE desired school attribute ONE ON EACH LINE", key="user_query")
    if st.button("Recommend School"):
        user_query_list = user_query.strip().split("\n")
        print(user_query_list)
        if len(user_query_list) > 5:
            st.write("Maximum of FIVE school attributes are supported. Please reduce attributes")
        else:
            retrieval_results_2 = []
            retrieval_results_1 = []
            for query in user_query_list:
                retrieval_results_1.append(retrieve_context(query, "school_one"))
                retrieval_results_2.append(retrieve_context(query, "school_two"))

            school_1_results = format_search_results(retrieval_results_1)
            school_2_results = format_search_results(retrieval_results_2)
            user_prompt = create_prompt(user_query, school_1_results, school_2_results)
            st.write(get_completion(user_prompt))

if __name__ == "__main__":
    main()