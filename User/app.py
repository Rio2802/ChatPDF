import boto3
import streamlit as st
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# AWS and Langchain setup
s3_client = boto3.client("s3")
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Environment & Config
BUCKET_NAME = os.getenv("BUCKET_NAME")
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
LLM_MODEL_ID = "anthropic.claude-v2:1"
TEMP_FOLDER = "/tmp/"

# Embeddings
bedrock_embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, client=bedrock_client)

# Load FAISS index from S3
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{TEMP_FOLDER}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{TEMP_FOLDER}my_faiss.pkl")

# Load LLM
def get_llm():
    return Bedrock(
        model_id=LLM_MODEL_ID,
        client=bedrock_client,
        model_kwargs={'max_tokens_to_sample': 512}
    )

# Get response from Claude using RAG
def get_response(llm, vectorstore, question):
    prompt_template = """
Human: Please use the given context to provide a concise and helpful answer.
If you don't know the answer, just say that you don't know. Don't try to make it up.

<context>
{context}
</context>

Question: {question}

Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain({"query": question})
    return result['result'], result.get("source_documents", [])

# Main UI
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📘")
    st.title("📘 Chat with Your PDF")
    st.markdown("This assistant lets you query a PDF document using RAG powered by AWS Bedrock & FAISS.")

    # Load Index from S3
    with st.spinner("Loading knowledge base..."):
        load_index()

    # Load FAISS Index
    vectorstore = FAISS.load_local(
        index_name="my_faiss",
        folder_path=TEMP_FOLDER,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    st.success("Vector store loaded successfully ✅")

    question = st.text_input("🔎 Ask a question from the PDF")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question to proceed.")
            return

        with st.spinner("💡 Thinking..."):
            llm = get_llm()
            answer, sources = get_response(llm, vectorstore, question)

        st.subheader("📌 Answer")
        st.write(answer)

        if sources:
            with st.expander("🧠 Source(s) used"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}**")
                    st.text(doc.page_content.strip()[:1000])  # Avoid very long output

if __name__ == "__main__":
    main()
