from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pypdf import PdfReader

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformer/all-minilm-l6-v2')
    knowledgebase = FAISS.from_texts(chunks, embeddings)
    return knowledgebase

def summarizer(pdf):
    if pdf is not None:
        if isinstance(pdf, str):
            # If `pdf` is a file path
            pdf_reader = PdfReader(pdf)
        else:
            # If `pdf` is a file-like object
            pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        knowledgebase = process_text(text)
        query = "summarize the content of the uploaded PDF file in approximately 3-5 sentences."
        if query:
            docs = knowledgebase.similarity_search(query)
            OpenAIModel = 'gpt-3.5-turbo-16k'
            llm = ChatOpenAI(model=OpenAIModel, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')
            response = chain.run(input_documents=docs, question=query)
        return response
from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarizer(pdf):
    if pdf is not None:
        if isinstance(pdf, str):
            # If `pdf` is a file path
            pdf_reader = PdfReader(pdf)
        else:
            # If `pdf` is a file-like object
            pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# Rest of the code remains unchanged

# Rest of the code remains unchanged

# Below is the Streamlit script
import streamlit as st
import os

def main():
    st.set_page_config(page_title="PDF summarizer")
    st.title("PDF summarizing app")
    st.write("Summarize your PDF file in just a few seconds.")
    st.divider()
    PDF = st.file_uploader('Upload your PDF doc', type='PDF')
    submit = st.button("Generate Summary")
    os.environ["langcahin"] = ""
    response = None
    if submit:
        response = summarizer(PDF)
    st.subheader('Summary of file:')
    st.write(response)

if __name__ == '__main__':
    main()
