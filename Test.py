#test.py
import streamlit as st
import os
from Utils import* 
def main():
    st.set_page_config(page_title="PDF summarizer")
    st.title("PDF summarizing app")
    st.write("summarize your PDF file in just a few seconds.")
    st.divider()
    PDF = st.file_uploader('upload your PDF doc',type='PDF')
    submit=st.button("generate summary")
    os.environ["langchain"] = ""
    response = None
    if submit:
        response = summarizer(PDF)
    st.subheader('summary of file:')
    st.write(response)
if __name__ == '__main__':
    main()
