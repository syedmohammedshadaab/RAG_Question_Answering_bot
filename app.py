import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from langchain_experimental.data_anonymizer import PresidioAnonymizer

# Function to extract text and tables from the PDF
def extract_text_and_tables(pdf):
    text = ""
    tables = []
    with pdfplumber.open(pdf) as pdf_document:
        # Iterate through each page in the PDF
        for page in pdf_document.pages:
            # Extract text and tables from the current page
            text += page.extract_text() or ""
            tables.extend(page.extract_tables())
    return text, tables

def main():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Ask your PDF")
    
    # Load environment variables from a .env file
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Initialize Presidio Anonymizer for sensitive data anonymization
    anonymizer = PresidioAnonymizer(analyzed_fields=["PHONE_NUMBER"])
    
    # Display the header of the Streamlit app
    st.header("Ask your PDF 💭")
    
    # Sidebar description for the app
    with st.sidebar:
        st.markdown('''
                    Upload PDF and ask questions related to the content in the PDF.
         ''')
    
    # File uploader widget to upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        # Extract text and tables from the uploaded PDF
        text, tables = extract_text_and_tables(pdf)
        
        if text:
            # Anonymize extracted text to protect sensitive data
            anonymized_text = anonymizer.anonymize(text)
            st.write("Anonymized Text:")
            st.text(anonymized_text)
            
            # Display extracted tables
            if tables:
                st.write("Extracted Tables:")
                for i, table in enumerate(tables):
                    st.write(f"Table {i+1}:")
                    st.write(table)
            else:
                st.write("No tables found in the PDF.")
            
            # Split the text into manageable chunks for processing
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(anonymized_text)
            
            # Initialize embeddings using OpenAI's "text-embedding-ada-002" model
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            
            # Create a vector store using FAISS for fast retrieval of relevant text chunks
            vectorestore = FAISS.from_texts(chunks, embeddings)
            
            # Input widget for the user to ask a question about the PDF content
            user_question = st.text_input("Ask a question about your PDF")
            if user_question:
                # Initialize a retriever to get relevant chunks from the vector store
                retriever = vectorestore.as_retriever()
                
                # Initialize the LLM (GPT-3.5-turbo) with specified parameters
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                
                # Define the prompt template to guide the model's responses
                template = """You are a helpful, respectful, and honest assistant.
                              Always answer as helpfully as possible, while being safe.
                              Your answers should not include any harmful, unethical, racist, sexist, toxic,
                              dangerous, or illegal content. Please ensure that your responses are socially
                              unbiased and positive in nature.
                              If a question does not make any sense, or is not factually coherent,
                              explain why instead of answering something not correct.
                              If you don't know the answer to a question, please don't share false information.
                              Question: {question}
                              Context: {context}
                              Answer:  
                           """
                
                # Create the prompt and chain it to the model
                prompt = ChatPromptTemplate.from_template(template)
                
                # Define the RAG (Retrieval-Augmented Generation) chain
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Display the question and generate the answer when the user clicks "Submit"
                if st.button("Submit"):
                    answer = rag_chain.invoke(user_question)
                    st.write(f"**Question:** {user_question}")
                    st.write(f"**Answer:** {answer}")
                
                # Example questions and ground truths for evaluation
                questions = ["what are the basic cause Fire and BLEVE", "what are the root causes Fire and BLEVE"]
                
                ground_truths = [
                    ["Basic cause of first fire was ignition of a vapour cloud formed by accidental release of a large quantity of propane from an open drain. Basic cause of first BLEVE was fire engulfment and overheating of the sphere."],
                    ["Root causes included: 1) Failure to follow operating procedure (drain valve operating sequence), 2) Inadequate storage sphere design (support legs not reinforced), 3) Inadequate drain system design (removable valve handles, open discharge in close proximity to valves), 4) Inadequate overpressure protection (absence of remote depressuring valve), 5) Insufficient active (water spray) and passive (insulation) fire protection, 6) Failure to train local fire brigade on how to deal with this type of incident."]
                ]
                
                answers = []
                contexts = []
                
                # Retrieve and store answers and contexts for the given questions
                for query in questions:
                    answer = rag_chain.invoke(query)
                    answers.append(answer)
                    contexts.append([doc.page_content for doc in retriever.get_relevant_documents(query)])
                
                # Prepare the data for evaluation
                data = {
                    "question": questions,
                    "answer": answers,
                    "contexts": contexts,
                    "ground_truths": ground_truths
                }
                
                # Create a Dataset object from the data
                dataset = Dataset.from_dict(data)
                
                # Evaluate the model's performance using specified metrics
                result = evaluate(
                    dataset=dataset,
                    metrics=[
                        context_precision,
                        context_recall,
                        faithfulness,
                        answer_relevancy
                    ],
                )
                
                # Convert the evaluation results to a DataFrame and display it
                df = result.to_pandas()
                st.write(df)
        else:
            st.error("No text found in the PDF.")
    else:
        st.warning("Please upload a PDF file.")

if __name__ == '__main__':
    main()
