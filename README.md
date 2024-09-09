# RAG-based Q&A Chatbot
## Overview
This project is a **RAG (Retrieval-Augmented Generation) based Q&A Chatbot** built using Streamlit. The chatbot allows users to upload a PDF document and ask questions about its content. It processes the PDF by extracting text and tables, anonymizing sensitive data, and generating answers based on the content of the document. The chatbot is powered by OpenAI's GPT-3.5-turbo model and uses FAISS for efficient text retrieval.
## Features
- **PDF Upload**: Upload PDFs and ask questions related to the content.
- **Text and Table Extraction**: Extract text and tables from the PDF using `pdfplumber`.
- **Data Anonymization**: Anonymize sensitive information (e.g., phone numbers) using `PresidioAnonymizer`.
- **RAG (Retrieval-Augmented Generation)**: Retrieve relevant text chunks from the document using FAISS and generate answers using OpenAI's GPT-3.5-turbo model.
- **Evaluation**: Assess the chatbot's answers with metrics like context precision, context recall, faithfulness, and answer relevancy using `ragas`.
## Tech Stack

- **Streamlit**: Web-based interface for uploading PDFs and interacting with the chatbot.
- **pdfplumber**: Extract text and tables from PDF documents.
- **LangChain**: Framework for building LLM-powered applications with a focus on RAG.
- **FAISS**: Vector storage for efficient retrieval of relevant document chunks.
- **OpenAI Embeddings**: Embed document chunks using the `text-embedding-ada-002` model.
- **Presidio Anonymizer**: Protect sensitive data by anonymizing phone numbers.
- **ragas**: Evaluate the chatbot's answer quality based on custom metrics.

