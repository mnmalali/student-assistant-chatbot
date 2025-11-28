# student-assistant-chatbot
Student Assistant Chatbot (RAG Based Course Information Assistant)

This project implements a Student Assistant Chatbot that helps Computer Science students access detailed and verified information about university course structures, prerequisites, credit hours, and study plans using a Retrieval-Augmented Generation (RAG) pipeline.

The system uses OpenAI models, LangChain, and ChromaDB to ensure all responses are grounded in official study plan documents.

🚀 Features

Natural language question answering for CS course plans

Retrieval-Augmented Generation (RAG)

Multi-university support (UoS, CUD, AAU, KU, NYUAD)

Automatic PDF ingestion

Regex-based text chunking

Vector database search using MMR

Gradio web interface

Fully grounded, no hallucinations

📂 Project Structure
student-assistant-chatbot/
│
├── ingest_database.py
├── chatbot.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/                    # Study plan PDFs (not included for size)
└── screenshots/             # Optional images


🧠 How It Works

Data ingestion

PDFs loaded using PyPDFDirectoryLoader

Splits text into structured chunks

Embeds chunks using OpenAI embedding model

Stores in a persistent ChromaDB vector store

Chatbot

User asks a question

Retriever fetches the most relevant chunks

A strict system prompt forces grounded answers

GPT-4o-mini generates a clean, structured response

Interface

Simple Gradio chat UI

Streams responses in real time

🛠 Tech Stack

Python

LangChain

ChromaDB

OpenAI (GPT + embeddings)

Gradio

PDF loading + regex-based chunking


Add more universities

Add metadata-based citations

Deploy online (HuggingFace / Render / Streamlit Cloud)
