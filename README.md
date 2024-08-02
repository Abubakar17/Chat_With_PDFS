Now the app is simple to run, just dont forget to insert your own GOOGLE API key in a new .env file to make sure the llm is configured.

Differences in apps:

1- Normal Q&A: Uses a load_qa_chain as the main QA chain and the history of conversations isnt displayed in the streamlit application.
2- Q&A with history: same method as 1 but the history is mantained and displayed this time around.
3- Different Retriever: as the name suggests the load_qa_chain is not used, instead a retriever is used.

Further:
- Gemini turbo 1.5 model is used as LLM (cuz its free)
- Google genai embeddings are used just because gemini llm was used
- FAISS vector store is used (Chromadb was the next best choice)
- vector store is stored locally
- only answer/response is displayed (to display context just remove the ["output_text"] after response in st.write.
