```bash
# avoid the error ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai' 
uvx --with "langchain-community<0.4" --with "langchain-google-vertexai" ragas quickstart rag_eval
```