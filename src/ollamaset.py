from langchain_ollama import ChatOllama

# Initialize the local LLM
local_llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434")


