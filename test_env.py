import sys
import streamlit
import langchain
import chromadb
import pandas

print(f"Python version: {sys.version}")
print(f"Streamlit version: {streamlit.__version__}")
print(f"LangChain version: {langchain.__version__}")
print(f"ChromaDB version: {chromadb.__version__}")
print(f"Pandas version: {pandas.__version__}")

print("\nAll key dependencies imported successfully!")
