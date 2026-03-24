"""
ingest.py — Sensor GPT | ELCIA COE
Reads sensors_dataset.csv and indexes it into a local ChromaDB vector store.
"""

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./chroma_db"
CSV_PATH = "./sensors_dataset.csv"


def load_documents(csv_path: str) -> list[Document]:
    """Convert each sensor row into a LangChain Document."""
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = row.get("Description", "")
        metadata = {
            "sensor_name": str(row.get("Sensor Name", "")),
            "sensor_type": str(row.get("Sensor Type", "")),
            "protocol": str(row.get("Communication Protocol", "")),
            "i2c": str(row.get("I2C Compatible", "")),
            "spi": str(row.get("SPI Compatible", "")),
            "voltage": str(row.get("Operating Voltage", "")),
            "cost_usd": str(row.get("Cost (USD)", "")),
            "environment": str(row.get("Target Environment", "")),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def ingest():
    if not os.path.exists(CSV_PATH):
        print("❌ sensors_dataset.csv not found. Run data_generator.py first.")
        return

    print("📄 Loading sensor documents...")
    docs = load_documents(CSV_PATH)
    print(f"   Loaded {len(docs)} sensor records.")

    print("🔢 Initializing embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("📦 Ingesting into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ ChromaDB populated at '{CHROMA_DIR}' with {len(docs)} entries.")
    return vectorstore


if __name__ == "__main__":
    ingest()
