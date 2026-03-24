# 🤖 Sensor GPT — ELCIA Center of Excellence

> A RAG-powered AI Chatbot that helps hardware engineers instantly find the right sensor based on specifications, protocols, and environmental conditions.

---

## 🚀 Features

- 💬 **Natural language queries** — Ask questions like *"What is the best low-cost I2C temperature sensor for high-humidity environments?"*
- 🔍 **RAG Pipeline** — Retrieval-Augmented Generation using LangChain + ChromaDB
- 🧠 **LLM Support** — Google Gemini, OpenAI GPT, or Local (no-key) prototype mode
- 📊 **Rich sensor database** — 30+ sensors across temperature, pressure, motion, gas, distance categories
- 🖥️ **Streamlit UI** — Clean, interactive chat interface

---

## 🛠️ Tech Stack

| Component      | Technology                          |
|----------------|-------------------------------------|
| Frontend       | Streamlit                           |
| RAG Framework  | LangChain                           |
| Vector DB      | ChromaDB (local)                    |
| Embeddings     | HuggingFace `all-MiniLM-L6-v2`     |
| LLM            | Google Gemini / OpenAI / Local Mock |
| Data           | Synthetic CSV sensor dataset        |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/KrishnaHarish/RIT-MCA-4sem-2026.git
cd RIT-MCA-4sem-2026/SensorGPT
```

### 2. Create a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the sensor dataset
```bash
python data_generator.py
```

### 5. Ingest data into ChromaDB
```bash
python ingest.py
```

### 6. Run the Streamlit app
```bash
streamlit run app.py
# OR on Windows with venv:
.\venv\Scripts\python.exe -m streamlit run app.py
```

---

## 🗂️ Project Structure

```
SensorGPT/
├── app.py               # Streamlit chatbot UI
├── data_generator.py    # Synthetic sensor dataset generator
├── ingest.py            # ChromaDB ingestion pipeline
├── requirements.txt     # Python dependencies
├── sensors_dataset.csv  # Generated sensor data (after running data_generator.py)
├── chroma_db/           # Vector store (after running ingest.py)
└── README.md
```

---

## 💡 Usage Modes

| Mode           | How to use                              |
|----------------|-----------------------------------------|
| Local (No Key) | Select "Local Prototype" in sidebar     |
| Google Gemini  | Paste your Gemini API key in sidebar    |
| OpenAI GPT     | Paste your OpenAI API key in sidebar    |

---

## 📌 Example Queries

- *"Recommend a sensor for measuring CO2 levels indoors"*
- *"Which sensors support SPI protocol and work below -20°C?"*
- *"What's the cheapest proximity sensor for a robotics project?"*
- *"Find a motion sensor compatible with 3.3V systems"*

---

## 🏫 ELCIA Center of Excellence — RIT, Bengaluru

This project was developed as part of the **Internship Programme at ELCIA Center of Excellence**, **Ramaiah Institute of Technology**, fulfilling the academic internship requirement for the MCA programme. It demonstrates applied AI/ML (RAG, LLMs, vector databases) for embedded systems and IoT engineering use cases.
