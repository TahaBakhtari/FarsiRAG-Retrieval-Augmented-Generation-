# FarsiRAG (Retrieval Augmented Generation)

FarsiRAG is a question answering system designed for Persian (Farsi) PDF documents. It leverages state-of-the-art language models and vector search to retrieve relevant information and generate answers in Persian.

## Features
- Extracts and normalizes text from Persian PDF files
- Splits text into manageable chunks for processing
- Generates embeddings using Sentence Transformers
- Uses FAISS for efficient similarity search
- Answers user questions using OpenAI models via OpenRouter

## Requirements
- Python 3.7+
- PyPDF2
- sentence-transformers
- faiss
- hazm
- openai
- re

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your Persian PDF file (e.g., `math.pdf`) in the project directory.
2. Update the `pdf_path` variable in `run.py` if needed.
3. Run the script:
```bash
python run.py
```
4. Enter your question in Persian when prompted.

## Notes
- You need an API key for OpenAI via OpenRouter. Set it in `run.py`.
- This project is intended for educational and research purposes.

## License
MIT License
