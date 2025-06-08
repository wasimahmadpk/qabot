from llama_index.core import Document
import fitz  # PyMuPDF
import os

def load_documents():
    docs = []
    folder_path = "./docs"

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            doc = fitz.open(path)

            full_text = ""
            for page in doc:
                full_text += page.get_text()

            docs.append(Document(text=full_text, metadata={"file_name": filename}))

    return docs
