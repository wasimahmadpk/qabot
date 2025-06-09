# Qabot – Question Answering Over Your Documents

Qabot is a simple and interactive app that allows you to upload your own documents (PDF, TXT, DOCX) and ask natural language questions about their content. It uses OpenAI's GPT models combined with LlamaIndex to retrieve relevant information from your files.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/wasimahmadpk/qabot.git
cd qabot

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key to a .env file
echo "OPENAI_API_KEY=your-openai-key-here" > .env

# Run the app
streamlit run app.py
