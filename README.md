# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run via Docker
docker build -t rag-gemini .
docker run -p 8501:8501 --env-file .env rag-gemini
