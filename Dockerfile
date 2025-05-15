FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY images/ ./images

CMD ["streamlit", "run", "src/main.py", "--server.port=10015", "--server.address=0.0.0.0"]