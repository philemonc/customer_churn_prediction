# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install streamlit
RUN pip install matplotlib
RUN pip install xgboost
RUN pip install scikit-learn
COPY app/streamlit_app.py /app/streamlit_app.py
COPY app/bank_customer_churn.csv /app/bank_customer_churn.csv
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]