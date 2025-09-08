FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y \
        git \
        aspell \
        aspell-en \
        aspell-ru \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app
WORKDIR /app

COPY requirements.txt requirements_qa.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements_qa.txt -r requirements.txt

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
