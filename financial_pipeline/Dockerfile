FROM python:3.12-slim

# Build deps for numpy/scipy C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/output

# Cron entrypoint: override CMD in docker-compose or crontab
CMD ["python", "main.py"]
