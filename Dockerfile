FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-render.txt .
RUN pip install --upgrade pip && pip install -r requirements-render.txt

COPY . .

EXPOSE 10000
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 0 --bind 0.0.0.0:${PORT:-10000} web_app:app"]
