# Crop Recommendation – container image (API + Streamlit app share the same code)
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY recommender.py api.py app.py ./
COPY models ./models
COPY data ./data

# Run as a non-root user
RUN useradd -m appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/health')" || exit 1

# Default: REST API.  For the Streamlit UI override the command (see docker-compose.yml)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
