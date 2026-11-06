FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt && useradd --create-home --uid 10001 appuser
COPY visionguard ./visionguard
COPY wsgi.py .
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4)"
CMD ["gunicorn","--bind","0.0.0.0:8000","--workers","1","--threads","2","--timeout","120","--access-logfile","-","wsgi:app"]
