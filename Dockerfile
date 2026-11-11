# Recommendation API image (CPU-only PyTorch).
#
# Data and trained models are NOT baked into the image (the MovieLens licence forbids
# redistribution); mount them at runtime:
#   docker build -t movie-recommender .
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/artifacts:/app/artifacts:ro" \
#     -v "$(pwd)/data/processed:/app/data/processed:ro" \
#     movie-recommender
FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU wheels are several times smaller than the default CUDA build.
ARG TORCH_VERSION=2.15.0
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 1000 app

USER app
ENV RECSYS_ARTIFACTS_DIR=/app/artifacts \
    RECSYS_DATA_DIR=/app/data/processed

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]

CMD ["uvicorn", "recsys.api:app", "--host", "0.0.0.0", "--port", "8000"]
