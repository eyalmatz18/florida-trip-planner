# Production-ready Dockerfile for the 10-Day Florida Family Trip Planner ADK Agent
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered structured JSON logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FAST_TOOL_MODEL="gemini-2.5-flash" \
    MASTER_SYNTHESIS_MODEL="gemini-2.5-pro"

WORKDIR /app

# Create a non-root user for security best practices
RUN groupadd -r adkuser && useradd -r -g adkuser adkuser

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY observability.py tools.py agent.py agent.yaml README.md /app/
COPY tests/ /app/tests/

RUN chown -R adkuser:adkuser /app
USER adkuser

# Note: GEMINI_API_KEY is injected securely at runtime via environment variables / Secret Manager
CMD ["python", "agent.py"]
