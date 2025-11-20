# Use the official Playwright Python image so all browser deps are preinstalled
# You can keep this tag or bump it later if needed.
FROM mcr.microsoft.com/playwright/python:jammy

# Avoid Python buffering & bytecode files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Working directory inside the container
WORKDIR /app

# Install system locales / fonts if needed (optional but nice for PDFs, etc.)
# RUN apt-get update && apt-get install -y \
#     fonts-liberation \
#     && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better build caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Ensure browsers are installed (idempotent on this base image but safe)
RUN playwright install --with-deps

# Now copy the actual application code
COPY app ./app
COPY README.md ./
COPY LICENSE ./

# Expose FastAPI port
EXPOSE 8000

# Environment variables for your app (values are overridden at runtime)
# Do NOT hardcode secrets here; set them via -e / env files when running.
ENV MAX_TOTAL_SECONDS=170

# Default command: run uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
