FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files into container
COPY . .

# ✅ Set PYTHONPATH to include src folder
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose Flask port
EXPOSE 10000

# Start Flask app with Gunicorn
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
