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

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all files and set PYTHONPATH
COPY . .
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose default port
EXPOSE 10000

# Start the app using a dynamic port (Render compatible)
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
