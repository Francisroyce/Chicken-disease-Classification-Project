FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TensorFlow and Git
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copy the entire project first (so setup.py is available)
COPY . .

# ✅ Set PYTHONPATH to include src folder
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# ✅ Now install Python dependencies including editable installs (-e .)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Flask port
EXPOSE 10000

# Start Flask app with Gunicorn
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
