# Dockerfile (at project root)

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

# Copy dependency files and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose the port Render uses
EXPOSE 10000

# Default start command
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
