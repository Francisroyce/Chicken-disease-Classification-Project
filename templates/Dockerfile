# Use official lightweight Python image
FROM python:3.10.16-slim-buster

# Install system dependencies
RUN apt-get update -y && apt-get install -y \
    git \
    curl \
    build-essential \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port Azure expects
EXPOSE 8000

# Run using gunicorn for production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
