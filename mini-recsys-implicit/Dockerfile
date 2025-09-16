# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir .

# Make port 8000 available
EXPOSE 8000

# Run the service
CMD ["python", "scripts/serve.py"]