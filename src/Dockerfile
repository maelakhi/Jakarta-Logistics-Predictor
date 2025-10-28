# 1. Base Image: Force the use of Python 3.10
FROM python:3.10-slim

# 2. Add System Dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Working Directory: Set the primary folder inside the container
WORKDIR /app

# 4. Dependencies: Copy requirements and install the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code: Copy the rest of your files
COPY . .

# 6. Port: Tell the container to listen on port 8080.
EXPOSE 8080

# 7. Command: Define the final command to run the application using Gunicorn.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]