# 1. Base Image: Start from Python 3.10 slim
FROM python:3.10-slim

# 2. Add System Dependencies (NEW STEPS)
# 'apt-get update' refreshes the list of packages.
# 'build-essential' installs all the C/C++ tools needed to compile libraries like numpy/scikit-learn.
# 'apt-get clean' removes temporary files to keep the final image small.
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
