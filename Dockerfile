# 1. Base Image: Force the use of Python 3.10 to ensure compatibility with ML packages
FROM python:3.10-slim

# 2. Add System Dependencies: Essential tools for compiling NumPy/scikit-learn
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Working Directory
WORKDIR /app

# 4. Dependencies: Copy requirements from root and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code: Copy the 'src' folder (which contains api.py/model.py) 
#    into the container's working directory (/app)
COPY src /app

# 6. Port
EXPOSE 8080

# 7. Command: Run the application using Gunicorn.
#    Since api.py is now inside /app, Gunicorn runs it successfully.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]
