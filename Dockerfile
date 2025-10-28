# 1. Base Image: Use the full Python 3.10 image for best compatibility
FROM python:3.10

# 2. Add System Dependencies: Essential tools for compiling NumPy/scikit-learn
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Working Directory (Where future commands will run)
WORKDIR /app

# 4. Dependencies: Copy the minimal requirements list from the root and install
COPY requirements.txt .
# Install all production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code: Copy the 'src' folder (containing api.py/model.py) into the container's /app folder
COPY src /app

# 6. Port
EXPOSE 8080

# 7. Command: Run the application using Gunicorn.
# CRITICAL FIX: The --chdir /app forces Gunicorn to navigate to the correct folder 
# before attempting to import 'api:app'.
CMD ["gunicorn", "--chdir", "/app", "--bind", "0.0.0.0:8080", "api:app"]
