# 1. Base Image: Use the full Python 3.10 image for best compatibility
FROM python:3.10

# 2. Add System Dependencies: Ensure essential tools are present
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set Working Directory for the build process (standard location)
WORKDIR /usr/src/app

# 4. Dependencies: Copy requirements from root and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code: Copy the contents of the 'src' folder (api.py, model.py, etc.) 
#    into the dedicated application directory (/app).
COPY src /app

# 6. Port
EXPOSE 8080

# 7. CRITICAL FIX: Use 'sh -c' to guarantee the PYTHONPATH is set 
#    and applied for the Gunicorn command.
CMD ["sh", "-c", "PYTHONPATH=/app gunicorn --bind 0.0.0.0:8080 api:app"]
