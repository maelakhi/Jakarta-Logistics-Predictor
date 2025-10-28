# 1. Base Image: Use the full Python 3.10 image for best compatibility
FROM python:3.10

# 2. Add System Dependencies: Ensure essential tools are present
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set Working Directory for the build process
WORKDIR /usr/src/app

# 4. Dependencies: Copy the minimal requirements list from the root and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code: Copy the 'src' folder (containing api.py/model.py) 
#    into the container's final application directory (/app)
COPY src /app

# 6. CRITICAL PATH FIX: Add /app to the PYTHONPATH
# This tells Python to search the /app directory for modules, resolving the ModuleNotFoundError.
ENV PYTHONPATH=/app

# 7. Port
EXPOSE 8080

# 8. Command: Run the application using Gunicorn.
# The command is clean because the PYTHONPATH handles the discovery.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]
