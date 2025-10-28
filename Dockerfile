# 1. Base Image: Use the full Python 3.10 image for best compatibility
FROM python:3.10

# 2. Add System Dependencies: Ensure essential tools are present
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. CRITICAL FIX: Set the working directory to /app immediately.
# This folder will serve as the Python search path for the running application.
WORKDIR /app

# 4. Dependencies: Copy requirements from root and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code: Copy the CONTENTS of your local 'src' folder 
#    DIRECTLY into the current working directory (/app).
# This ensures api.py, model.py, and the .pkl file are all at the top level of the container's execution path.
COPY src/. .

# 6. Port
EXPOSE 8080

# 7. Command: Run the application using Gunicorn.
# The command is simple because 'api.py' is now guaranteed to be in the current working directory.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]
