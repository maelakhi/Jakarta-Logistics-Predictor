# 1. Base Image: Use the full Python 3.10 image for best compatibility
FROM python:3.10

# 2. Add System Dependencies: Ensure essential tools are present
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. CRITICAL: Copy requirements.txt from the local root into a temporary directory
# We need this file for the installation step.
WORKDIR /tmp
COPY requirements.txt .

# 4. Install Dependencies: Run pip install from the temporary directory
RUN pip install --no-cache-dir -r requirements.txt

# 5. Set Final Working Directory: The folder where the application will execute from.
WORKDIR /app

# 6. Application Code: Copy the CONTENTS of your local 'src' folder 
#    DIRECTLY into the final working directory (/app).
# This ensures api.py, model.py, and the .pkl file are all at the top level of /app.
COPY src/. .

# 7. Port
EXPOSE 8080

# 8. Command: Run the application using Gunicorn.
# The command is simple because 'api.py' is now guaranteed to be the 'api' module in the current directory.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]
