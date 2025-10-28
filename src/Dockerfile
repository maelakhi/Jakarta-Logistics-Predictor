# 1. Base Image: Start with a lightweight, official Python base.
FROM python:3.10-slim

# 2. Working Directory: Set the primary folder inside the container.
WORKDIR /app

# 3. Dependencies: Copy the requirements list and install the libraries.
# This installs flask, scikit-learn, numpy, pandas, and gunicorn.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Application Code: Copy the rest of your files (api.py, model.py, etc.).
COPY . .

# 5. Port: Tell the container to listen on port 8080.
EXPOSE 8080

# 6. Command: Define the final command to run the application using Gunicorn.
# gunicorn is a production web server that runs the Flask app named 'app'
# which is defined inside your 'api.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]