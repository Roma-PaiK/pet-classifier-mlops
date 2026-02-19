# 1. Use an official, lightweight Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
# We do this before copying the rest of the code to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code and the trained model
COPY app.py .
COPY models/ /app/models/

# 5. Expose the port FastAPI will run on
EXPOSE 8000

# 6. Command to run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]