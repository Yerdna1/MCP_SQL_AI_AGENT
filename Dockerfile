# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache to reduce image size
# --upgrade: Ensures pip is up-to-date
# -r requirements.txt: Installs packages from the requirements file
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# We'll copy the 'app' directory specifically later, or mount it in docker-compose
COPY ./app ./app/
COPY .env .

# Make port 7860 available to the world outside this container (Gradio default)
EXPOSE 7860

# Define environment variables (can be overridden by docker-compose)
# Example: ENV NAME World

# Run app/main.py when the container launches
# We'll define the actual entry point later, assuming it will be app/main.py
CMD ["python", "app/main.py"]
