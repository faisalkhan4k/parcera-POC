FROM python:3.11-slim

# Install system dependencies required for PyAudio/SoundDevice
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Run the app
CMD ["python", "main.py"]