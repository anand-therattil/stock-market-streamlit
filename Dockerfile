# Use Python 3.12.1 as the base image
FROM python:3.12.1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .


# Install the dependencies specified in requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt



# Copy the rest of the application's code into the container
COPY . .

# Command to run the application
CMD ["streamlit", "run", "frontend.py"]