# Use the official Python 3.10.14 image from the Docker Hub
FROM python:3.10.14

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app_folder/ /app/

# Install the required Python packages
RUN pip install  -r requirements.txt

# Expose port 8051 for the Streamlit app
EXPOSE 8501

# Define the command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]
