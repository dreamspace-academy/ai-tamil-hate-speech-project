# Getting the Base Image
FROM tiangolo/uvicorn-gunicorn:python3.7

# Creating a new folder
RUN mkdir /frontend

# Copy the requirements file
COPY requirements.txt /frontend

# Change the working directory
WORKDIR /frontend

# Installing Packages
RUN pip install -r requirements.txt

# Copy everything to working directory
COPY . /frontend

# Exposing the port
EXPOSE 8501

# Running the streamlit service
CMD ["streamlit", "run", "frontend.py"]
