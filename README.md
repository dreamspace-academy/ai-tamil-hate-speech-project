<h1 align="center">AI Tamil Hate Speech Detector</h1>


<p align="center">
<img alt="OMDENA NYU" src="misc/Omdena-NYU.png">
</p>


## Contents

1. [Project Introduction](#section01)
    - [Solution](#section01a)
2. [Project Setup and Documentation](#section02)

3. [Project Details](#section03)
    - [Screenshot](#section03a)
    - [Directory Details](#section03b)
4. [License](#section04)


<a id='section01'></a>

## Introduction

This project is the result of a collaboration between DreamSpace Academy, NYU CIC, and Omdena, and was funded by NYU CIC. The goal of the project is to detect hate speech on social media platforms that's in either Tamil, English or Tanglish (English transliterated into Tamil). A global team of 50 AI changemakers took on the task to detect hate speech in Tamil language.The partner for this challenge is social enterprise DreamSpace Academy (DSA). The Challenge is supported by the NYU Center on International Cooperation and the Netherlands Ministry of Foreign Affairs.


The focus is on the following hate-speech related categories:

* Community-based hate speech

* Religion-based hate speech

* Gender-based hate speech

* Political hate speech


<a id='section01a'></a>

### Solution

* **An AI model written in Python**: Built using `Fastapi` and `Streamlit` making the complete code base in Python.


<a id='section02'></a>

## Project Setup and Documentation

1.  **Clone the Repo.**


2. **Run the backend service.** (Make sure Docker is running.)
    - Go to the `backend` folder
    - Run the `Docker Compose` command

    ```console  
    $ cd backend
    backend:~$ sudo docker-compose up -d
    ```

3. **Run the frontend service.**
   
    - Go to the `frontend` folder
    <!---
    - Create the docker image from the `Docker File`
    - Then execute the docker image to spin up a container.
    ```console  
    $ cd frontend
    frontend:~$ sudo docker build -t streamlit_app .
    frontend:~$ sudo docker run -d --name streamlit_app streamlit_app
    ```
    --->
    - Run the app with the streamlit run command
    ```console  
    $ cd frontend
    frontend:~$ streamlit run NLPfile.py
    ```

4. **Access to Fastapi Documentation**: 
    - Hate Classification: http://localhost:8080/api/v1/classification/docs


<a id='section03'></a>

## Project Details

<a id='section03a'></a>

### Screenshot

<p align="center">
<img alt="">
</p>

<a id='section03b'></a>

### Directory Details

* **Front End**: streamlit code is in the `frontend` folder. Along with the `Dockerfile` and `requirements.txt`

* **Back End**: Fastapi code is in the `backend` folder.
    * The project has been implemented as a microservice, with its own fastapi server and requirements and Dockerfile.
    * Directory tree as below:
    ```
    - classification
        > app
            > api
                > bert_model_artifacts
                    - model.bin
                    - network.py
    ```
    * Each folder model will need the following files:
        * Model bin file is the saved model after training.
        * `network.py` for customised model, define class here.

    * `config.json`: This file contains the details of the models in the backend and the dataset they are trained on.

<a id='section04'></a>

## License

This project is licensed under the Apache License 2.0. You may not use any trademarks associated with the software without permission. The full text of the license can be found in the [LICENSE](LICENSE) file.
