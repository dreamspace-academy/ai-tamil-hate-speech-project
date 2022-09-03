import requests
import json

data = {'text':'hello everyone'}
endpoint = 'http://localhost:8080/api/v1/classification/predict'

result = requests.post(endpoint, json=data)
print(f"Request status: {result.status_code}")
print(f"Prediction: {result.content}")