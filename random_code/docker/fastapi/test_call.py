import requests

url = "http://0.0.0.0:8000/ask"
files = {"image": open("/Users/ksulahian/Downloads/cats.jpg", "rb")}
params = {"text": "what type of animal and how many are there?"}

response = requests.post(url, files=files, params=params)
print(response.json())