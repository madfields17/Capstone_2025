import requests

url = "http://localhost:5050/predict"

# Test Bona Fide
print("Bona Fide Results:")
audio_path = "bona_fide_example.wav"
with open(audio_path, 'rb') as f:
    files = {'audio': f}
    response = requests.post(url, files=files)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

# Test Spoofed
print("Spoofed Results:")
audio_path = "spoofed_example.wav"
with open(audio_path, 'rb') as f:
    files = {'audio': f}
    response = requests.post(url, files=files)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
