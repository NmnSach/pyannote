import requests
from dotenv import load_dotenv
import os

API_URL = "https://api-inference.huggingface.co/models/arpanghoshal/EmoRoBERTa"
#headers = {"Authorization": f"Bearer {API} "}

# Load the .env file
load_dotenv()

# Get the API key from the environment variables
API = os.getenv("API")
headers = {"Authorization": f"Bearer {API}"}



def query_sentiment(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
# output=query_sentiment({
#     "inputs": "Damn it's not working!!!"
# })
    

# print(output[0][0])

# APT_TOKEN="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"
# API_URL = "https://api-inference.huggingface.co/models/Baghdad99/saad-speech-recognition-english-audio-to-text" # Updated API URL
# headers = {"Authorization": f"Bearer {API}"}

# def query_att(filename):
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers=headers, data=data)
#     return response.json()
# output1 = query_att("1.mp3")
# text=output1
# print(text)
# output = query_sentiment()
# print(output)


API_URL_SOUND = "https://api-inference.huggingface.co/models/Baghdad99/saad-speech-recognition-hausa-audio-to-text"
headers = {"Authorization": f"Bearer {API}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL_SOUND, headers=headers, data=data)
    return response.json()

output = query("1.mp3")
text = str(output['text'])
print(text)

senti = query_sentiment({
    "inputs": text
})

print(senti[0][0])

# output=query_sentiment({
#     "inputs": text
# })
# print(output)
#File upload
#transcript
#transcript ka sentiment analysis


#transcript ka print basically pyannote jaisa ho --- dialogue by dialogue  -- taaki user output zyada readable ho

from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token={API})

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("1.mp3")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...