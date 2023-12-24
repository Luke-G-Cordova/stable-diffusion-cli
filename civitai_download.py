import subprocess
import requests
from os import path
import os
import json

def download_file_vid(version_id):
    try:
        print(f"Downloading from: https://civitai.com/api/download/models/{version_id}")
        res = requests.get(f"https://civitai.com/api/v1/model-versions/{version_id}?type=Model&format=SafeTensor")
        jres = json.loads(res.content)
        dir = path.join("./models", jres["model"]["type"] if jres["model"]["type"] != "TextualInversion" else "embeddings", path.splitext(jres["files"][0]["name"])[0])
        if not path.isdir(dir):
            os.mkdir(dir)
        destination = path.join(dir, jres["files"][0]["name"]) 
        
        subprocess.run(
            [
                "", 
                f"https://civitai.com/api/download/models/{version_id}", 
                "-O", 
                destination,
                "--content-disposition"
            ]
        )
        with open(path.join(dir, "info.json"), "w") as f:
            f.write(json.dumps(jres))

        print(f"Download successful. File saved at: {destination}")
    except Exception as e:
        print(f"Failed to download. Error: {e}")

