import subprocess
import requests
from os import path
import os
import json
import urllib.parse as url
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
import colors as co

def convert_file_to_diffusers_ckpt(
    model_path,
    scheduler_type="dpm",
    force=False,
):
    fname, ext = path.splitext(path.split(model_path)[1])
    new_path = path.join(path.abspath("./models/"), "checkpoint", fname)
    if path.isdir(new_path) and not force:
        print(f"{co.green}{new_path} {co.neutral}has already been created.\nUsing {co.green}{new_path}{co.neutral} as model.{co.reset}")
        model_path = new_path
    else:
        print(f"{co.neutral}Attempting to convert {co.green}{model_path}{co.neutral} to huggingface model{co.reset}")
        from_safetensors = ext == ".safetensors"

        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=model_path,
            original_config_file=None,
            image_size=None,
            prediction_type=None,
            model_type=None,
            extract_ema=False,
            scheduler_type=scheduler_type,
            num_in_channels=None,
            upcast_attention=False,
            from_safetensors=from_safetensors,
            device="cuda",
            stable_unclip=None,
            stable_unclip_prior=None,
            clip_stats_path="",
            controlnet=False,
            vae_path=None,
            pipeline_class=None,
        )
        # if args.half:
        #     pipe.to(torch_dtype=torch.float16)

        # if args.controlnet:
        #     # only save the controlnet model
        #     pipe.controlnet.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
        # else:
        model_path = new_path
        pipe.save_pretrained(model_path, safe_serialization=True)

def query_for_file(query, **kwargs):
    try:
        models_url = f"https://civitai.com/api/v1/models?query={url.quote( query )}"
        params = ""
        if len(kwargs) > 0:
            for key, val in kwargs.items():
                if isinstance(val, str):
                    params = f"{params}&{url.quote(key.strip())}={url.quote(val.strip())}"
                elif isinstance(val, list):
                    for i in range(len(val)):
                        val[i] = f"types[]={val[i]}"
                    params = f"{params}&{'&'.join(val)}"
                elif isinstance(val, int):
                    params = f"{params}&{url.quote(key.strip())}={val}"
                else:
                    print(f"{co.red}Unexpected type. Not querying with {key}={val} of type {type(val)}{co.reset}")
            models_url = models_url + params

        print(f"{co.neutral}Finding matching results from: {co.blue}{models_url}{co.reset}")
        res = requests.get(f"{models_url}")
        jres = json.loads(res.content)
        if len(jres["items"]) > 0:
            print(f"{co.neutral}Downloading: {co.blue}{jres['items'][0]['name']}{co.reset}")
            download_file_vid(jres['items'][0]['modelVersions'][0]['id'])
        else:
            print(f"{co.red}Could not find model{co.reset}")
    except Exception as e:
        print(f"{co.red}{e}{co.reset}")

def download_file_vid(version_id):
    try:
        print(f"{co.neutral}Downloading from: {co.green}https://civitai.com/api/v1/model-versions/{version_id}?type=Model&format=SafeTensor{co.reset}")
        res = requests.get(f"https://civitai.com/api/v1/model-versions/{version_id}?type=Model&format=SafeTensor")
        jres = json.loads(res.content)

        dir = path.join(
            "./models", 
            jres["model"]["type"] if \
                jres["model"]["type"] != "TextualInversion" else \
                "embeddings", 
            path.splitext(jres["files"][0]["name"])[0]
        )

        if not path.isdir(dir):
            os.mkdir(dir)
        destination = path.join(dir, jres["files"][0]["name"]) 

        # use wget to get around user agent stuff
        subprocess.run(
            [
                "C:/Program Files (x86)/GnuWin32/bin/wget.exe", 
                f"https://civitai.com/api/download/models/{version_id}", 
                "-O", 
                destination,
                "--content-disposition"
            ]
        )
        # don't save info on checkpoints
        if jres["model"]["type"] != "Checkpoint":
            with open(path.join(dir, "info.json"), "w") as f:
                f.write(json.dumps(jres))

        print(f"{co.neutral}Download successful. File saved at: {co.green}{destination}{co.reset}")
    except Exception as e:
        print(f"{co.red}Failed to download. Error: \n{e}{co.reset}")