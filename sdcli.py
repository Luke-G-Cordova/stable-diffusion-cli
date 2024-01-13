import argparse
from os import path
import os
import txt2img
import img2img
from util.civitai_download import convert_file_to_diffusers_ckpt, query_for_file
import util.colors as co
import json
import re

parser = argparse.ArgumentParser()

with open("args.json", "r") as file:
    unparsed = json.loads(file.read())
for arg in list(unparsed.values()):
    kwargs = {}
    if "default" in arg and arg["default"] == "None":
        arg["default"] = None
    if "type" in arg:
        if arg["type"] == "str":
            arg["type"] = str
        elif arg["type"] == "int":
            arg["type"] = int
        elif arg["type"] == "float":
            arg["type"] = float
    for key in arg:
        if key != "args":
            kwargs[key] = arg[key] if arg[key] != "None" else None
    parser.add_argument(
        *arg["args"],
        **kwargs
    )

args = parser.parse_args()

if args.help_list_lora:
    if path.isdir("./models/lora"):
        for filename in os.listdir("./models/lora"):
            lora_name, _ = path.splitext(filename)
            print(f"{co.green}{lora_name}{co.reset}")
    else:
        print(f"{co.neutral}Could not find directory: {co.red}models/lora{co.reset}")
    exit()
if args.help_list_models:
    if path.isdir("./models"):
        for filename in os.listdir("./models"):
            if path.isdir(path.join("./models", filename)):
                config_file = path.join("./models", filename, "model_index.json")
                if path.isfile(config_file):
                    with open(config_file, "r") as f:
                        jsConf = json.loads(f.read())
                        print(f"{co.neutral}MODEL: {co.green}{filename}{co.neutral} :SCHEDULER: {co.yellow}{jsConf['scheduler'][1]}{co.reset}" )
    exit()

_, f = path.split(args.model_path)
f, _ = path.splitext(f)
model_path = args.model_path if \
    path.isdir(args.model_path) or \
    path.isfile(args.model_path) else \
    path.join(
        "models", 
        "checkpoint", 
        f, 
    )

if path.isdir(model_path):
    print(f"{co.neutral}Model path: {co.green}{model_path}{co.reset}")
elif path.isfile(model_path):
    convert_file_to_diffusers_ckpt(model_path, args.scheduler_type)
    print(f"{co.neutral}Model path: {co.green}{model_path}{co.reset}")
else:
    query = path.splitext(path.split(model_path)[1])[0]
    query = re.sub("(_|)v[0-9.]+", "", query, flags=re.I)
    query = re.sub("_" , "", query)
    query = re.split("(?<!^)[A-Z](?=[a-z])", query)
    query = " ".join(query)
    query_for_file(query, types=["Checkpoint"], sort="Highest Rated", period="AllTime", limit=1) 
    convert_file_to_diffusers_ckpt(model_path, args.scheduler_type, force=True)
    _, fn = path.split(model_path)
    fn, ext = path.splitext(fn)
    rm_path = f"models/checkpoint/{fn}/{fn}"
    if ext == "":
        if path.isfile(rm_path + ".safetensors"):
            os.remove(rm_path + ".safetensors")
        elif path.isfile(rm_path + ".ckpt"):
            os.remove(rm_path + ".ckpt")
        else:
            print(f"{co.red}Could not find path: {rm_path}.safetensors to remove{co.reset}")
    else:
        os.remove(rm_path + ext)

embeddings_path = args.embeddings_path
if path.isdir(embeddings_path):
    print(f"{co.neutral}Embedding path: {co.green}{embeddings_path}{co.reset}")
else:
    print(f"{co.yellow}Embeddings path not found{co.reset}")

if args.prompt is None: 
    if path.isfile(args.prompt_file):
        print(f"{co.neutral}Prompt path: {co.green}{args.prompt_file}{co.reset}")
        with open(args.prompt_file, "r") as f:
            args.prompt = f.read().strip()
    else:
        raise ValueError(f"{co.red}--prompt was left unset and can not find {args.prompt_file}{co.reset}")
if args.negative_prompt is None: 
    if path.isfile(args.negative_prompt_file):
        print(f"{co.neutral}Prompt path: {co.green}{args.negative_prompt_file}{co.reset}")
        with open(args.negative_prompt_file, "r") as f:
            args.negative_prompt = f.read().strip()
    else:
        raise ValueError(f"{co.red}--negative_prompt was left unset and can not find {args.negative_prompt_file}{co.reset}")


if args.task == "txt2img":
    txt2img.start(
        model_path=model_path,
        scheduler_type=args.scheduler_type,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        embeddings_path=embeddings_path,
        width=args.width,
        height=args.height,
        seed=args.seed,
        clip_skip=args.clip_skip,
        lora_path=args.lora_path,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        embed_prompts=args.embed_prompts,
        allow_tf32=args.allow_tf32,
        save_generation_data=args.no_gen_data,
        out_dir=args.out_dir,
        out_name=args.out_name,
        vae_path=args.vae_path,
    )
elif args.task == "img2img":
    img2img.start(
        model_path=model_path,
        scheduler_type=args.scheduler_type,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        embeddings_path=embeddings_path,
        width=args.width,
        height=args.height,
        seed=args.seed,
        clip_skip=args.clip_skip,
        lora_path=args.lora_path,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        embed_prompts=args.embed_prompts,
        allow_tf32=args.allow_tf32,
        save_generation_data=args.no_gen_data,
        out_dir=args.out_dir,
        out_name=args.out_name,
        image_path=args.image_path,
        image_strength=args.image_strength,
        pose_path=args.pose_path,
    )
else:
    print(f"{co.red}No current support for {args.task}{co.reset}")

