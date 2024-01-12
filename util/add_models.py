import json
from os import path
import os
import util.colors as co
from safetensors.torch import load_file
import torch

def add_lora(
    pipe,
    lora_names,
    lora_weights,
    lora_path
):
    lora_gen_data_files = []
    if path.isdir(lora_path):
        for i, name in enumerate(lora_names):
            lora = path.join(lora_path, name, name)
            if path.isfile(lora + ".safetensors"):
                lora = lora + ".safetensors"
            elif path.isfile(lora + ".ckpt"):
                lora = lora + ".ckpt"
            else:
                print(f"{co.neutral}LoRA file: {co.red}{path.join(lora, '.safetensors')}{co.yellow}:{co.neutral}Will not use this lora{co.reset}")
                del lora_names[i]
                del lora_weights[i]
                continue

            if path.isfile(path.join(lora_path, name, "info.json")):
                lora_gen_data_files.append(tuple([  path.join(lora_path, name, "info.json"), lora_weights[i]  ]))
                
            _, ext = path.splitext(lora)
            pipe.load_lora_weights(
                lora,
                weight_name=name+ext,
                adapter_name=name,
            )
            print(f"{co.neutral}LoRA file: {co.green}{lora}{co.green}:{co.yellow}{lora_weights[i]}{co.reset}")
    else:
        raise ValueError(f"{co.red}{lora_path} is not a path to a lora directory{co.reset}")

    pipe.set_adapters(lora_names, lora_weights)
    return lora_gen_data_files

def get_available_poses(pose_path):
    possible_poses = {}
    if path.isdir(pose_path):
        for dirname in os.listdir(pose_path):
            if path.isdir(path.join(pose_path, dirname)):
                for file in os.listdir(path.join(pose_path, dirname)):
                    if path.isfile(path.join(pose_path, dirname, file)) and path.splitext(file)[1] == '.png':
                        possible_poses[path.splitext(file)[0]] = path.join(pose_path, dirname, file)
            elif path.isfile(path.join(pose_path, dirname)) and path.splitext(dirname)[1] == '.png':
                possible_poses[path.splitext(dirname)[0]] = path.join(pose_path, dirname)
    else:
        print(f"{co.neutral}Cannot find pose path: {co.red}{pose_path}{co.neutral} :Will not use embeddings{co.reset} ")
    return possible_poses

def  get_trained_textual_inversions(embeddings_path):
    possible_embeddings = {}
    if path.isdir(embeddings_path):
        for dirname in os.listdir(embeddings_path):
            if path.isdir(path.join(embeddings_path, dirname)):
                filename = path.join(embeddings_path, dirname, "info.json")
                if path.isfile(filename):
                    with open(filename, 'r') as f:
                        jf = json.loads(f.read())
                        if path.isfile(path.join(embeddings_path, dirname, jf["files"][0]["name"])):
                            for word in jf["trainedWords"]:
                                possible_embeddings[word] = path.join(embeddings_path, dirname, jf["files"][0]["name"])
                            possible_embeddings[dirname] = path.join(embeddings_path, dirname, jf["files"][0]["name"])
                        else:
                            print(f"{co.red}{path.join(embeddings_path, dirname, jf['files'][0]['name'])} does not exist{co.reset}")
                else:
                    print(f"{co.red}{filename} does not exist{co.reset}")
    else:
        print(f"{co.neutral}Cannot find embedding path: {co.red}{embeddings_path}{co.neutral} :Will not use embeddings{co.reset} ")
    return possible_embeddings

def add_text_inversion_embeddings(
    text_inversion_files,
    pipe,
):
    embeddings_data = []
    for file in text_inversion_files:
        if path.isfile(file):
            dir, _ = path.split(file)
            _, dirname = path.split(dir)
            _, ext = path.splitext(file)
            if ext == ".safetensors":
                embed_file = load_file(file, device="cpu")
                pipe.load_textual_inversion(embed_file["emb_params"], token=dirname, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            elif ext == ".pt":
                embed_file = torch.load(file, map_location="cpu")
                pipe.load_textual_inversion(embed_file["string_to_param"]["*"], token=dirname, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            else:
                print(f"{co.neutral}Unrecognized file format: {co.red}{file}{co.neutral} :Embeddings must be of type .safetensors or .pt")
                continue
            print(f"{co.neutral}Embedding: {co.green}{file}{co.green}{co.reset}")
            if path.isfile(path.join(dir, "info.json")):
                embeddings_data.append(path.join(dir, "info.json"))
    return embeddings_data