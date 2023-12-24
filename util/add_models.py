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


def add_text_inversion_embeddings(
    embeddings_path,
    pipe,
):
    embeddings_data = []
    if path.isdir(embeddings_path):
        for dirname in os.listdir(embeddings_path):
            if path.isdir(path.join(embeddings_path, dirname)):
                filename = path.join(embeddings_path, dirname, dirname)
                if path.isfile(filename + ".safetensors"):
                    filename = filename + ".safetensors"
                    embed_file = load_file(filename, device="cpu")
                    pipe.load_textual_inversion(embed_file["emb_params"], token=dirname, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
                    print(f"{co.neutral}Embedding: {co.green}{filename}{co.green}{co.reset}")
                    if path.isfile(path.join(embeddings_path, dirname, "info.json")):
                        embeddings_data.append(path.join(embeddings_path, dirname, "info.json"))
                elif path.isfile(filename + ".pt"):
                    filename = filename + ".pt"
                    embed_file = torch.load(filename, map_location="cpu")
                    pipe.load_textual_inversion(embed_file["string_to_param"]["*"], token=dirname, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
                    print(f"{co.neutral}Embedding: {co.green}{filename}{co.green}{co.reset}")
                    if path.isfile(path.join(embeddings_path, dirname, "info.json")):
                        embeddings_data.append(path.join(embeddings_path, dirname, "info.json"))
                else:
                    print(f"{co.neutral}Unrecognized file format: {co.red}{filename}{co.neutral} :Embeddings must be of type .safetensors or .pt")
            elif path.isfile(path.join(embeddings_path, dirname)):
                fn, ext = path.splitext(dirname)
                filename = dirname
                if ext == ".safetensors":
                    embed_file = load_file(path.join(embeddings_path, filename), device="cpu")
                    pipe.load_textual_inversion(embed_file["emb_params"], token=fn, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
                    print(f"{co.neutral}Embedding: {co.green}{path.join(embeddings_path, filename)}{co.green}{co.reset}")
                else:
                    embed_file = torch.load(path.join(embeddings_path, filename), map_location="cpu")
                    pipe.load_textual_inversion(embed_file["string_to_param"]["*"], token=fn, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
                    print(f"{co.neutral}Embedding: {co.green}{path.join(embeddings_path, filename)}{co.green}{co.reset}")
    else:
        print(f"{co.neutral}Cannot find embedding path: {co.red}{embeddings_path}{co.neutral} :Will not use embeddings{co.reset} ")
    return embeddings_data