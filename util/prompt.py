import torch
import re
from os import path
import json

def get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    split_character = ",",
    device = torch.device("cpu")
):
    max_length = pipe.tokenizer.model_max_length
    # Simple method of checking if the prompt is longer than the negative
    # prompt - split the input strings using `split_character`.
    count_prompt = len(prompt.split(split_character))
    count_negative_prompt = len(negative_prompt.split(split_character))

    # If prompt is longer than negative prompt.
    if count_prompt >= count_negative_prompt:
        input_ids = pipe.tokenizer(
            prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipe.tokenizer(
            negative_prompt,
            truncation = False,
            padding = "max_length",
            max_length = shape_max_length,
            return_tensors = "pt"
        ).input_ids.to(device)

    # If negative prompt is longer than prompt.
    else:
        negative_ids = pipe.tokenizer(
            negative_prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipe.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation = False,
            padding = "max_length",
            max_length = shape_max_length
        ).input_ids.to(device)

    # Concatenate the individual prompt embeddings.
    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(
            pipe.text_encoder(input_ids[:, i: i + max_length])[0]
        )
        neg_embeds.append(
            pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
        )

    return torch.cat(concat_embeds, dim = 1), torch.cat(neg_embeds, dim = 1)

def parse_prompt(prompt):
    prompt = prompt.strip()
    lines = prompt.split("\n")
    pieces = []
    for line in lines:
        line = line.split("//")[0]
        ls = line.split(",")
        for l in ls:
            if l.strip() != "":
                pieces.append(l.strip())

    # detect lora
    name = []
    weight = []
    i = 0
    while i < len(pieces):
        s = re.findall("\<lora:.+:[0-9.]+\:?\w*?\>", pieces[i])
        if len(s) > 0:
            for mat in s:
                # get lora name
                mat = mat[1:]
                mat = mat[:-1]
                lora_info = mat.split(":")
                name.append(lora_info[1])
                weight.append(float(lora_info[2]))
                if len(lora_info) == 4:
                    with open(path.join("models", "lora", lora_info[1], "info.json"), 'r') as f:
                        jf = json.loads(f.read())
                        for p in jf["trainedWords"]:
                            if p.strip() != "":
                                pieces.append(p.strip())
            pieces.remove(pieces[i])
        else:
            pieces[i] = pieces[i].strip()
            i+=1
    return ",".join(pieces), name, weight