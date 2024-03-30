import torch
import re
from os import path
import json

def get_prompt_embeddings(pipeline, prompt, negative_prompt, split_character, device):
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    count_prompt = len(prompt.split(split_character))
    count_negative_prompt = len(negative_prompt.split(split_character))

    # create the tensor based on which prompt is longer
    # if count_prompt >= count_negative_prompt:
    input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
    shape_max_length = input_ids.shape[-1]
    negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                        max_length=shape_max_length, return_tensors="pt").input_ids.to(device)
    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])
    
    prompt_embeddings = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeddings = torch.cat(neg_embeds, dim=1)

    if prompt_embeddings.size() != negative_prompt_embeddings.size():
        negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
                                       max_length=shape_max_length).input_ids.to(device)
        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, max_length):
            concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])
        
        prompt_embeddings = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeddings = torch.cat(neg_embeds, dim=1)

    # print(prompt_embeddings.size)
    # print(prompt_embeddings.shape)
    # print(negative_prompt_embeddings.size)
    # print(negative_prompt_embeddings.shape)

    return prompt_embeddings, negative_prompt_embeddings

def parse_prompt(prompt, trained_textual_inversions, lora_path):
    prompt = prompt.strip()
    lines = prompt.split("\n")
    pieces = []
    for line in lines:
        line = line.split("//")[0]
        ls = line.split(",")
        for l in ls:
            if l.strip() != "":
                pieces.append(l.strip())

    used_textual_inversions = []
    # detect lora and textual inversions
    name = []
    weight = []
    pose_names = []
    i = 0
    while i < len(pieces):

        found_poses = re.findall("\<pose:.+\>", pieces[i])
        if len(found_poses) > 0:
            for mat in found_poses:
                mat = mat[1:]
                mat = mat[:-1]
                pose_info = mat.split(":")
                pose_names.append(pose_info[1])
            pieces.remove(pieces[i])
            continue

        found_lora = re.findall("\<lora:.+:[0-9.]+\:?\w*?\>", pieces[i])
        if len(found_lora) > 0:
            for mat in found_lora:
                # get lora name
                mat = mat[1:]
                mat = mat[:-1]
                lora_info = mat.split(":")
                name.append(lora_info[1])
                weight.append(float(lora_info[2]))
                if len(lora_info) == 4:
                    with open(path.join(lora_path, lora_info[1], "info.json"), 'r') as f:
                        jf = json.loads(f.read())
                        for p in jf["trainedWords"]:
                            if p.strip() != "":
                                pieces.append(p.strip())
            pieces.remove(pieces[i])
        else:
            pieces[i] = pieces[i].strip()
            if pieces[i] in trained_textual_inversions:
                used_textual_inversions.append(trained_textual_inversions[pieces[i]])
            i+=1
    return ", ".join(pieces), name, weight, used_textual_inversions, pose_names