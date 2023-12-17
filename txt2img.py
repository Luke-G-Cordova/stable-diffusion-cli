from diffusers import(
  DiffusionPipeline
)
import os
from os import path
from safetensors.torch import load_file
import torch
import random
import re
import tqdm
import colors as co



# ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']
def get_scheduler_import(scheduler_type):
    if scheduler_type == "pndm":
        from diffusers import PNDMScheduler as sc
    elif scheduler_type == "lms":
        from diffusers import LMSDiscreteScheduler as sc
    elif scheduler_type == "heun":
        from diffusers import HeunDiscreteScheduler as sc
    elif scheduler_type == "euler":
        from diffusers import EulerDiscreteScheduler as sc
    elif scheduler_type == "euler-ancestral":
        from diffusers import EulerAncestralDiscreteScheduler as sc
    elif scheduler_type == "dpm":
        from diffusers import DPMSolverMultistepScheduler as sc
    elif scheduler_type == "ddim":
        from diffusers import DDIMScheduler as sc
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
    return sc

def parse_prompt(prompt):
    pieces = prompt.split(",")

    # detect lora
    name = []
    weight = []
    i = 0
    while i < len(pieces):
        s = re.findall("\<lora:.+:[0-9.]+\>", pieces[i])
        if len(s) > 0:
            for mat in s:
                # get lora name
                mat = mat[1:]
                mat = mat[:-1]
                lora_info = mat.split(":")
                name.append(lora_info[1])
                weight.append(float(lora_info[2]))
            pieces.remove(pieces[i])
        else:
            i+=1
    return ",".join(pieces), name, weight

def add_lora(
    pipe,
    lora_names,
    lora_weights,
    lora_path
):
    if path.isdir(lora_path):
        for i, name in enumerate(lora_names):
            if path.isfile(path.join(lora_path, name + ".safetensors")):
                pipe.load_lora_weights(
                    path.join(lora_path, name + ".safetensors"),
                    weight_name=name+".safetensors",
                    adapter_name=name,
                )
                print(f"{co.neutral}LoRA file: {co.green}{path.join(lora_path, name + '.safetensors')}{co.green}:{co.yellow}{lora_weights[i]}{co.reset}")
            elif path.isfile(path.join(lora_path, name + ".ckpt")):
                pipe.load_lora_weights(
                    path.join(lora_path, name + ".ckpt"),
                    weight_name=name+".ckpt",
                    adapter_name=name,
                )
                print(f"{co.neutral}LoRA file: {co.green}{path.join(lora_path, name + '.ckpt')}{co.green}:{co.yellow}{lora_weights[i]}{co.reset}")
            else:
                print(f"{co.neutral}LoRA file: {co.red}{path.join(lora_path, name +'.safetensors')}{co.yellow}:{co.neutral}Will not use this lora{co.reset}")
                del lora_names[i]
                del lora_weights[i]
    else:
        raise ValueError(f"{co.red}{lora_path} is not a path to a lora directory{co.reset}")

    pipe.set_adapters(lora_names, lora_weights)

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

def start(
    model_path, 
    scheduler_type="pndm",
    embeddings_path="",
    prompt="",
    negative_prompt="",
    seed=-1,
    clip_skip=1,
    lora_path="",
    num_inference_steps=20,
    guidance_scale=7,
    width=512,
    height=512,
    batch_size=1,
    embed_prompts=False,
    allow_tf32=False,
):
    if torch.cuda.is_available():
        device_name = torch.device("cuda")
    else:
        device_name = torch.device("cpu")
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        
    print(f"{co.neutral}Using device: {co.blue}{device_name}{co.neutral} : for inference{co.reset}")

    scheduler = get_scheduler_import(scheduler_type).from_pretrained(
        model_path,
        subfolder="scheduler"
    )
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        safety_checker=None,
        use_safetensors=True
    )
    if path.isdir(embeddings_path):
        for filename in os.listdir(embeddings_path):
            fn, ext = path.splitext(filename)
            if ext == ".safetensors":
                embed_file = load_file(path.join(embeddings_path, filename), device="cpu")
                pipe.load_textual_inversion(embed_file["emb_params"], token=fn, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            else:
                embed_file = torch.load(path.join(embeddings_path, filename), map_location="cpu")
                pipe.load_textual_inversion(embed_file["string_to_param"]["*"], token=fn, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            print(f"{co.neutral}Embedding: {co.green}{path.join(embeddings_path, filename)}{co.green}{co.reset}")
    pipe.to(device_name)


    generators = []
    seeds = []
    if seed == -1:
        for _ in range(batch_size):
            seeds.append(random.randrange(0, 1000000))
            generators.append(torch.Generator(device="cuda").manual_seed(seeds[-1]))
    else:
        seeds.append(seed)
        generators.append(torch.Generator(device="cuda").manual_seed(seeds[0]))
    print(f"{co.neutral}SEEDS USED: {co.yellow}{seeds}{co.reset}")

    # generator = torch.Generator(device="cuda").manual_seed(seed)
    
    prompt, pLora, pWeight = parse_prompt(prompt)
    negative_prompt, nLora, nWeight = parse_prompt(negative_prompt)
    if embed_prompts:
        prompt_embeds, negative_prompt_embeds = get_prompt_embeddings(
            pipe, 
            prompt,
            negative_prompt,
            split_character = ",",
            device = device_name
        )

    print(f"{co.neutral}PROMPT: {co.reset}{prompt}")
    print(f"{co.neutral}NEGATIVE_PROMPT: {co.reset}{negative_prompt}")

    add_lora(pipe=pipe, lora_names=pLora+nLora, lora_weights=pWeight+nWeight, lora_path=lora_path)

    clip_layers = pipe.text_encoder.text_model.encoder.layers

    pipe.enable_xformers_memory_efficient_attention()
    for i in range(batch_size):
        if clip_skip > 1:
            pipe.text_encoder.text_model.encoder.layers = clip_layers[:-(clip_skip)]
        print(f"{co.neutral}INFERENCING WITH SEED: {co.yellow}{seeds[i]}{co.reset}")
        if embed_prompts:
            image = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generators[i],
                num_images_per_prompt=1,
                width=width,
                height=height
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generators[i],
                num_images_per_prompt=1,
                width=width,
                height=height
            ).images[0]

        if clip_skip > 1:
            pipe.text_encoder.text_model.encoder.layers = clip_layers

        print(f"{co.neutral}saving to {co.green}output/{seeds[i]}.png{co.reset}")
        image.save(f"output/{seeds[i]}.png")

        




    

