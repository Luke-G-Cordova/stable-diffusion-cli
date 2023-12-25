from diffusers import(
  DiffusionPipeline
)
import os
from os import path
import torch
import random
import util.colors as co
import json
from util.add_models import ( 
    add_lora, 
    add_text_inversion_embeddings 
)
from util.data import (
    generation_data,
    get_scheduler_import
)
from util.prompt import(
    get_prompt_embeddings,
    parse_prompt
)

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
    save_generation_data=True,
    out_dir="output"
):
    # default cuda
    if torch.cuda.is_available():
        device_name = torch.device("cuda")
    else:
        device_name = torch.device("cpu")
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
     
    print(f"{co.neutral}Using device: {co.blue}{device_name}{co.neutral} : for inference{co.reset}")

    # set scheduler
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

    # load embeddings
    embeddings_data = add_text_inversion_embeddings(embeddings_path, pipe)

    pipe.to(device_name)

    # set seeds
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

    # parse the prompt and make embeddings if needed
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

    # load lora
    lora_data = add_lora(pipe=pipe, lora_names=pLora+nLora, lora_weights=pWeight+nWeight, lora_path=lora_path)

    clip_layers = pipe.text_encoder.text_model.encoder.layers

    # not sure if my machine benefits from this optimization
    pipe.enable_xformers_memory_efficient_attention()

    # main inference loop
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

        if not path.isdir(out_dir):
            print(f"{co.neutral}Making out_dir: {co.green}{out_dir}{co.reset}")
            os.mkdir(out_dir)
        print(f"{co.neutral}saving to {co.green}{out_dir}/{seeds[i]}/{seeds[i]}.png{co.reset}")
        if not path.isdir(f"{out_dir}/{seeds[i]}"):
            os.mkdir(f"{out_dir}/{seeds[i]}")
        image.save(f"{out_dir}/{seeds[i]}/{seeds[i]}.png")
        
        if save_generation_data:
            with open(f"output/{seeds[i]}/generation_data.json", 'w') as f:
                f.write(json.dumps(generation_data(
                    prompt, 
                    negative_prompt,
                    lora_data,
                    embeddings_data,
                    model_path,
                    scheduler_type,
                    seeds[i],
                    clip_skip,
                    num_inference_steps,
                    width,
                    height,
                    guidance_scale
                )))