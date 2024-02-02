from diffusers import(
  AutoencoderKL,
  AutoPipelineForImage2Image,
  ControlNetModel,
  StableDiffusionControlNetPipeline
)
from diffusers.utils import load_image
import os
from os import path
import numpy as np
import torch
import random
import util.colors as co
import json
from PIL import Image
import cv2
from util.add_models import ( 
    add_lora, 
    add_text_inversion_embeddings,
    get_available_poses,
    get_trained_textual_inversions 
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
    out_dir="output",
    out_name="",
    image_path=None,
    image_strength=.8,
    vae_path=None,
    pose_path="",
    save_init_image=True,
):
    # default cuda
    if torch.cuda.is_available():
        device_name = torch.device("cuda")
    else:
        device_name = torch.device("cpu")
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
     
    print(f"{co.neutral}Using device: {co.blue}{device_name}{co.neutral} : for inference{co.reset}")

    trained_textual_inversions = get_trained_textual_inversions(embeddings_path)

    available_poses = get_available_poses(pose_path)

    # parse the prompt and make embeddings if needed
    prompt, pLora, pWeight, pTextInvs, pose_names = parse_prompt(prompt, trained_textual_inversions)
    negative_prompt, nLora, nWeight, nTextInvs, _ = parse_prompt(negative_prompt, trained_textual_inversions)

    print(f"{co.neutral}PROMPT: {co.reset}{prompt}")
    print(f"{co.neutral}NEGATIVE_PROMPT: {co.reset}{negative_prompt}")

    canny_image = None
    save_image = None
    controlnet = None
    use_pose = None
    if len(pose_names) > 0:
        if pose_names[0] == "random":
            use_pose = random.sample(list(available_poses.values()), 1)[0]
        else:
            use_pose = available_poses[pose_names[0]]
        pose = load_image(use_pose)
        save_image = pose
        
        pose = np.array(pose)
        pose = cv2.Canny(pose, 100, 200)
        pose = pose[:, :, None]
        pose = np.concatenate([pose, pose, pose], axis=2)
        canny_image = Image.fromarray(pose)
        width = canny_image.width
        height = canny_image.height
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
        pipeline = StableDiffusionControlNetPipeline
    else:
        pipeline = AutoPipelineForImage2Image

    # set scheduler
    scheduler = get_scheduler_import(scheduler_type).from_pretrained(
        model_path,
        subfolder="scheduler",
        use_karras_sigmas=True,
    )
    
    if vae_path is not None:
        vae = AutoencoderKL.from_single_file(
            vae_path, use_safetensors=True
        ) 
        pipe = pipeline.from_pretrained(
            model_path,
            vae=vae,
            scheduler=scheduler,
            safety_checker=None,
            use_safetensors=True,
            controlnet=controlnet,
        )
    else:
        pipe = pipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            safety_checker=None,
            use_safetensors=True,
            controlnet=controlnet,
        )

    if use_pose is not None:
        print(f"{co.neutral}Using Pose: {co.green}{use_pose}{co.reset}")
    print(f"{co.neutral}Using width: {co.green}{width}{co.neutral}, and height: {co.green}{height}{co.reset}")
    # pipe.enable_model_cpu_offload()
    # load embeddings
    embeddings_data = add_text_inversion_embeddings(pTextInvs+nTextInvs, pipe)

    init_image = canny_image if canny_image is not None else load_image(image_path)

    pipe.to(device_name)

    if embed_prompts:
        prompt_embeds, negative_prompt_embeds = get_prompt_embeddings(
            pipe, 
            prompt,
            negative_prompt,
            split_character = ",",
            device = device_name
        )

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
                height=height,
                image=init_image,
                strength=image_strength,
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
                height=height,
                image=init_image,
                strength=image_strength,
            ).images[0]

        if clip_skip > 1:
            pipe.text_encoder.text_model.encoder.layers = clip_layers

        if not path.isdir(out_dir):
            print(f"{co.neutral}Making out_dir: {co.green}{out_dir}{co.reset}")
            os.mkdir(out_dir)
        print(f"{co.neutral}saving to {co.green}{out_dir}/{seeds[i]}/{out_name}{seeds[i]}.png{co.reset}")
        if not path.isdir(f"{out_dir}/{seeds[i]}"):
            os.mkdir(f"{out_dir}/{seeds[i]}")
        image.save(f"{out_dir}/{seeds[i]}/{out_name}{seeds[i]}.png")
        if save_init_image:
            init_image_name = path.split(use_pose)
            init_image_name = f"{path.split(init_image_name[0])[1]}--{init_image_name[1]}"
            if save_image is not None:
                save_image.save(f"{out_dir}/{seeds[i]}/{init_image_name}")
            else:
                init_image.save(f"{out_dir}/{seeds[i]}/{init_image_name}")
        
        if save_generation_data:
            with open(f"{out_dir}/{seeds[i]}/{out_name}generation_data.json", 'w') as f:
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
                    guidance_scale,
                    use_pose,
                )))