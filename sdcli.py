import argparse
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from os import path
import os
import txt2img
import colors as co
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    default="./models/stable-diffusion-v2",
    type=str,
    help="Path to the base model to run stable diffusion on. If evaluates to .ckpt or .safetensor file will convert to and use a huggingface diffusers model stored in ./models"
)
parser.add_argument(
    "--scheduler_type",
    default="pndm",
    type=str,
    help="scheduler type. One of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']"
)
parser.add_argument(
    "--embeddings_path",
    default="models/embeddings",
    type=str,
    help="Path to directory containing embedding files."
)
parser.add_argument(
    "--prompt",
    default=None,
    type=str,
    help="text prompt"
)
parser.add_argument(
    "--negative_prompt",
    default=None,
    type=str,
    help="negative text prompt"
)
parser.add_argument(
    "--prompt_file",
    default="prompts/p.txt",
    type=str,
    help="A file storing the prompt. --prompt flag must not be set to use this"
)
parser.add_argument(
    "--negative_prompt_file",
    default="prompts/n.txt",
    type=str,
    help="A file storing the negative prompt. --negative_prompt flag must not be set to use this"
)
parser.add_argument(
    "--seed",
    default=-1,
    type=int,
    help="Seed. If seed is -1 a random seed will be generated and used"
)
parser.add_argument(
    "--clip_skip",
    default=1,
    type=int,
    help="clip skip"
)
parser.add_argument(
    "--inference_steps",
    default=30,
    type=int,
    help="amount of inference steps"
)
parser.add_argument(
    "--guidance_scale",
    default=7,
    type=float,
    help="guidance scale"
)
parser.add_argument(
    "--width",
    default=512,
    type=int,
    help="width of output image"
)
parser.add_argument(
    "--height",
    default=512,
    type=int,
    help="height of output image"
)
parser.add_argument(
    "--task",
    default="txt2img",
    type=str,
    help="Task to preform, txt2img, img2img"
)
parser.add_argument(
    "--lora_path",
    default="models/lora",
    type=str,
    help="path to directory storing lora"
)
parser.add_argument(
    "--batch_size",
    default=1,
    type=int,
    help="batch size"
)
parser.add_argument(
    "--embed_prompts",
    action="store_true",
    help="For longer prompts with 77 or more tokens use this flag to first make them embeddings."
)
parser.add_argument(
    "--help_list_lora",
    action="store_true",
    help="Use this flag to find and list lora. This will only look in the models/lora directory."
)
parser.add_argument(
    "--help_list_models",
    action="store_true",
    help="Use this flag to find and list lora. This will only look in the models/lora directory."
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

model_path = args.model_path
if path.isdir(model_path):
    print(f"{co.neutral}Model path: {co.green}{model_path}{co.reset}")
elif path.isfile(model_path):
    fname, ext = path.splitext(model_path)
    new_path = path.join(path.abspath("./models/"), fname)
    if path.isdir(new_path):
        print(f"{co.green}{new_path} {co.neutral}has already been created.\nUsing {co.green}{new_path}{co.neutral} as model.{co.reset}")
        model_path = new_path
    else:
        print(f"{co.neutral}Attempting to convert {co.green}{model_path}{co.neutral} to huggingface model{co.reset}")
        from_safetensors = True if ext == ".safetensors" else False

        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=model_path,
            original_config_file=None,
            image_size=None,
            prediction_type=None,
            model_type=None,
            extract_ema=False,
            scheduler_type=args.scheduler_type,
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
else:
    print(f"{co.neutral}Model path: {co.red}{model_path}{co.neutral} :does not exist{co.reset}")
    exit()

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
    )
else:
    print(f"{co.red}No current support for {args.task}{co.reset}")

