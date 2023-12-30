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
    "--out_dir",
    default="output",
    type=str,
    help="path to store generated images and generation data"
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
parser.add_argument(
    "--allow_tf32",
    action="store_true",
    help="inference will go faster with slight inaccuracy"
)
parser.add_argument(
    "--no_gen_data",
    action="store_false",
    help="Setting this flag prevents generation data from being stored with images"
)
parser.add_argument(
    "--image_path",
    default=None,
    type=str,
    help="if task is img2img, this path indicates the input image"
)
parser.add_argument(
    "--image_strength",
    default=.8,
    type=float,
    help="if task is img2img, this path indicates the input image"
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
        image_path=args.image_path,
        image_strength=args.image_strength,
    )
else:
    print(f"{co.red}No current support for {args.task}{co.reset}")

