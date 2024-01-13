# stable-diffusion-cli

[Stable diffusion](https://github.com/CompVis/stable-diffusion) is a diffusion based image generation ai. There are several open source projects like [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allow people to generate images manually using a web interface. This projects main purpose is to automate this task. The workflow is as follows:

- Utilize a web interface tool like Automatic1111 to determine a combination of constants. Prompts, variables, and weights all contribute to making consistent and interesting image in some style.
- Use these constants and some variables with this project to automate making these images.

# Docs

## How To's

### [Inferencing:](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference)

`$ python sdcli.py --model_path "model" --prompt "astronaut"`

Inferencing in the context of this project basically refers to generating an image. To start inferencing you need a [checkpoint](#checkpoint) and a [prompt](#prompt-syntax).

Set a width and height using `--width` and `--height` flags. The width/height must be devisable by 4, good values for width or height tend to be 512 or 768.

Set a negative prompt using `--negative_prompt` flag. This prompt contains all things you don't want in the photo. It tends to be a list of things like bad quality, mutated hands, grainy photo, etc.

Set a seed using `--seed` flag. If left as default, a random seed will be used. This value is a random number and is what makes photos different despite their prompts being the same.

Set amount of inference steps using the `--inference_steps` flag. This is how many iterations of diffusion is done to the image. Generally this is a number between 20 and 35. Be careful, the higher this number the longer it takes to generate an image and more inference steps does not mean a better image.

Set the cfg or guidance scale using the `--guidance_scale` flag. This scale is the amount the model will follow the prompt. The higher the guidance scale the more strictly the model will follow the prompt. Good values are usually between 5 and 14 but 7 is the most common value by far.

Set the batch size or the amount of photos generated using `--batch_size` flag. This variable is meant for generating many photos of the same prompt with different seeds and comparing to see how reliable the prompt is. This is not really the scope or purpose of this project but I included the functionality anyway. If you are looking for this functionality specifically, try out [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)!

### [Checkpoint:](https://stable-diffusion-art.com/models/#:~:text=Updated%20December%205%2C%202023%20By,depends%20on%20the%20training%20images.)

`$ python sdcli.py --model_path "path/to/model/" --prompt "astronaut"`

This command will inference with the specified model.

If `model_path` is a .safetensors or .ckpt file, it will automatically convert it into a huggingface diffusers model directory and store this directory in `models/checkpoint/<model_name>/`. It will then inference using that newly created model.

If `model_path` is the name of a directory stored in `models/checkpoint/` this command will use that directory. For example if a directory `models/checkpoint/my_model/` exists, it can be selected by only specifying `my_model` like the following: `$ python sdcli.py --model_path my_model --prompt "astronaut`

If `model_path` is not a recognizable name of a directory or file, the program will attempt to query [Civitai](https://civitai.com/) for a checkpoint to use. Checkpoints tend to be big so this could take a while.

[Civitai](https://civitai.com/) and [Huggingface](https://huggingface.co/) both have several checkpoint models and other types of models for download. Also look at [downloading](#downloading).

### [LoRA:](https://huggingface.co/blog/lora)

`$ python sdcli.py --model_path "model" --lora_path "path/to/lora_dir/" --prompt "astronaut, <lora:some_lora:.75>"`

This command applies `some_lora` with a weight of .75 and inferences. `some_lora` is the name of a lora directory stored at `<lora_path>/some_lora`. **CURRENTLY ALL LORA MUST BE DOWNLOADED WITH THE `download.py` SCRIPT [MORE DETAILS](#downloading)**

All used lora must be referenced in the prompt using the syntax `<lora:{lora_name}:{lora_weight}>`. Any substring in the prompt that follows this syntax will be cut from the prompt so the final prompt will not include it. Also if your lora has any trigger words, you can specify `<lora:{lora_name}:{lora_weight}:add_trained>` and those trigger words will be added, comma separated, to the prompt. Go to [Prompt Syntax](#prompt-syntax) for more details.

### [Samplers/Schedulers:](https://huggingface.co/docs/diffusers/api/schedulers/overview)

`$ python sdcli.py --model_path "model" --scheduler_type "dpm"`

Currently the only schedulers available in this project are [pndm](https://huggingface.co/docs/diffusers/api/schedulers/pndm), [lms](https://huggingface.co/docs/diffusers/api/schedulers/lms_discrete), [ddim](https://huggingface.co/docs/diffusers/api/schedulers/ddim), [euler](https://huggingface.co/docs/diffusers/api/schedulers/euler), [euler-ancestral](https://huggingface.co/docs/diffusers/api/schedulers/euler_ancestral), and [dpm](https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver). This project is set to always use [karras sigmas](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/4384#discussioncomment-4562593) at the moment.

### Task - [txt2img](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img), [img2img:](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)

`$ python sdcli.py --model_path "model" --task "txt2img"`

There are two tasks to choose from `txt2img` and `img2img`. With `txt2img` you can generate an image based on a prompt. With `img2img` you can generate an image given both a prompt and an input image. For `img2img` the flags `--image_path` and `--image_strength` are used. `--image_path` is the path to the input image, `--image_strength` is the amount of impact the image will have on the final photo between 0 and 1. The higher the strength, the more creativity or freedom the model will with the final photo and likewise the lower the strength, the more the final photo will resemble the original.

### [Textual Inversions - Embeddings:](https://huggingface.co/docs/diffusers/using-diffusers/textual_inversion_inference)

`$ python sdcli.py --model_path "model" --embeddings_path "path/to/embeddings_dir/" --prompt "astronaut" --negative_prompt "embedding_trained_word"`

Given `embedding_trained_word` is a trained word from one of the embeddings files at `--embeddings_path` this command will include the embedding. `embedding_trained_word` will not be removed from final prompt. Textual Inversions are often negative prompt concepts to help the model not generate these bad concepts. Look at [downloading](#downloading) for more details about downloading embeddings.

### [Poses:](https://huggingface.co/lllyasviel/sd-controlnet-openpose)

`$ python sdcli.py --model_path "model" --pose_path "path/to/pose_dir/" --prompt "athlete, <pose:jumping_pose>"`

Currently this project only supports [openPose](https://huggingface.co/lllyasviel/sd-controlnet-openpose) controlnet.

A used pose must be referenced in the prompt using the syntax `<pose:{pose_name}>`. The pose must be a .png file stored in a directory under the `--pose_path`. Go to [Prompt Syntax](#prompt-syntax) for more details on prompt syntax. Go to [downloading](#downloading) for more details about downloading poses.

## Commands

`$ python sdcli.py`
this is the main command to run. It allows inferencing via the cli.

### Arguments:

 <!-- sdcli_params_start -->

> ### model_path:
>
> Path to the base model to run stable diffusion on. If evaluates to .ckpt or .safetensor file will convert to and use a huggingface diffusers model stored in ./models
>
> flags: `['-mp', '--model_path']`
>
> default: `./models/stable-diffusion-v2`
>
> type: `str`

> ### scheduler_type:
>
> scheduler type. One of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']
>
> flags: `['-st', '--scheduler_type']`
>
> default: `pndm`
>
> type: `str`

> ### embeddings_path:
>
> Path to directory containing embedding files.
>
> flags: `['-ep', '--embeddings_path']`
>
> default: `models/embeddings`
>
> type: `str`

> ### prompt:
>
> text prompt
>
> flags: `['-p', '--prompt']`
>
> default: `None`
>
> type: `str`

> ### negative_prompt:
>
> negative text prompt
>
> flags: `['-n', '--negative_prompt']`
>
> default: `None`
>
> type: `str`

> ### prompt_file:
>
> A file storing the prompt. --prompt flag must not be set to use this
>
> flags: `['-pp', '--prompt_file']`
>
> default: `prompts/p.txt`
>
> type: `str`

> ### negative_prompt_file:
>
> A file storing the negative prompt. --negative_prompt flag must not be set to use this
>
> flags: `['-np', '--negative_prompt_file']`
>
> default: `prompts/n.txt`
>
> type: `str`

> ### seed:
>
> Seed. If seed is -1 a random seed will be generated and used
>
> flags: `['-s', '--seed']`
>
> default: `-1`
>
> type: `int`

> ### clip_skip:
>
> clip skip
>
> flags: `['-c', '--clip_skip']`
>
> default: `1`
>
> type: `int`

> ### inference_steps:
>
> amount of inference steps
>
> flags: `['-i', '--inference_steps']`
>
> default: `30`
>
> type: `int`

> ### guidance_scale:
>
> guidance scale
>
> flags: `['-g', '-cfg', '--guidance_scale']`
>
> default: `7`
>
> type: `float`

> ### width:
>
> width of output image
>
> flags: `['--width']`
>
> default: `512`
>
> type: `int`

> ### height:
>
> height of output image
>
> flags: `['--height']`
>
> default: `512`
>
> type: `int`

> ### task:
>
> Task to preform, txt2img, img2img
>
> flags: `['--task']`
>
> default: `txt2img`
>
> type: `str`

> ### lora_path:
>
> path to directory storing lora
>
> flags: `['-lp', '--lora_path']`
>
> default: `models/lora`
>
> type: `str`

> ### out_dir:
>
> path to store generated images and generation data
>
> flags: `['-op', '--out_dir']`
>
> default: `output`
>
> type: `str`

> ### out_name:
>
> name added to output file along with its seed. Will store as <out_dir>/<out_name><seed>.png
>
> flags: `['-on', '--out_name']`
>
> default: ``
>
> type: `str`

> ### batch_size:
>
> batch size
>
> flags: `['-bs', '--batch_size']`
>
> default: `1`
>
> type: `int`

> ### embed_prompts:
>
> For longer prompts with 77 or more tokens use this flag to first make them embeddings.
>
> flags: `['-e', '--embed_prompts']`
>
> action: `store_true`

> ### help_list_lora:
>
> Use this flag to find and list lora. This will only look in the models/lora directory.
>
> flags: `['-hl', '--help_list_lora']`
>
> action: `store_true`

> ### help_list_models:
>
> Use this flag to find and list lora. This will only look in the models/lora directory.
>
> flags: `['-hm', '--help_list_models']`
>
> action: `store_true`

> ### allow_tf32:
>
> inference will go faster with slight inaccuracy
>
> flags: `['-atf32', '--allow_tf32']`
>
> action: `store_true`

> ### no_gen_data:
>
> Setting this flag prevents generation data from being stored with images
>
> flags: `['-ng', '--no_gen_data']`
>
> action: `store_false`

> ### image_path:
>
> if task is img2img, this path indicates the input image
>
> flags: `['-ip', '--image_path']`
>
> default: `None`
>
> type: `str`

> ### image_strength:
>
> if task is img2img, this path indicates the input image
>
> flags: `['-is', '--image_strength']`
>
> default: `0.8`
>
> type: `float`

> ### vae_path:
>
> path to custom vae file. If none is provided, will attempt to use default.
>
> flags: `['-vp', '--vae_path']`
>
> default: `None`
>
> type: `str`

> ### pose_path:
>
> path to custom pose file. If none is provided, will attempt to use default.
>
> flags: `['-sp', '--pose_path']`
>
> default: `models/poses`
>
> type: `str`

<!-- sdcli_params_end -->
