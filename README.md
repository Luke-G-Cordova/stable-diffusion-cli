# stable-diffusion-cli

[Stable diffusion](https://github.com/CompVis/stable-diffusion) is a diffusion based image generation ai. There are several open source projects like [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allow people to generate images manually using a web interface. This projects main purpose is to automate this task. The workflow is as follows:

- Utilize a web interface tool like Automatic1111 to determine a combination of constants. Prompts, variables, and weights all contribute to making consistent and interesting image in some style.
- Use these constants and some variables with this project to automate making these images.

# Docs

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
