# stable-diffusion-cli

[Stable diffusion](https://github.com/CompVis/stable-diffusion) is a diffusion based image generation ai. There are several open source projects like [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allow people to generate images manually using a web interface. This projects main purpose is to automate this task. The workflow is as follows:

- Utilize a web interface tool like Automatic1111 to determine a combination of constants. Prompts, variables, and weights all contribute to making consistent and interesting image in some style.
- Use these constants and some variables with this project to automate making these images.

# Docs

# Install and setup

## [Conda:](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

You can choose from 1 of 2 environments. The difference is that `environment_cu118.yml` uses `cuda 11.8` and `environment_cu121.yml` uses `cuda 12.1`. If you are not sure which cuda to use, you should probably just use `cuda 11.8` but I added 2 separate envs in case you need to support `cuda 12.1` instead.

#### cuda 11.8

`$ conda env create -f environment_cu118.yml`

#### cuda 12.1

`$ conda env create -f environment_cu121.yml`

Use this command to download and install the dependencies, this may take a while. Once this is done running you should have a conda environment named sdcli. Run the following to activate the environment:

`$ conda activate sdcli`

You are now working in the new environment and can start inferencing! See [below.](#inferencing)

# How To's

## [Inferencing:](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference)

`$ python sdcli.py --model_path "model" --prompt "astronaut"`

Inferencing in the context of this project basically refers to generating an image. To start inferencing you need a [checkpoint](#checkpoint) and a [prompt](#prompt-syntax).

Set a width and height using `--width` and `--height` flags. The width/height must be devisable by 4, good values for width or height tend to be 512 or 768.

Set a negative prompt using `--negative_prompt` flag. This prompt contains all things you don't want in the photo. It tends to be a list of things like bad quality, mutated hands, grainy photo, etc.

Set a seed using `--seed` flag. If left as default, a random seed will be used. This value is a random number and is what makes photos different despite their prompts being the same.

Set amount of inference steps using the `--inference_steps` flag. This is how many iterations of diffusion is done to the image. Generally this is a number between 20 and 35. Be careful, the higher this number the longer it takes to generate an image and more inference steps does not mean a better image.

Set the cfg or guidance scale using the `--guidance_scale` flag. This scale is the amount the model will follow the prompt. The higher the guidance scale the more strictly the model will follow the prompt. Good values are usually between 5 and 14 but 7 is the most common value by far.

Set the batch size or the amount of photos generated using `--batch_size` flag. This variable is meant for generating many photos of the same prompt with different seeds and comparing to see how reliable the prompt is. This is not really the scope or purpose of this project but I included the functionality anyway. If you are looking for this functionality specifically, try out [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)!

## Prompt Syntax:

The prompt parser takes the prompt and extracts the useful information from it so that it can then add the proper models without adding something that shouldn't be used. Currently the information the parser needs to extract are as follows:

### Lora names and lora weight.

The parser needs to know the `lora_name`, `lora_weight`, and it needs to know if it should add the loras trained words. The syntax to let the parser know this information is a string of the following format:

`<lora:{lora_name}:{lora_weight}>`

or

`<lora:{lora_name}:{lora_weight}:add_trained>`

`lora_name` should correspond to the name of directory of the lora that should be added in the `--lora_path`. For example if `--lora_path` is `models/lora/` and there is a lora in that path `models/lora/my_interesting_lora/` then the `lora_name` should be `my_interesting_lora`. Say you wanted to apply this lora with a weight of .75, the syntax to tell the parser this would be:

`<lora:my_interesting_lora:.75>`

If `my_interesting_lora` has trigger words or trained words, you could also put:

`<lora:my_interesting_lora:.75:add_trained>`

Which would add all of the trigger words to the final prompt for you.

All lora syntax found in the prompt accept for the trained words gets removed from the final prompt and is not processed by the model. For example if `my_interesting_lora` had a trained word that was `elephant` and your prompt was:

`1boy, smiling, standing, sunset, <lora:my_interesting_lora:.75:add_trained>`

The `my_interesting_lora` would be applied to the model with a weight of .75 and the final prompt passed to the model for inference would be:

`1boy, smiling, standing, sunset, elephant`

### Textual Inversion embeddings.

The parser needs to know which embeddings need to be included in the model. It does this by collecting all trained words from embeddings in the `--embeddings_path` and searching through the prompt for each trained word. If it finds one of the trained words in the prompt, it includes that embedding in the model. If you want to include an embedding, you must include its trained word in the prompt somewhere. This trained word does not get removed from the final prompt and is passed to the model for inference.

If having trouble downloading embeddings reference [downloading](#downloading)

### Pose names.

Currently the parser only supports the [openpose controlnet](https://huggingface.co/lllyasviel/sd-controlnet-openpose) but will support more in the future. Also poses are only supported with the `--task` flag set to `img2img` see [poses](#poses). To let the parser know which pose to include use the following syntax:

`<pose:{pose_name}>`

`pose_name` should correspond to the name of directory of the pose that should be added in the `--pose_path`. For example if `--pose_path` is `models/poses/` and there is a pose in that path `models/poses/my_unique_pose/` then the `pose_name` should be `my_unique_pose` and the full syntax should be:

`<pose:my_unique_pose>`

Similarly to the lora syntax, the pose syntax is removed from the final prompt before inferencing. If your prompt is:

`five people, standing, <pose:my_unique_pose>`

The `my_unique_pose` will be inputted to the model and the final prompt that gets passed to the model for inferencing will be:

`five people, standing`

### Misc.

You can alternatively supply a prompt and negative prompt via separate files using the `--prompt_file` and `--negative_prompt_file` flags respectively. I find it much easier to use this feature because it allows for better editing of my prompts and I can pre save prompts this way. Do not set the `--prompt` or `--negative_prompt` flags if you wish to use files. Return characters or \n characters are ignored which enables formatting these files as lists.

I included a commenting functionality to comment out parts of a prompt. Use `//` to comment out a line to be ignored by the parser. Anything after and including the `//` in a line will not be included in the final prompt.

## Downloading:

`$ python download.py --task "query" --query "some model"`

This script allows you to download a model from [civitai](https://civitai.com/) using the [civitai api](https://github.com/civitai/civitai/wiki/REST-API-Reference)

The `--task` flag accepts one of `query` or `v_id`. If the task is query, the [models](https://github.com/civitai/civitai/wiki/REST-API-Reference#get-apiv1models) endpoint is used, the `--query` flag is a search string. In this case the `--kwargs` flag can store a json string containing any query parameters for the search. If the task is v_id, the [model-versions](https://github.com/civitai/civitai/wiki/REST-API-Reference#get-apiv1models-versionsmodelversionid) endpoint is used, the `--query` flag must be a specific models version id number.

To reliably download a specific model ([Checkpoint](#checkpoint), [LoRA](#lora), [Textual Inversion](#textual-inversions---embeddings), Vae, [Pose](#poses), etc.) you need its version id. Eventually I may make a browser extension that makes this easier because getting these version id's is definitely a workaround right now but for the time being there are two places you can look for this number. The first is when you go to [civitai](https://civitai.com/models) and find a model, click it, and in the url at the end it will sometimes say `?modelVersionId=some_number`. Copy the some_number only and use that. If the url doesn't include the number which sometimes it won't You can look in the second place. Go to the model you want, on the right hand side under the Details panel is a File panel dropdown. Click that to open the drop down. Now inspect the download button on the file but don't download it, on Chrome if you right click the download button and hit inspect, this will open up the dev tools. There should be a long link that looks something like this `href="/api/download/models/276923?type=Model&format=SafeTensor&size=full&fp=fp16"` close to the highlighted element. The long number, in this case `276923`, is the version id number and you can use that.

...Told you it was a work around 😅

When you have the version id, set the `--query` flag to that number.

## [Checkpoint:](https://stable-diffusion-art.com/models/#:~:text=Updated%20December%205%2C%202023%20By,depends%20on%20the%20training%20images.)

`$ python sdcli.py --model_path "path/to/model/" --prompt "astronaut"`

This command will inference with the specified model.

If `model_path` is a .safetensors or .ckpt file, it will automatically convert it into a huggingface diffusers model directory and store this directory in `models/checkpoint/<model_name>/`. It will then inference using that newly created model.

If `model_path` is the name of a directory stored in `models/checkpoint/` this command will use that directory. For example if a directory `models/checkpoint/my_model/` exists, it can be selected by only specifying `my_model` like the following: `$ python sdcli.py --model_path my_model --prompt "astronaut`

If `model_path` is not a recognizable name of a directory or file, the program will attempt to query [Civitai](https://civitai.com/) for a checkpoint to use. Checkpoints tend to be big so this could take a while.

[Civitai](https://civitai.com/) and [Huggingface](https://huggingface.co/) both have several checkpoint models and other types of models for download. Also look at [downloading](#downloading).

## [LoRA:](https://huggingface.co/blog/lora)

`$ python sdcli.py --model_path "model" --lora_path "path/to/lora_dir/" --prompt "astronaut, <lora:some_lora:.75>"`

This command applies `some_lora` with a weight of .75 and inferences. `some_lora` is the name of a lora directory stored at `<lora_path>/some_lora`. **CURRENTLY ALL LORA MUST BE DOWNLOADED WITH THE `download.py` SCRIPT [MORE DETAILS](#downloading)**

All used lora must be referenced in the prompt using the syntax `<lora:{lora_name}:{lora_weight}>`. Any substring in the prompt that follows this syntax will be cut from the prompt so the final prompt will not include it. Also if your lora has any trigger words, you can specify `<lora:{lora_name}:{lora_weight}:add_trained>` and those trigger words will be added, comma separated, to the prompt. Go to [Prompt Syntax](#prompt-syntax) for more details.

## [Samplers/Schedulers:](https://huggingface.co/docs/diffusers/api/schedulers/overview)

`$ python sdcli.py --model_path "model" --scheduler_type "dpm"`

Currently the only schedulers available in this project are [pndm](https://huggingface.co/docs/diffusers/api/schedulers/pndm), [lms](https://huggingface.co/docs/diffusers/api/schedulers/lms_discrete), [ddim](https://huggingface.co/docs/diffusers/api/schedulers/ddim), [euler](https://huggingface.co/docs/diffusers/api/schedulers/euler), [euler-ancestral](https://huggingface.co/docs/diffusers/api/schedulers/euler_ancestral), and [dpm](https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver). This project is set to always use [karras sigmas](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/4384#discussioncomment-4562593) at the moment.

## Task - [txt2img](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img), [img2img:](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)

`$ python sdcli.py --model_path "model" --task "txt2img"`

There are two tasks to choose from `txt2img` and `img2img`. With `txt2img` you can generate an image based on a prompt. With `img2img` you can generate an image given both a prompt and an input image. For `img2img` the flags `--image_path` and `--image_strength` are used. `--image_path` is the path to the input image, `--image_strength` is the amount of impact the image will have on the final photo between 0 and 1. The higher the strength, the more creativity or freedom the model will with the final photo and likewise the lower the strength, the more the final photo will resemble the original.

## [Textual Inversions - Embeddings:](https://huggingface.co/docs/diffusers/using-diffusers/textual_inversion_inference)

`$ python sdcli.py --model_path "model" --embeddings_path "path/to/embeddings_dir/" --prompt "astronaut" --negative_prompt "embedding_trained_word"`

Given `embedding_trained_word` is a trained word from one of the embeddings files at `--embeddings_path` this command will include the embedding. `embedding_trained_word` will not be removed from final prompt. Textual Inversions are often negative prompt concepts to help the model not generate these bad concepts. Look at [downloading](#downloading) for more details about downloading embeddings.

## [Vae:](https://en.wikipedia.org/wiki/Stable_Diffusion#:~:text=Stable%20Diffusion%20consists%20of%203,semantic%20meaning%20of%20the%20image.)

`$ python sdcli.py --model_path "model" --vae_path "path/to/vae_file.safetensors"`

To use a custom vae, provide a path to the vae file. This vae will then be applied to the model before inference.

## [Poses:](https://huggingface.co/lllyasviel/sd-controlnet-openpose)

`$ python sdcli.py --model_path "model" --pose_path "path/to/pose_dir/" --task "img2img" --prompt "athlete, <pose:jumping_pose>"`

Currently this project only supports [openPose](https://huggingface.co/lllyasviel/sd-controlnet-openpose) controlnet.

A used pose must be referenced in the prompt using the syntax `<pose:{pose_name}>`. The pose must be a .png file stored in a directory under the `--pose_path`. Go to [Prompt Syntax](#prompt-syntax) for more details on prompt syntax. Go to [downloading](#downloading) for more details about downloading poses.

To use a pose, the `--task` flag must be set to `img2img`. `txt2img` will remove the pose syntax from the final prompt but will NOT include the pose in the model only `img2img` will include the pose in the final model.

## Commands

## `$ python sdcli.py`

This is the main command to run for inferencing. It allows inferencing via the cli.

### Arguments:

<!-- sdcli_params_start -->

<hr/>

### model_path:

Path to the base model checkpoint to run stable diffusion on. If evaluates to `.ckpt` or `.safetensor` file will convert to and use a huggingface diffusers model stored in `./models` by the same name as the original.

flags: `['-mp', '--model_path']`

default: `./models/stable-diffusion-v2`

type: `str`

<hr/>

### scheduler_type:

Scheduler type also known as Sampler type. Can be one of `['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']`

flags: `['-st', '--scheduler_type']`

default: `pndm`

type: `str`

<hr/>

### embeddings_path:

Path to directory containing textual inversion embedding files.

flags: `['-ep', '--embeddings_path']`

default: `models/embeddings`

type: `str`

<hr/>

### prompt:

Text prompt used for inferencing

flags: `['-p', '--prompt']`

default: `None`

type: `str`

<hr/>

### negative_prompt:

Negative text prompt used for inferencing

flags: `['-n', '--negative_prompt']`

default: `None`

type: `str`

<hr/>

### prompt_file:

A file storing the prompt. --prompt flag must not be set to use this.

flags: `['-pp', '--prompt_file']`

default: `prompts/p.txt`

type: `str`

<hr/>

### negative_prompt_file:

A file storing the negative prompt. --negative_prompt flag must not be set to use this.

flags: `['-np', '--negative_prompt_file']`

default: `prompts/n.txt`

type: `str`

<hr/>

### seed:

Seed. If seed is -1 a random seed will be generated and used.

flags: `['-s', '--seed']`

default: `-1`

type: `int`

<hr/>

### clip_skip:

clip skip

flags: `['-c', '--clip_skip']`

default: `1`

type: `int`

<hr/>

### inference_steps:

Amount of inference steps.

flags: `['-i', '--inference_steps']`

default: `30`

type: `int`

<hr/>

### guidance_scale:

Guidance scale, cfg scale.

flags: `['-g', '-cfg', '--guidance_scale']`

default: `7`

type: `float`

<hr/>

### width:

Width of output image.

flags: `['--width']`

default: `512`

type: `int`

<hr/>

### height:

Height of output image.

flags: `['--height']`

default: `512`

type: `int`

<hr/>

### task:

Task to preform. One of txt2img or img2img.

flags: `['--task']`

default: `txt2img`

type: `str`

<hr/>

### lora_path:

Path to directory storing lora.

flags: `['-lp', '--lora_path']`

default: `models/lora`

type: `str`

<hr/>

### out_dir:

Path to store generated images and generation data.

flags: `['-op', '--out_dir']`

default: `output`

type: `str`

<hr/>

### out_name:

Name added to output file along with its seed. Will store as <out_dir>/<out_name><seed>.png

flags: `['-on', '--out_name']`

default: ``

type: `str`

<hr/>

### group_by_seed:

This flag will store each generated image in a dir named after its seed. It's path will be <out_dir>/<seed>/<out_name><seed>.png

flags: `['-gbs', '--group_by_seed']`

action: `store_true`

<hr/>

### batch_size:

batch size

flags: `['-bs', '--batch_size']`

default: `1`

type: `int`

<hr/>

### embed_prompts:

For longer prompts with 77 or more tokens use this flag to first make them embeddings.

flags: `['-e', '--embed_prompts']`

action: `store_true`

<hr/>

### help_list_lora:

Use this flag to find and list lora. This will only look in the models/lora directory.

flags: `['-hl', '--help_list_lora']`

action: `store_true`

<hr/>

### help_list_models:

Use this flag to find and list lora. This will only look in the models/lora directory.

flags: `['-hm', '--help_list_models']`

action: `store_true`

<hr/>

### allow_tf32:

Inference will go faster with slight inaccuracy.

flags: `['-atf32', '--allow_tf32']`

action: `store_true`

<hr/>

### no_gen_data:

Setting this flag prevents generation data from being stored with images.

flags: `['-ng', '--no_gen_data']`

action: `store_false`

<hr/>

### image_path:

If task is img2img, this path indicates the input image.

flags: `['-ip', '--image_path']`

default: `None`

type: `str`

<hr/>

### image_strength:

If task is img2img, this path indicates the input image.

flags: `['-is', '--image_strength']`

default: `0.8`

type: `float`

<hr/>

### vae_path:

Path to custom vae file.

flags: `['-vp', '--vae_path']`

default: `None`

type: `str`

<hr/>

### pose_path:

Path to custom pose directory. If none is provided, will attempt to use default.

flags: `['-sp', '--pose_path']`

default: `models/poses`

type: `str`

<hr/>

### no_save_init_image:

Include this flag to not save the init image for img2img

flags: `['-nsii', '--no_save_init_image']`

action: `store_true`

<hr/>
<!-- sdcli_params_end -->
