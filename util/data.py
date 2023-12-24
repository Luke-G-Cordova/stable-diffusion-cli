import json

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

def generation_data(
    lora_data,
    prompt, 
    negative_prompt,
    embeddings_data
):
    # lora_data = [ (filename, weight), (filename, weight) ]
    lora_objs = {}
    for lora in lora_data:
        file = lora[0]
        weight = lora[1]
        with open(file, 'r') as f:
            info = json.loads(f.read())
            info["applied_weight"] = weight
            lora_objs[info["model"]["name"]] = info
    embeddings_objs = {}
    for emb in embeddings_data:
        with open(emb, 'r') as f:
            info = json.loads(f.read())
            embeddings_objs[info["model"]["name"]] = info
    return {
        "prompt": prompt, 
        "negative_prompt":negative_prompt,
        "lora": lora_objs,
        "textual_inversion_embeddings":  embeddings_objs,
    }