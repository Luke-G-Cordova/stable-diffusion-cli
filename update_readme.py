import json
import re

# prepare args to be inserted
with open("args.json", "r") as f:
    args = json.loads(f.read())
sdcli_params = "\n\n<hr/>\n"
for arg_name in args:
    arg = args[arg_name]
    sdcli_params += f"\n ### {arg_name}:\n\n {arg['help']}\n\n flags: `{arg['flags']}`\n"
    for key in arg:
        if key != "help" and key != "flags":
            sdcli_params += f"\n {key}: `{arg[key]}`\n"
    sdcli_params += "\n<hr/>\n"



with open("README.md", "r") as f:
    readme = f.read()
    readme = re.sub("(?<=<!-- sdcli_params_start -->)[\S\s]+(?=<!-- sdcli_params_end -->)", sdcli_params, readme)
with open("README.md", "w") as f:
    f.write(readme)
    