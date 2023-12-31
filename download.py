import argparse
import json
from util.civitai_download import download_file_vid, query_for_file
import util.colors as co

parser = argparse.ArgumentParser()

parser.add_argument(
    "--task",
    default="query",
    type=str,
    help="query or v_id"
)
parser.add_argument(
    "--query",
    default="",
    type=str,
    help="The query to use. If --task is v_id, this should be the version id of the model to download."
)
parser.add_argument(
    "--kwargs",
    default=None,
    type=str,
    help="other query params as a json string"
)

args = parser.parse_args()

if args.task == "query":
    jwargs = json.loads(args.kwargs)
    query_for_file(args.query, **jwargs)
    print(f"{co.neutral}Downloaded{co.reset}")
elif args.task == "v_id":
    download_file_vid(args.query)
    print(f"{co.neutral}Downloaded{co.reset}")
else:
    print(f"{co.red}womp womp{co.reset}")