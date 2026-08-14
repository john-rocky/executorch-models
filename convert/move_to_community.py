"""Move all shipped repos from mlboydaisuke to executorch-community.

Run ONLY after the org join request is approved (needs write role in the org).
HF sets up automatic redirects from the old URLs, so published links keep working.
Afterwards, update the GitHub catalog links: see sed line at the bottom.
"""
import sys

from huggingface_hub import HfApi

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ship_hf import SHIP

ORG = "executorch-community"

api = HfApi()
user = api.whoami()["name"]
dry = "--go" not in sys.argv
for name, *_ in SHIP:
    src = f"{user}/{name}"
    dst = f"{ORG}/{name}"
    if dry:
        print(f"would move {src} -> {dst}")
    else:
        api.move_repo(from_id=src, to_id=dst, repo_type="model")
        print(f"moved {src} -> {dst}")

if dry:
    print("\ndry run — rerun with --go after org membership is confirmed")
else:
    print("\nnow update catalog links:")
    print("  cd ~/code/executorch-models && "
          "sed -i '' 's|huggingface.co/mlboydaisuke/|huggingface.co/executorch-community/|g' "
          "README.md cards/*.md && git commit -am 'point links at executorch-community' && git push")
