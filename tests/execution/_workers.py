import json
from pathlib import Path


def record_index(config, index):
    out_dir = Path(config["out_dir"])
    (out_dir / f"{index}.json").write_text(json.dumps({"value": config["value"], "index": index}))
