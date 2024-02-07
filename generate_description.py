import argparse
from pathlib import Path
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to generate description.json for all dirs in dataset")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()

    for dir in args.dataset.iterdir():
        if dir.is_file():
            continue
        images = [file for file in dir.iterdir() if file.suffix != 'json']
        j = {"prompted": {}, 
             "normal": [dir.name + '/' + image.name for image in images], 
             "descriptor": "",
             "class": dir.name}
        with open(dir.parent / f'{dir.name}-description.json', 'w') as file:
            print(json.dumps(j), file=file)