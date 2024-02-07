import argparse
from PIL import Image
from pathlib import Path
import json
import pandas as pd
from html2image import Html2Image
import imgkit
import os
pd.set_option('display.max_colwidth', None)


# convert your links to html tags 
def path_to_image_html(path):
    return '<img src="'+ path + '" width="128" >'


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Utility to compare different checkpoints")
    parser.add_argument("--images", type=Path, help="Path to dirs with images")
    parser.add_argument("--out", type=str, default="img-table", help="Name for output files")
    args = parser.parse_args()

    header = ['prompt']
    rows = []

    for dir in sorted(args.images.iterdir()):
        header.append(dir.name)
        with open(dir / 'description.json') as file:
            data = json.load(file)['prompted']
        first_dir = len(rows) == 0
        for i, (prompt, imgs) in enumerate(data.items()):
            if first_dir:
                rows.append([prompt])
            rows[i].append(str(dir / imgs[0]))
    
    df = pd.DataFrame(columns=header, data=rows)

    # Create the dictionariy to be passed as formatters
    format_dict = {}
    for image_col in header[1:]:
        format_dict[image_col] = path_to_image_html
    
    df.to_html(f'{args.out}.html', escape=False, formatters=format_dict)
    html = df.to_html(escape=False, formatters=format_dict)

    os.system(f"wkhtmltoimage --enable-local-file-access {args.out}.html {args.out}.jpg")
