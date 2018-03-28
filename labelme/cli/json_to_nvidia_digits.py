#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from labelme import utils



def labelme_shapes_to_nvidia_digits(img_shape, shapes, labelcolormap):
    img_digits = Image.new('RGB', tuple(np.flipud(img_shape[:2])))
    draw_digits = ImageDraw.Draw(img_digits)
    for shape in reversed(shapes):
        xy = list(map(tuple, shape['points']))
        color = labelcolormap[shape['label']]
        draw_digits.polygon(xy=xy,
                           outline=color,
                           fill=color)
    return img_digits


def load_classes_and_colormap(classfile, colormapfile):
    classes = open(classfile).readlines()
    colormap_str = open(colormapfile).readlines()   
    if len(classes) != len(colormap_str):
        raise RuntimeError("Class name list and color map are not the same length")
    
    colormap = {}
    for clazz, color in zip(classes, colormap_str):
        rgb = tuple([int(x) for x in color.split(sep=' ')])
        if len(rgb) != 3:
            raise RuntimeError('Incorrect RGB value defined in color map file')
        colormap[clazz.strip()] = rgb
        
    return colormap


def process_json(json_path, color_map, output_path):
    data = json.load(json_path.open())
    img = utils.img_b64_to_arr(data['imageData'])
    labelme_shapes_to_nvidia_digits(img.shape, data['shapes'], color_map).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='json file or directory containing labelme json files')
    parser.add_argument('-o', '--out', default=None)
    parser.add_argument('--classnames', default='class_names.txt',
                        help='File containing list of class names')
    parser.add_argument('--colormap', default='color_map.txt',
                        help='File containing the color map (in same order as class names file!)')
    args = parser.parse_args()

    inp = Path(args.input)

    # Output directory
    # TODO deal with input as directory, iterating over all json files
    if args.out is None:
        out_dir = Path(inp).name.replace('.', '_')
        out_dir = Path(inp).parent / out_dir
    else:
        out_dir = Path(args.out)
        
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    color_map = load_classes_and_colormap(Path(args.classnames), Path(args.colormap))
    
    if inp.is_file():
        output_path = out_dir / inp.name.replace('.json', '.png')
        print('Saving to: {}'.format(output_path.absolute()))
        process_json(inp, color_map, output_path)


if __name__ == '__main__':
    main()
