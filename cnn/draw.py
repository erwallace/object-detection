from utils import parse_xml
from PIL import Image, ImageDraw
import os

def draw_bounding_boxes_from_path(xml_file, png_file, path):
    colour_map = {"with_mask": "green", "without_mask": "red", "mask_weared_incorrect": "yellow"}
    _, boxes, labels = parse_xml(os.path.join(path, xml_file))
    image = Image.open(os.path.join(path, png_file))
    draw = ImageDraw.Draw(image)
    for i, ibox in enumerate(boxes):
        draw.rectangle([(ibox[0], ibox[1]), (ibox[2], ibox[3])], outline=colour_map[labels[i]], width=3)
        draw.text((ibox[0], ibox[1]), text=labels[i])
    return image


def draw_bounding_boxes_from_img_label(img, label):
    colour_map = {0: "red", 1: "yellow", 2: "green"}

    boxes, labels = label['boxes'], label['labels']
    draw = ImageDraw.Draw(img)

    for i, ibox in enumerate(boxes):
        draw.rectangle([(ibox[0], ibox[1]), (ibox[2], ibox[3])], outline=colour_map[int(labels[i])], width=3)
    
    return img