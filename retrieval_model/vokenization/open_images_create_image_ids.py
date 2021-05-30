import json
import os
from pathlib import Path
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

imageDir = '/data/t-zujieliang/data_for_vokenization/open_images/train'

savedids = '/home/t-zujieliang/projects/vokenization/data/vokenization/images/open_images_500k.ids'


with open(savedids, 'w') as idfile:
    for dirpath, dirnames, filenames in os.walk(imageDir):
        for filename in filenames:
            image_id = os.path.splitext(filename)[0]
            idfile.write(image_id + '\n')
            
        