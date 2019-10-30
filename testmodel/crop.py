# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import glob, os
for infile in glob.glob("*.jpg"):
    im = Image.open(infile)
    width, height = im.size
    x = 0
    y = 0
    count = 0
    while (height-y > 256):
        if(width-x>256):
            region = im.crop((x, y,x+256 ,y+256))
            strr = str(count)
            name = strr + infile
            region.save("./pic/"+ name, 'JPEG')
            x = x + 128
            count = count + 1
        else:
            x = 0
            y = y + 128
