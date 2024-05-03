# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:45:10 2024

@author: tohya

file: test_folder.py
"""

from PIL import Image
import numpy as np

def test_pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        if(len(Image.Image.getbands(img))!=4):
            img=img.convert("RGB")
        else:    
            img=img.convert("RGBX") # in case RGBN(RGB,NIR)
        return img
    
    
if __name__ == "__main__":
    
    file1 = "assets/folder_test_img/cat.jpg"
    file2 = "assets/folder_test_img/ortho_crop.tif"
    
    img1 = test_pil_loader(file1)   
    shape1 = np.array(img1).shape

    img2 = test_pil_loader(file2)   
    shape2 = np.array(img2).shape
    
    print(f"3 bands jpeg image(RBG) shape: {shape1}")
    print(f"4 bands tiff image(RGBN) shape: {shape2}")
    
    