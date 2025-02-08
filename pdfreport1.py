# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:25:13 2023

@author: youm
"""

from PIL import Image  # install by > python3 -m pip install --upgrade Pillow  # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation
import os

DIR = r'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\pipeline output\Nov 15'#r'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\pipeline output\july 7'#r'Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\pipeline output\june 13'
bxs = ['14_bx1','14_bx2','14_bx2_B..vs..14_bx1_B']   #'15_bx1','15_bx2']

need to make it so 14_bx1 doesn't include ..14_bx1_B'



def main():
    for bx in bxs:
        images = []
        for fo in sorted(os.listdir(DIR)):
            fold = DIR+'/'+fo
            if not os.path.isdir(fold):
                continue
            if len(os.listdir(fold)) < 2:
                continue
            if bx not in fold:
                continue
            #if "barp" in fold:
                #continue
            files = os.listdir(fold)
            files = sorted(files,key=lambda x: os.path.getmtime(fold+"/"+x))
            print(files)
            #continue

            for file in files:
                print(fold,file)
                images.append(Image.open(fold+'/'+file).convert('RGB'))
                print(len(images))
        if len(images) == 0:
            continue
        pdf_path = DIR+'/'+bx+'.pdf'
        images[0].save(
            pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:]
        )


if __name__ == "__main__":
    main()