import PIL
import os
import os.path
from PIL import Image

f = r'C:/Users/Halil Ramazan/Desktop/mela/jpeg/test'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((224,224))
    img.save(f_img)
f = r'C:/Users/Halil Ramazan/Desktop/mela/jpeg/train'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((224,224))
    img.save(f_img)    