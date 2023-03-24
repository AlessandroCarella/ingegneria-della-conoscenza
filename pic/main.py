import os 
from PIL import Image


print (os.listdir("pic"))
"""

"""

files = [[], []]

for couple in files:
    #Read the two images
    image1 = Image.open(os.path.join(os.path.abspath("pic"), couple[0]))
    image2 = Image.open(os.path.join(os.path.abspath("pic"), couple[1]))
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save(os.path.join(os.path.abspath("pic"), str(couple[0]) + " + " + str(couple[1]) + ".jpeg"),"JPEG")

