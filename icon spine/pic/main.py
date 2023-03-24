import os 
from PIL import Image

"""

"""

files = [['cervical tilt2023-03-19 23-58-04.jpeg', 'cervical tilt2023-03-19 23-58-26.jpeg'], ['degree spondylolisthesis2023-03-19 23-57-40.jpeg', 'degree spondylolisthesis2023-03-19 23-58-20.jpeg'], 
 ['direct tilt2023-03-19 23-58-01.jpeg', 'direct tilt2023-03-19 23-58-23.jpeg'], ['lumbar lordosis angle2023-03-19 23-57-35.jpeg', 'lumbar lordosis angle2023-03-19 23-58-15.jpeg'], 
 ['pelvic incidence2023-03-19 23-57-32.jpeg', 'pelvic incidence2023-03-19 23-58-12.jpeg'], ['pelvic radius2023-03-19 23-57-38.jpeg', 'pelvic radius2023-03-19 23-58-18.jpeg'], 
 ['pelvic slope2023-03-19 23-57-41.jpeg', 'pelvic slope2023-03-19 23-58-21.jpeg'], ['pelvic tilt2023-03-19 23-57-34.jpeg', 'pelvic tilt2023-03-19 23-58-14.jpeg'], 
 ['sacral slope2023-03-19 23-57-37.jpeg', 'sacral slope2023-03-19 23-58-17.jpeg'], ['sacrum angle2023-03-19 23-58-06.jpeg', 'sacrum angle2023-03-19 23-58-28.jpeg'], 
 ['scoliosis slope2023-03-19 23-58-08.jpeg', 'scoliosis slope2023-03-19 23-58-29.jpeg'], ['thoracic slope2023-03-19 23-58-03.jpeg', 'thoracic slope2023-03-19 23-58-25.jpeg']]

for couple in files:
    image1 = Image.open(os.path.join(os.path.abspath("pic"), couple[0]))
    image2 = Image.open(os.path.join(os.path.abspath("pic"), couple[1]))
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save(os.path.join(os.path.abspath("pic"), str(couple[0]) + " + " + str(couple[1]) + ".jpeg"),"JPEG")

