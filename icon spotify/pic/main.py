import os 
from PIL import Image


print (os.listdir("pic"))
"""

"""

files = [['acousticness2023-03-20 23-07-03.jpeg', 'acousticness2023-03-20 23-07-59.jpeg'], ['danceability2023-03-20 23-07-03.jpeg', 'danceability2023-03-20 23-07-58.jpeg'], 
 ['energy2023-03-20 23-07-03.jpeg', 'energy2023-03-20 23-07-58.jpeg'], ['instrumentalness2023-03-20 23-07-03.jpeg', 'instrumentalness2023-03-20 23-07-59.jpeg'], 
 ['key2023-03-20 23-07-03.jpeg', 'key2023-03-20 23-07-58.jpeg'], ['liveness2023-03-20 23-07-03.jpeg', 'liveness2023-03-20 23-07-59.jpeg'], 
 ['loudness2023-03-20 23-07-03.jpeg', 'loudness2023-03-20 23-07-58.jpeg'], 
 ['mode2023-03-20 23-07-03.jpeg', 'mode2023-03-20 23-07-59.jpeg'], ['speechiness2023-03-20 23-07-03.jpeg', 'speechiness2023-03-20 23-07-59.jpeg'], 
 ['tempo2023-03-20 23-07-03.jpeg', 'tempo2023-03-20 23-07-59.jpeg'], ['valence2023-03-20 23-07-03.jpeg', 'valence2023-03-20 23-07-59.jpeg']]

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

