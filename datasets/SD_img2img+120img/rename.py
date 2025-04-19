import os
import shutil
origin_path="./datasets/SD_img2img+120img/SD_eps0.25/labels"
place_path="./datasets/SD_img2img+120img/labels"

label_name_list=os.listdir(origin_path)
label_name_list=[name[:-4] for name in label_name_list]#remove .txt

count=0
for name in label_name_list:
    shutil.copy(os.path.join(origin_path,name+".txt"),os.path.join(place_path,'sd'+name+".txt"))
    count+=1
print(count)