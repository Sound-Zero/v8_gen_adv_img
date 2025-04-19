import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

all_img_path='./datasets/coco/images/120img/'
target_img_path='./datasets/FGSM120/eps0.15'
place_img_path='./datasets/FGSM120/Reshaped_eps0.15/images'



#确保全波图片都有同名文件
bool_check=True
name_list=os.listdir(all_img_path)
for img_name in name_list:
 
    if not os.path.exists(os.path.join(target_img_path,img_name)):
        bool_check=False
        print(f"缺少目标图片: {img_name}")
        break


print(bool_check)
if bool_check:
    cropped_err=[]
    cropped_count=0
    #根据all_img_path中的图片，调整target_img_path图片的长宽，并复制保存到place_img_path中
    all_name_list=os.listdir(all_img_path)
    target_img_list=os.listdir(target_img_path)


    for i in range(len(name_list)):
        original_img=cv2.imread(os.path.join(all_img_path,all_name_list[i]))
        original_img_shape=original_img.shape

        fgsm_img=cv2.imread(os.path.join(target_img_path,target_img_list[i]))
        #中心裁剪成origin_img_size大小
        fgsm_height, fgsm_width = fgsm_img.shape[:2]
        original_height, original_width = original_img_shape[:2]
        start_y = (fgsm_height - original_height) // 2
        start_x = (fgsm_width - original_width) // 2
        cropped_fgsm_img = fgsm_img[start_y:start_y + original_height, start_x:start_x + original_width]

        cv2.imwrite(os.path.join(place_img_path,target_img_list[i]),cropped_fgsm_img)
        cropped_count+=1
 
print(f"共处理{cropped_count}张图片")





