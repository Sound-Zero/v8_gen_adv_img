import os
import shutil
img_file_path="./datasets/FGSM_last_60+PGD_last_60/images"

all_label_file_path="./datasets/coco/labels/val2017"
place_label_file_path="./datasets/FGSM_last_60+PGD_last_60/labels"

"""
    根据img_file_path内图片文件名称，
    从all_label_file_path中找到对应的label文件，
    复制到place_label_file_path内
"""



# img_name_list=os.listdir(img_file_path)
# img_name_list=[name[:-4] for name in img_name_list if not name.endswith("pgd.jpg")]
# not_found_list=[]
# copy_count=0
# for img_name in img_name_list:
#     label_path=os.path.join(all_label_file_path,img_name+".txt")
#     if os.path.exists(label_path) and os.path.exists(os.path.join(place_label_file_path,img_name+".txt")==False  ):
#         shutil.copy(label_path,os.path.join(place_label_file_path,img_name+".txt"))
#         copy_count+=1
#     else:
#         not_found_list.append(img_name)

# print("not found label:",not_found_list,"\nnot found sum:",len(not_found_list))
# print("copy count:",copy_count)



# img_name_list=os.listdir(img_file_path)
# img_name_list=[name[:-4] for name in img_name_list]#remove ".jpg"

# not_found_list=[]
# copy_count=0
# for img_name in img_name_list:
#     label_path=os.path.join(all_label_file_path,img_name+".txt")
#     if os.path.exists(label_path) and os.path.exists(os.path.join(place_label_file_path,img_name+".txt")==False  ):
#         shutil.copy(label_path,os.path.join(place_label_file_path,img_name+".txt"))
#         copy_count+=1
#     else:
#         not_found_list.append(img_name)

# print("not found label:",not_found_list,"\nnot found sum:",len(not_found_list))
# print("copy count:",copy_count)
