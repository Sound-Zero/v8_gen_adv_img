

import os


def label_rename():
    all_label_path = "./datasets/FGSM120+PGD60/labels"
    label_name_list = os.listdir(all_label_path)
    label_name_list=[name[:-4] for name in label_name_list if (not name.endswith("pgd.txt")) and (not name.endswith("fgsm.txt"))]

    rename_count = 0
    for name in label_name_list:
        rename_count += 1
        os.rename(os.path.join(all_label_path,name+".txt"),os.path.join(all_label_path,name+"fgsm.txt"))
    print(rename_count)

def img_rename():
    all_img_path = "./datasets/FGSM120+PGD60/images"
    img_name_list = os.listdir(all_img_path)
    img_name_list=[name[:-4] for name in img_name_list if (not name.endswith("pgd.jpg")) and (not name.endswith("fgsm.jpg"))]

    rename_count = 0
    for name in img_name_list:
        rename_count += 1
        os.rename(os.path.join(all_img_path,name+".jpg"),os.path.join(all_img_path,name+"fgsm.jpg"))
    print(rename_count)

if __name__ == '__main__':
    label_rename()
    img_rename()

