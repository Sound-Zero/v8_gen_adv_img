
import os

def get_file_names(directory, extension):
    """获取指定目录下特定扩展名的文件名列表，并去掉扩展名"""
    try:
        return {file.rsplit('.', 1)[0] for file in os.listdir(directory) if file.endswith(extension)}
    except FileNotFoundError:
        print(f"目录不存在: {directory}")
        return set()

# 根据特定条件查找图片并移动到指定目录
txt_path = 'txt'  # 包含txt文件的目录
all_img_path = 'val_img'  # 所有图片所在的目录
img_place_path = 'finded_image'  # 符合条件的图片存放目录
not_find_txt = "not_find_img.txt"  # 未找到图片的记录文件

# 获取txt文件名和图片文件名
txt_names = get_file_names(txt_path, '.txt')
img_names = get_file_names(all_img_path, '.jpg')

print("txt_num:", len(txt_names))
print("img_num:", len(img_names))

# 确保目标目录存在
os.makedirs(img_place_path, exist_ok=True)

not_found_img_names = []
for name in txt_names:
    if name in img_names:
        try:
            # 移动符合条件的图片
            src_path = os.path.join(all_img_path, f"{name}.jpg")
            dst_path = os.path.join(img_place_path, f"{name}.jpg")
            os.rename(src_path, dst_path)
        except FileNotFoundError:
            not_found_img_names.append(name)
        except OSError as e:
            print(f"移动文件时出错: {e}")
    else:
        not_found_img_names.append(name)

# 处理未找到的图片名称
if not_found_img_names:
    print("未找到图片数量:", len(not_found_img_names))
    try:
        # 删除旧的未找到图片记录文件（如果存在）
        if os.path.exists(not_find_txt):
            os.remove(not_find_txt)
        # 写入新的未找到图片记录
        with open(not_find_txt, 'w') as f:
            f.write("\n".join(not_found_img_names))
    except IOError as e:
        print(f"写入未找到图片记录文件时出错: {e}")