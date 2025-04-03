文件运行根目录：YOLOv8\\data_precess

json2txt.py功能：
将instances_val2017.json转化为YOLO可识别的txt文件，存放至txt目录内

lable_clean.py功能：
筛选txt文件内的类别，只留下class=
#   1: bicycle
#   2: car
#   3: motorcycle
#   5: bus
#   6: train
#   7: truck
类别，并删除空内容txt文件

fin_image.py功能：
根据剩余的txt文件，从val_img找到图片，移动图片到finded_image文件夹内

txt_info文件夹：
记录YOLO识别类别

val_img文件夹：
存放COCO2017验证集数据