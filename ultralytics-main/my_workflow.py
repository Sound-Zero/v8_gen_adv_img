from my_yolo import work_once,val_once
import shutil
import os
import pynvml
import time
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

all_img_path="./datasets/不重要数据集/80img/images"      #原始图片存放位置
running_path="./datasets/coco/images/val2017"           #模型工作目录
runned_img_path="./datasets/coco/images/trained_img"    #被使用过的原始图片存放位置


adv_img_path="./adv_images"#默认噪声图存储位置，请从trainer.py内同步修改，用于噪声图reshape和原图一致
default_model_path='./runs/detect/001/train138_best.pt'

def generate_adv_img_workflow(img_batch=12):


    print("初始化文件内容")
    if (not os.path.exists(running_path))or (not os.path.exists(runned_img_path)) or (not os.path.exists(all_img_path)):
        print("文件夹不存在，请重试")
    else:
        for filename in os.listdir(running_path):
            souorce_path=os.path.join(running_path,filename)

            target_path=os.path.join(all_img_path,filename)
            shutil.move(souorce_path,target_path)

        for filename in os.listdir(runned_img_path):
            souorce_path=os.path.join(runned_img_path,filename)

            target_path=os.path.join(all_img_path,filename)
            shutil.move(souorce_path,target_path)
        print("初始化完成")

    #1.把120img文件夹内的图片，从头抽取img_batch张图片，剪切粘贴放到running_path文件夹内
    img_name_list=os.listdir(all_img_path)#获取所有图片名称
    img_name_list=[f for f in img_name_list if f.endswith('.jpg')]
    batch_num=len(img_name_list)//img_batch
    if len(img_name_list)%img_batch  !=0:
        batch_num+=1

    for i in range(batch_num):
        #等待GPU至少空出3000MB的显存
        wait_for_gpu_memory(threshold=3000, gpu_id=0, check_interval=10) #等待GPU内存释放


        ####################################################################
        print("######工作流第",i,'/',batch_num,"次开始######")
        img_name_list=os.listdir(all_img_path)#获取所有图片名称
        img_name_list=[f for f in img_name_list if f.endswith('.jpg')]

        if len(img_name_list)>=img_batch:#如果图片数量大于等于img_batch
            img_batch_list=img_name_list[0:img_batch]#获取需要移动的图片名称
            img_batch_list=[f for f in img_batch_list if f.endswith('.jpg')]
        else:
            img_batch_list=img_name_list
            img_batch_list=[f for f in img_batch_list if f.endswith('.jpg')]
            img_batch=len(img_batch_list)

        for img_name in img_batch_list:
            shutil.move(os.path.join(all_img_path,img_name),os.path.join(running_path,img_name))
        print("载入图片完成")

        
        ######################################################################
        

        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        work_once(batch=img_batch,model_path=default_model_path)#高占用显存，需要等待GPU释放后再运行

        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # 再次进行Python垃圾回收
        #等待GPU空出
        wait_for_gpu_memory(threshold=1024, gpu_id=0, check_interval=20) #等待GPU内存释放
        ######################################################################

        runned_img_list=os.listdir(running_path)
        for img_name in runned_img_list:
            shutil.move(os.path.join(running_path,img_name),os.path.join(runned_img_path,img_name))
        print("移除载入完成")
    remove_and_reshape(train_num=batch_num,eps_file='eps_0.25')#eps_0.25为默认噪声图文件名，请根据trainer.py实际情况修改

def remove_and_reshape(train_num=0,eps_file=''):
    ''' 
    从新到旧删除多余train文件，修改噪声图shape,与原图保持一致
    eps_file为默认噪声图文件名，请根据trainer.py实际情况修改
    噪声图路径=adv_img_path+eps_file
    '''


    global adv_img_path,all_img_path
    adv_file=os.path.join(adv_img_path,eps_file)

    print("初始化文件内容")
    if (not os.path.exists(running_path))or (not os.path.exists(runned_img_path)) or (not os.path.exists(all_img_path)):
        print("文件夹不存在，请重试")
    else:
        for filename in os.listdir(running_path):
            souorce_path=os.path.join(running_path,filename)

            target_path=os.path.join(all_img_path,filename)
            shutil.move(souorce_path,target_path)

        for filename in os.listdir(runned_img_path):
            souorce_path=os.path.join(runned_img_path,filename)

            target_path=os.path.join(all_img_path,filename)
            shutil.move(souorce_path,target_path)
        print("初始化完成")

    if train_num:
        train_file_path='./runs/detect'#yolo默认的保存位置
        train_file_list=os.listdir(train_file_path)
        #确保是文件夹
        train_file_list=[f for f in train_file_list if f.startswith('train') and os.path.isdir(os.path.join(train_file_path,f))]
        #按照时间顺序从新到旧排序
        train_file_list.sort(key=lambda x: os.path.getmtime(os.path.join(train_file_path,x)))
        #删除多余的train文件
        if train_num>len(train_file_list):
            print("train_num大于train文件夹数量，请重试")
            return
        else:
            train_file_list=train_file_list[0:train_num]
            print("删除多余train文件夹",train_file_list)
            
            count=0
            try:
                for i in range(train_num):
                    shutil.rmtree(os.path.join(train_file_path,train_file_list[i]))
                    count+=1
            except:
                print("删除train文件夹失败，请重试")
                return
            print("删除多余train文件夹成功",count)
    if os.path.exists(adv_img_path)  and os.path.exists(all_img_path):
        all_img_name_list=os.listdir(all_img_path)
        all_img_name_list=[f[:-4] for f in all_img_name_list if f.endswith('.jpg')]

        target_img_name_list=os.listdir(os.path.join(adv_file))
        target_img_name_list=[f[:-4] for f in target_img_name_list if f.endswith('.jpg')]

        bool_check=True
        for img_name in target_img_name_list:
            if img_name not in all_img_name_list:
                print('噪声图缺少对应原图:',img_name)
        if bool_check:
            count=0
            for img_name in target_img_name_list:
                origin_img_path=os.path.join(all_img_path,img_name+'.jpg')
                origin_img=cv2.imread(origin_img_path)
                
                temp_adv=os.path.join(adv_file,img_name+'.jpg')
                adv_img=cv2.imread(temp_adv)

                adv_height , adv_width =adv_img.shape[:2]
                origin_height,origin_width=origin_img.shape[:2]

                start_y = (adv_height - origin_height) // 2
                start_x = (adv_width - origin_width) // 2
                cropped_fgsm_img = adv_img[start_y:start_y + origin_height, start_x:start_x + origin_width]

                cv2.imwrite(os.path.join(adv_file,img_name+'.jpg'),cropped_fgsm_img)
                count+=1
            print("修改噪声图shape成功，共修改",count,"张图片")


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 ID 获取显存使用信息，单位为 MB
    :param gpu_id: 显卡 ID
    :return: total 总显存，used 已用显存, free 可用显存
    """
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(f"显卡 ID {gpu_id} 对应的显卡不存在！")
        return 0, 0, 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = round(meminfo.total / 1024 ** 2, 2)  # 转换为 MB
    used = round(meminfo.used / 1024 ** 2, 2)  # 转换为 MB
    free = round(meminfo.free / 1024 ** 2, 2)  # 转换为 MB
    pynvml.nvmlShutdown()
    return total, used, free

def wait_for_gpu_memory(threshold=3000, gpu_id=0, check_interval=10):
    """
    等待GPU显存超过指定阈值后再继续运行
    :param threshold: 显存阈值（MB）
    :param gpu_id: 显卡 ID
    :param check_interval: 检查间隔（秒）
    """
    print(f"等待GPU {gpu_id} 的可用显存超过 {threshold} MB...")
    while True:
        _, _, gpu_mem_free = get_gpu_mem_info(gpu_id)
        if gpu_mem_free >= threshold:
            print(f"GPU {gpu_id} 的可用显存已达到 {gpu_mem_free} MB，开始运行程序。")
            break
        time.sleep(check_interval)

    


def val_adv_img_workflow():

    """
    要求：确保running_path文件夹存在且没有图片
    功能：读取adv_img_path路径下的子目录，每个子目录下有若干图片，一一验证所有子目录内的图片

    model={
        "default":"./yolov8s.pt",

        'group1':"./runs/detect/001/train138_best.pt",                #第一组
        
        'group2':"./runs/detect/001/FGSM120_train138_best.pt",        #第二组

        'group3':"./runs/detect/001/PGD60+FGSM120_train188_best.pt",  #第三组

        'temp':"./runs/detect/001/best.pt",   #临时模型

        'round_train':'./runs/detect/001/Round_training_best.pt',   #循环训练三轮
        'fusion_model':'./runs/detect/001/fusion_model.pt',         #融合模型
        'fusion_best138':"./runs/detect/001/fusion_best138.pt",     #融合后训练138Epoch
    }

    """


    
    if not os.path.exists(running_path) :
        print("running_path文件夹不存在，请重试")
        return
    elif os.listdir(running_path):
        print("running_path文件夹内有图片，请先清空running_path文件夹")
        return
    else:
        #展示噪声图片目录下子目录
        files=os.listdir(adv_img_path)
        for i in range(0, len(files), 5):
            print(" ".join(f"{file:<20}" for file in files[i:i+5]))

        
        for f in files:
            print("######开始验证",f,'######')
            img_file_path=os.path.join(adv_img_path,f)
            
            img_name_list=os.listdir(img_file_path)
            for img_name in img_name_list:
                shutil.move(os.path.join(img_file_path,img_name),os.path.join(running_path,img_name))
            print("载入图片完成")

            val_once(model='group1',data_yaml='default')#验证

            img_name_list=os.listdir(running_path)
            for img_name in img_name_list:
                shutil.move(os.path.join(running_path,img_name),os.path.join(img_file_path,img_name))
            print("移除载入")
            





def name_fix():
    ''' 
    功能：修正adv_images文件夹内图片名称
    '''
    
    files=os.listdir(adv_img_path)
    total_count=0
    for f in files:
        count=0
        print("######开始修正",f,'######')
        img_file_path=os.path.join(adv_img_path,f)
        img_name_list=os.listdir(img_file_path)
        for img_name in img_name_list:
            if img_name.endswith('.jpg.jpg'):
                new_img_name=img_name[:-4]
                os.rename(os.path.join(img_file_path,img_name),os.path.join(img_file_path,new_img_name))
                count+=1
        print(f,"修正完成，共修正",count,"张图片")
        total_count+=count
    print("总修正数量：",total_count)







if __name__=="__main__":
    pass
    #name_fix()
    #generate_adv_img_workflow()
    #val_adv_img_workflow()
 
    #remove_and_reshape(train_num=0,eps_file='eps_0.25')