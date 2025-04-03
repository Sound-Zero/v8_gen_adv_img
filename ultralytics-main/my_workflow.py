from my_detect import work_once,val_once
import shutil
import os
import copy
import pynvml
import time

all_img_path="./datasets/coco/images/120img"
running_path="./datasets/coco/images/val2017"
runned_img_path="./datasets/coco/images/trained_img"
img_batch=12

adv_img_path="./adv_images"


def generate_adv_img_workflow():
    #1.把120img文件夹内的图片，从头抽取img_batch张图片，剪切粘贴放到running_path文件夹内
    img_name_list=os.listdir(all_img_path)#获取所有图片名称
    img_name_list=[f for f in img_name_list if f.endswith('.jpg')]

    batch_num=len(img_name_list)//img_batch

    for i in range(batch_num):
        #等待GPU至少空出3000MB的显存
        wait_for_gpu_memory(threshold=3000, gpu_id=0, check_interval=10) #等待GPU内存释放

        if i==0: 
            print("初始化文件内容")
            if (not os.path.exists(running_path))or (not os.path.exists(runned_img_path)) or (not os.path.exists(all_img_path)):
                print("文件夹不存在，请重试")
                break
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


        ####################################################################
        print("######工作流第",i,'/',batch_num,"次开始######")
        img_name_list=os.listdir(all_img_path)#获取所有图片名称
        img_name_list=[f for f in img_name_list if f.endswith('.jpg')]

        img_batch_list=img_name_list[0:img_batch]#获取需要移动的图片名称
        img_batch_list=[f for f in img_batch_list if f.endswith('.jpg')]

        for img_name in img_batch_list:
            shutil.move(os.path.join(all_img_path,img_name),os.path.join(running_path,img_name))
        print("载入图片完成")

        
        ######################################################################
        

        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        work_once(batch=img_batch)#高占用显存，需要等待GPU释放后再运行

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
    # val_once(batch=img_batch)

    #确保running_path文件夹存在且没有图片

    
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

            val_once()#验证

            img_name_list=os.listdir(running_path)
            for img_name in img_name_list:
                shutil.move(os.path.join(running_path,img_name),os.path.join(img_file_path,img_name))
            print("移除载入")
            





def name_fix():#名称修正
    #修正adv_images文件夹内图片名称
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
    #name_fix()
    #generate_adv_img_workflow()
    val_adv_img_workflow()