from ultralytics import YOLO
import os
import torch
import shutil
import gc


def work_once(batch=12,model_path=''):

    if model_path:
        model=model_path
    else:
        model='./runs/detect/001/train138_best.pt'

    task="train"
    if task=='train':
        yolo=YOLO(model=model)
        print('模型加载完成')
        data_yaml=r"ultralytics-main/ultralytics/cfg/my_workspace.yaml"
        yolo.train(
            data=data_yaml,
            imgsz=640,
            epochs=1,
            batch=batch,
            device=0,
            
            # ####禁用数据增强
            hsv_h=0.0, 
            hsv_s=0.0, 
            hsv_v=0.0, 
            degrees=0.0, 
            translate=0.0, 
            scale=0.0,
            shear=0.0, 
            perspective=0.0, 
            flipud=0.0,
            fliplr=0.0, 
            mosaic=0.0, 
            mixup=0.0


        )


def val_once(model='',data_yaml=''):

 
    if model:
        model=model
    else:
        model="./yolov8s.pt"

    if data_yaml:
        data_yaml=data_yaml
    else:
        data_yaml=r"ultralytics-main/ultralytics/cfg/my_workspace.yaml"

    yolo=YOLO(model=model)
    with torch.no_grad():  # 添加梯度上下文管理
        results=yolo.val(data=data_yaml)


def predict_once(model='',img_path='',conf=0.4,show=True,save=True,save_txt=True):

    if model:
        model=model
    else:
        model='./yolov8s.pt'#default_model

    if img_path:
        path=img_path
    else:
        path="./datasets/coco/images/120img"#default_img_path
    

    detect_classes=[1,2,3,5,6,7]#限定检测的类别
    #   1: bicycle
    #   2: car
    #   3: motorcycle
    #   5: bus
    #   6: train
    #   7: truck
    yolo=YOLO(model=model)
    results=yolo.predict(source=path,imgsz=640,classes=detect_classes,conf=conf,show=show,save=save,save_txt=save_txt)    

    

def train_once(model='',data_yaml='',epochs=50+88,resume=False,batch=16):

    if model:
        model=model
    else:
        model="./yolov8s.pt"



    yolo=YOLO(model=model)
    print('模型加载完成')

    if data_yaml:
        data_yaml=data_yaml
    else:
        data_yaml=r"ultralytics-main/ultralytics/cfg/my_workspace.yaml"


    yolo.train(
        data=data_yaml,
        imgsz=640,
        epochs=epochs,
        device=0,
        resume=resume,

        batch=batch,

    
    )

    print('运行完成')


#将最新生成的train文件夹中的best.pt文件复制到001文件夹中，best.pt同名覆盖
def copy_best_model():
    train_save_dir="./runs/detect"
    #所有的train文件都是以train+数字命名的,例如train7
    train_folders = [f for f in os.listdir(train_save_dir) if f.startswith('train') and os.path.isdir(os.path.join(train_save_dir, f))]
    #找到最新的模型文件(即数字最大的文件)
    train_folders.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    if train_folders:
        newest_folder = train_folders[-1]
        best_model_path = os.path.join(train_save_dir, newest_folder, 'weights', 'best.pt')
        target_path = os.path.join(train_save_dir, '001')
        #复制最新的模型文件到001文件夹，同名则覆盖
        shutil.copy(best_model_path, target_path)
        print('复制成功')
    else:
        print('无最新的模型文件')

#按照列表重命名新生成的文件
def rename_file_name(name_list=[],file_path=''):
    if name_list and file_path:
        if os.path.exists(file_path):
            #获取文件名列表,按照从新到旧排序，越新索引越小
            file_list=os.listdir(file_path)
            #筛除非文件夹
            file_list=[f for f in file_list if os.path.isdir(os.path.join(file_path, f))]
            file_list.sort(key=lambda x: os.path.getmtime(os.path.join(file_path, x)),reverse=True)

            #重命名文件夹
            for i,name in enumerate(name_list):
                os.rename(os.path.join(file_path, file_list[i]), os.path.join(file_path, name))
            print('重命名成功')
        else:
            print('文件不存在')
    else:
        print('参数不全')


#找到最新的训练文件夹
def find_newest_train_folder():
    store_path='./runs/detect'
    train_file=os.listdir(store_path)
    #确保是文件夹,而且是train开头
    train_file=[f for f in train_file if os.path.isdir(os.path.join(store_path, f)) and f.startswith('train')]
    #找到最新训练文件夹
    train_file.sort(key=lambda x: os.path.getmtime(os.path.join(store_path, x)),reverse=True)

    
    train_path=os.path.join(store_path,train_file[0])
    return train_path

#模型批量验证所有数据集
def batch_val(Model=''):
        # val_once(model=Model,data_yaml=data_yaml["sd_i2i"])
        # gc.collect()
        # torch.cuda.empty_cache()
        # val_once(model=Model,data_yaml=data_yaml["sd_t2i"])
        # gc.collect()
        # torch.cuda.empty_cache()
        val_once(model=Model,data_yaml=data_yaml["120img"])
        gc.collect()
        torch.cuda.empty_cache()
        val_once(model=Model,data_yaml=data_yaml["fgsm120"])
        gc.collect()
        torch.cuda.empty_cache()
        val_once(model=Model,data_yaml=data_yaml["pgd60+fgsm120"])
        gc.collect()
        torch.cuda.empty_cache()
        val_once(model=Model,data_yaml=data_yaml["sd_i2i_120"])
        gc.collect()
        torch.cuda.empty_cache()

        rename_file_name(name_list=['sd_i2i_120','pgd60+fgsm120','fgsm120','120img'],file_path='./runs/detect/')

#循环训练
def round_train(iteration=3):
    for i in range(iteration):
        if i==0:
            train_once(model=model['default'],data_yaml=data_yaml['120img'],epochs=33)
            gc.collect()
            torch.cuda.empty_cache()
            copy_best_model()
            train_once(model=model['temp'],data_yaml=data_yaml['pgd60+fgsm120'],epochs=33)
            gc.collect()
            torch.cuda.empty_cache()
            copy_best_model()
            train_once(model=model['temp'],data_yaml=data_yaml['sd_i2i_120'],epochs=33)
            gc.collect()
            torch.cuda.empty_cache()
            copy_best_model()
        else:
            train_once(model=model['temp'],data_yaml=data_yaml['120img'],epochs=33)
            gc.collect()
            torch.cuda.empty_cache()
            copy_best_model()
            train_once(model=model['temp'],data_yaml=data_yaml['pgd60+fgsm120'],epochs=33)
            gc.collect()
            torch.cuda.empty_cache()
            copy_best_model()
            train_once(model=model['temp'],data_yaml=data_yaml['sd_i2i_120'],epochs=33)
            gc.collect()
            torch.cuda.empty_cache()
            copy_best_model()


#模型存储路径
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


#数据集配置文件
data_yaml={         
    'default':r"ultralytics-main/ultralytics/cfg/my_workspace.yaml",

    '120img':r"ultralytics-main/ultralytics/cfg/my_120.yaml",       #第一组
    
    "fgsm120":r"ultralytics-main/ultralytics/cfg/my_FGSM120.yaml",  #第二组
    'fgsm120_eps0.15':r"ultralytics-main/ultralytics/cfg/my_FGSM120_eps0.15.yaml",

    'pgd60+fgsm120':r"ultralytics-main/ultralytics/cfg/my_FGSM120+PGD60.yaml",  #第三组

    # "sd_t2i":r"ultralytics-main/ultralytics/cfg/my_SD_text2img.yaml",               #SD(120)+120
    # 'sd_i2i':r"ultralytics-main/ultralytics/cfg/my_SD_img2img.yaml",                #SD(120)+120
    'sd_i2i_120':r"ultralytics-main/ultralytics/cfg/my_SD_img2img_120.yaml",        #SD(120)
    
    'fusion_set':r"ultralytics-main/ultralytics/cfg/my_Fusion Set.yaml"             #120img + pgd60 + fgsm120 + sd_i2i_120
}

#数据集图片存储路径
img_path={
    '120img':'./datasets/coco/images/120img',

    'fgsm120':'./datasets/FGSM120/Reshaped/images',
    
    'gpd60+fgsm120':'./datasets/FGSM120+PGD60/Reshaped/images',

    'sd_i2i':'./datasets/SD_img2img+120img/images',     #SD(120)+120img
    'sd_t2i':'./datasets/SD_text2img+120img/images',    #SD(36)+120img
    'sd_i2i_120':'./datasets/SD_img2img+120img/SD_eps0.25/images',    #SD(120)
    'sd_t2i_36':'./datasets/SD_text2img+120img/SD_text2img/images',   #SD(36)

    'fusion_set':'./datasets/Fusion Set/images'
}



if __name__ == '__main__':


    '''
    #####################
        一键训练
    #####################
    '''
    retrain=False

    if retrain:
        #第一组训练
        train_once(model=model['default'],data_yaml=data_yaml['120img'],epochs=138)
        gc.collect()
        torch.cuda.empty_cache()
        #第二组训练
        train_once(model=model['default'],data_yaml=data_yaml['fgsm120'],epochs=138)
        gc.collect()
        torch.cuda.empty_cache()
        #第三组训练
        train_once(model=model['default'],data_yaml=data_yaml['pgd60+fgsm120'],epochs=188)
        gc.collect()
        torch.cuda.empty_cache()
        # name_list从新到旧排序
        rename_file_name(name_list=['group3','group2','group1'],file_path='./runs/detect/')



    '''
    #####################
        循环训练
    #####################
    '''
    #round_train(iteration=3)
    

    '''
    #####################
        融合训练
    #####################
    '''
    # fusion_train=True
    # if fusion_train:
    #     train_once(model=model['fusion_model'],data_yaml=data_yaml['fusion_set'],epochs=138)
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     copy_best_model()


    '''
    #####################
        一键验证
    #####################
    '''

    # Model=model['temp']
    # batch_val(Model=Model)


    '''
    #####################
        其他功能
    #####################
    '''
    #work_once()
    #train_once(model=model['group1'],data_yaml=data_yaml['sd_i2i'],epochs=188)
    #predict_once(model=model['group1'],img_path='./FGSM_adv_img/eps_0.05',conf=0.4,show=True,save_txt=True)
    #val_once(model=model['group1'],data_yaml=data_yaml['120img'])

        