from ultralytics import YOLO




def work_once(batch=12):
    # 原调用方式
    import os
    #model="./yolov8s.pt"
    model="./runs/detect/001/best.pt"

    task="train"
    if task=='train':
        yolo=YOLO(model=model)
        print('模型加载完成')
        data_yaml=r"ultralytics-main/ultralytics/cfg/my_val.yaml"
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


def val_once():
    import os
    #model="./yolov8s.pt"
    model="./runs/detect/001/best.pt"

    task="val"
    if task=='val':
        import torch
        yolo=YOLO(model=model)
        with torch.no_grad():  # 添加梯度上下文管理
            results=yolo.val(data=r'ultralytics-main/ultralytics/cfg/my_val.yaml')



def main():
    # 原调用方式
    import os
    model="./runs/detect/001/last.pt"

    task="predict"

    if task=='val':
        import torch
        yolo=YOLO(model=model)
        with torch.no_grad():  # 添加梯度上下文管理
            results=yolo.val(data=r'ultralytics-main/ultralytics/cfg/my_val.yaml')
    elif task=='train':

        yolo=YOLO(model=model)
        print('模型加载完成')
        data_yaml=r"ultralytics-main/ultralytics/cfg/my_val.yaml"
        yolo.train(
            data=data_yaml,
            imgsz=640,
            epochs=25,
            device=0


        )
    elif task=='predict':
        current_path=os.getcwd()
        my_img_path=os.path.join(current_path,"./")
        print(my_img_path)
        detect_classes=[1,2,3,5,6,7]#限定检测的类别

        yolo=YOLO(model=model)
        results=yolo.predict(source=my_img_path,imgsz=640,classes=detect_classes,conf=0.2,show=True,save=True,save_txt=True)    
    
    
    
    print('运行完成')





if __name__ == '__main__':  # 新增多进程保护
    #val_once()
    #work_once()
    main()
