#SSIM


import os
import cv2
from pytorch_msssim import ssim, ms_ssim
import torch
import torch.nn.functional as F



all_img_path = "./datasets/coco/images/120img"
all_sample_img_path = "./sample_image/seed1024/DPM++2M/text2img"

img_name_list = os.listdir(all_img_path)[:9]  # 均以.jpg结尾
sample_img_name_list = os.listdir(all_sample_img_path)  # 均以.png结尾


def reshape_img(img1, img2):
    # 获取图像的宽度和高度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 确定最短宽度和最短高度
    min_width = min(w1, w2)
    min_height = min(h1, h2)

    # 计算目标宽高
    target_width = min_width
    target_height = min_height

    # 缩放图像
    img1_resized = cv2.resize(img1, (target_width, target_height), interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return img1_resized, img2_resized

def calc_ssim():
    sum_ssim = 0
    for sample_img_name in sample_img_name_list:
        img_path = os.path.join(all_img_path, sample_img_name[:-4] + ".jpg")
        sample_img_path = os.path.join(all_sample_img_path, sample_img_name)
        
    
        
        img1 = cv2.imread(img_path)
        img2 = cv2.imread(sample_img_path)

        if img1.shape != img2.shape:
            img1, img2 = reshape_img(img1, img2)

        img1 = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        img2 = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        ssim_score = ssim(img1, img2)
        sum_ssim += ssim_score
        print(f"SSIM for {sample_img_name}: {ssim_score}")
    print(f"Average SSIM: {sum_ssim/len(sample_img_name_list)}")


def center_crop_and_resize(image_path, target_size=(640, 640)):
    """
    读取图像，裁剪中心区域，并缩放到目标大小。
    
    :param image_path: 图像路径
    :param target_size: 目标大小 (宽度, 高度)
    :return: 缩放后的图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    
    # 获取图像的高度和宽度
    height, width, _ = img.shape
    
    # 计算中心区域的起始和结束坐标
    startx = width // 2 - target_size[0] // 2
    starty = height // 2 - target_size[1] // 2
    
    # 裁剪中心区域
    cropped_img = img[starty:starty + target_size[1], startx:startx + target_size[0]]
    # 缩放到目标大小
    resized_img = cv2.resize(cropped_img, target_size)
    
    return resized_img
# 示例使用
def calc_ms_ssim(multi_compare=False,**kwargs):
    sum_msssim = 0
    print("#######################MS-SSIM#######################")
    for sample_img_name in sample_img_name_list:
        img_path = os.path.join(all_img_path, sample_img_name[:-4] + ".jpg")
        sample_img_path = os.path.join(all_sample_img_path, sample_img_name)
        
        img1 = cv2.imread(img_path)
        img2 = cv2.imread(sample_img_path)

        if img1.shape != img2.shape:
            img1, img2 = reshape_img(img1, img2)



        img1 = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        img2 = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]


        ms_ssim_score = ms_ssim(img1, img2,**kwargs)
        sum_msssim += ms_ssim_score


        if multi_compare:#多图比较
            msssim_score_list=[]#与该图片外的其他所有图片一一比较
            for name in [name for name in sample_img_name_list if name!= sample_img_name]:
                temp_img_path = os.path.join(all_img_path, name[:-4] + ".jpg")
                temp_sample_img_path = os.path.join(all_sample_img_path, name)

                temp_img1=cv2.imread(temp_img_path)
                temp_img2 = cv2.imread(temp_sample_img_path)

        
                if temp_img1.shape != temp_img2.shape:
                    temp_img1, temp_img2 = reshape_img(temp_img1, temp_img2)

                temp_img1 = torch.tensor(temp_img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                temp_img2 = torch.tensor(temp_img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                
                temp_score=ms_ssim(temp_img1, temp_img2,**kwargs)

                msssim_score_list.append(temp_score)





            print(f"MS-SSIM for {sample_img_name}: {ms_ssim_score}",torch.mean(torch.stack(msssim_score_list)))
        else:
            print(f"MS-SSIM for {sample_img_name}: {ms_ssim_score}")
    print(f"Average MS-SSIM: {sum_msssim/len(sample_img_name_list)}")


def calc_psnr(multi_compare=False):
    sum_psnr = 0
    print("#######################PSNR#######################"  )
    for sample_img_name in sample_img_name_list:
        img_path = os.path.join(all_img_path, sample_img_name[:-4] + ".jpg")
        sample_img_path = os.path.join(all_sample_img_path, sample_img_name)

        img1 = cv2.imread(img_path)
        img2 = cv2.imread(sample_img_path)

        if img1.shape!= img2.shape:
            img1, img2 = reshape_img(img1, img2)


        img1 = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        img2 = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

        mse=F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')

        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        sum_psnr += psnr


        if multi_compare:
            psnr_score_list=[]#与该图片外的其他所有图片一一比较
            for name in [name for name in sample_img_name_list if name!= sample_img_name]:
                temp_img_path = os.path.join(all_img_path, name[:-4] + ".jpg")
                temp_sample_img_path = os.path.join(all_sample_img_path, name)

                temp_img1=cv2.imread(temp_img_path)
                temp_img2 = cv2.imread(temp_sample_img_path)


                if temp_img1.shape!= temp_img2.shape:
                    temp_img1, temp_img2 = reshape_img(temp_img1, temp_img2)

                temp_img1 = torch.tensor(temp_img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                temp_img2 = torch.tensor(temp_img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

                temp_mse=F.mse_loss(temp_img1, temp_img2)
                if temp_mse == 0:
                    return float('inf')
                temp_psnr = 20 * torch.log10(255.0 / torch.sqrt(temp_mse))
                psnr_score_list.append(temp_psnr)




            print(f"PSNR for {sample_img_name}: {psnr}",f"MSE:{mse}",psnr_score_list)
        else:
            print(f"PSNR for {sample_img_name}: {psnr}",f"MSE:{mse}")
    print(f"Average PSNR: {sum_psnr/len(sample_img_name_list)}",)




if __name__ == '__main__':
    # dic = {
    #     "win_sigma": 2,
    #     "win_size": 15,  # 增大win_size值
    #     'size_average': True,
    #     'weights': [0.25,0.1, 0.2, 0.35,0.1],

    # }
    calc_ssim()
    calc_ms_ssim()
    calc_psnr()

    #print(ms_ssim.__doc__)