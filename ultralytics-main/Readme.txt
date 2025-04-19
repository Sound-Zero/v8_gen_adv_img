my-yolo.py用于训练、验证、预测模型。

my-workflow.py用于批量生成噪声图片、使用best模型批量验证的噪声图片。

my-result.py用于从YOLO训练后的train//result.csv绘制result.png

hello-torch.py用于检测pytorch环境。

项目修改部分：
主要修改部分位于main//ultralytics//engine//trainer.py内class BaseTrainer类方法_do_train()三处自定义部分。

配置文件：路径main//ultralytics//cfg

my_120.yaml使用原生120COCO包含车辆的图片组成的数据集

my_FGSM120.yaml使用120张FGSM图片组成的数据集。
my_FGSM60+PGD60.yaml使用前60张FGSM图片，后60张PGD图片组成数据集。
my_FGSM120+PGD60.yaml使用120张FGSM图片，后60张PGD图片组成数据集。

my_SD_img2img.yaml使用120张原图(COCO标签)+由RoLA图生图(人工标签)组成的数据集。
my_SD_text2img.yaml使用120张原图(COCO标签)+由RoLA文生图(人工标签)组成的数据集。

注意：
使用my_yolo.py进行训练时，需要确保trainer.py文件中BaseTrainer类的_do_train()方法内自定义attack_task为空。

使用my_workflow.py用于批量生成噪声图片时，根据trainer.py第382行attack_task分配对应任务。
