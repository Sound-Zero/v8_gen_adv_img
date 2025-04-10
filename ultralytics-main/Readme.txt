my-yolo.py用于训练、验证、预测模型。

my-workflow.py用于批量生成噪声图片、使用best模型批量验证的噪声图片。

my-result.py用于从YOLO训练后的train//result.csv绘制result.png

hello-torch.py用于检测pytorch环境。

项目修改部分：
主要修改部分位于main//ultralytics//engine//trainer.py内第382、462、628行自定义部分
配置文件路径main//ultralytics//cfg
配置文件my_FGSM60+PGD60.yaml使用前60张FGSM图片，后60张PGD图片组成数据集。
配置文件my_FGSM120+PGD60.yaml使用120张FGSM图片，后60张PGD图片组成数据集。

注意：
使用my-yolo.py进行训练时，需要根据trainer.py第382行attack_task为空。
使用my-workflow.py用于批量生成噪声图片时，根据trainer.py第382行attack_task分配对应任务。
