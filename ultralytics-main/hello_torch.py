import torch
print(torch.__version__)
print(torch.cuda.is_available())
#检测cudnn版本
print(torch.backends.cudnn.version())