from ultralytics.utils.plotting import plot_results
# import os
# # 设置环境变量以解决OpenMP冲突
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 基本用法
plot_results(file='./结果/120图片训练、验证结果/train50+88+25/results.csv', dir='./结果/120图片训练、验证结果/train50+88+25')

# 扩展用法 - 自定义图表
plot_results(
    file='./结果/120图片训练、验证结果/train50+88/results.csv',  # CSV文件路径
    dir='./结果/120图片训练、验证结果/train50+88',  # 输出目录
    segment=False,  # 是否为分割任务
    pose=False,  # 是否为姿态估计任务

)