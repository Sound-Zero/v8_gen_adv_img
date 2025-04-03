import os
import tqdm

print("file_num:",len(os.listdir('txt')))
print("file_path:",os.getcwd())
# 获取当前目录下的所有 txt 文件,只保留以 lables 开头的行
for filename in tqdm.tqdm(os.listdir('txt')):

    if filename.endswith('.txt'):
        # 打开文件进行读写操作
        with open(os.path.join('txt',filename), 'r') as file:
            lines = file.readlines()

        # 过滤掉不是以 1、2、3、5、6、7 开头的行
        lables=['1','2','3','5,6','7']
        filtered_lines=[]
        for line in lines:
            if line.split()[0] in lables:
                filtered_lines.append(line)
            else:
                continue


        with open(os.path.join('txt',filename), 'w') as file:
            file.writelines(filtered_lines)

for filename in tqdm.tqdm(os.listdir('txt')):#删除空文件
    if os.path.getsize(os.path.join('txt',filename))==0:
        os.remove(os.path.join('txt',filename))
print("所有 txt 文件已处理完毕。")