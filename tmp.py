# # 读取datasets/en_val.json文件，输出长度
# import json
# datasets_val_json_path = "datasets/en_val.json"
# val_lines = json.load(open(datasets_val_json_path, mode = 'r', encoding = 'utf-8'))
# num_val = len(val_lines)
# print("测试集长度：",num_val)
#
# # 读取datasets/en_train.json文件，输出长度
# datasets_train_json_path = "datasets/en_train.json"
# train_lines = json.load(open(datasets_train_json_path, mode = 'r', encoding = 'utf-8'))
# num_train = len(train_lines)
# print("训练集长度：",num_train)
#
# print("总长度：",num_val+num_train)
# print("测试集占比：",num_val/(num_val+num_train))

# 读取5000_days_stage.csv，以image_path_ls列中"/"分割取倒数第二个，取unique值
# import pandas as pd
# csv = pd.read_csv('5000_days_stage.csv')
# image_path_ls = csv['image_path_ls'].tolist()
# image_path_ls = [i.split('/')[-2] for i in image_path_ls]
# image_path_ls = list(set(image_path_ls))
# print(len(image_path_ls))


# 读取./datasets/en_val.json文件，输出caption关键词长度
import json
datasets_val_json_path = "./datasets/en_val.json"
val_lines = json.load(open(datasets_val_json_path, mode = 'r', encoding = 'utf-8'))
caption_ls = []
max_len = 0
max_caption = None
for i in val_lines:
    max_len = max(max_len,len(i['caption']))
    if max_len == len(i['caption']):
        max_caption = i['caption']
print(max_len)
print(max_caption)
# 把max_caption中按照is分割，取-1组成list
max_captions_split = []
for i in max_caption:
    max_captions_split.append(i.split(' is ')[0])
print(max_captions_split)
