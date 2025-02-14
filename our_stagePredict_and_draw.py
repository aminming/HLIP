import os
from os.path import abspath, dirname
import sys
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFile
import pandas as pd
from clip import CLIP
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True

# import time
# time.sleep(3600)

# --------------------------------------------------------------------------------------------------
def fun_dataset(dataset_select):
    if dataset_select == 'our_stagePredict':
        captions = ['stage one',
                    'stage two',
                    'stage three',
                    'stage four']

    return captions


# --------------------------------------------------------------------------------------------------
def overlay_mask(original_size, tiles, tile_size, tile_positions, mask_colors, output_size):
    """
    使用掩码颜色重建原始图像，并调整为输出尺寸。

    参数：
        original_size (tuple): 原始图像的尺寸 (宽度, 高度)。
        tiles (list): 已处理的 tiles 列表，每个 tile 为 (Image, 阶段名称)。
        tile_size (tuple): 每个 tile 的尺寸 (宽度, 高度)。
        tile_positions (list): 每个 tile 的位置 (x_start, y_start, x_end, y_end)。
        mask_colors (dict): 阶段名称到掩码颜色的映射。
        output_size (tuple): 输出图像的尺寸 (宽度, 高度)。

    返回：
        Image: 重建并调整尺寸的图像。
    """
    # 初始化重建图像，使用 RGB 模式以节省内存
    reconstructed_image = Image.new('RGB', original_size)

    for tile, position in zip(tiles, tile_positions):
        x_start, y_start, x_end, y_end = position

        # 如果阶段名称在 mask_colors 中，生成掩码
        if tile[1] in mask_colors:
            mask = Image.new('RGBA', tile[0].size, (0, 0, 0, 0))  # 初始化透明掩码
            draw = ImageDraw.Draw(mask)
            draw.rectangle([(0, 0), tile[0].size], fill=mask_colors[tile[1]])  # 填充掩码颜色
            tile_with_mask = Image.alpha_composite(tile[0].convert('RGBA'), mask)  # 应用掩码
        else:
            tile_with_mask = tile[0].convert('RGBA')  # 如果没有掩码颜色，直接转换为 RGBA

        # 将 tile 粘贴到重建图像上
        reconstructed_image.paste(tile_with_mask.convert('RGB'), (x_start, y_start))

    # 判断是否需要调整尺寸
    if original_size != output_size:
        # 如果差距较大，采用分步缩放避免内存问题
        if max(original_size) > max(output_size) * 2:
            temp_size = tuple(dim // 2 for dim in original_size)
            reconstructed_image = reconstructed_image.resize(temp_size, Image.Resampling.BICUBIC)
        reconstructed_image = reconstructed_image.resize(output_size, Image.Resampling.BICUBIC)

    return reconstructed_image


# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    clip = CLIP()

    dataset_select = 'our_stagePredict'
    captions = fun_dataset(dataset_select)
    dataset = '../test_csv/{}.csv'.format(dataset_select)
    csv_original = pd.read_csv(dataset)

    sa = len(csv_original)
    csv = csv_original

    dic_sum = {}
    dic_acc = {}
    for i in list(set(csv['cancer_classification'].tolist())):
        dic_sum[i] = 0
        dic_acc[i] = 0
    num_log = 0

    stage_dirs = ["F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage1",
        "F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage2",
                  "F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage3",
                  "F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage4"]
    stage_dirs = ["F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage2"]
    for stage_dir in stage_dirs:
        stage_result_dict = {}
        image_dir = os.path.join(stage_dir, "till")
        # 只选取文件夹，排除其他csv文件和json文件
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.svs')]
        image_files = [
                       'TCGA-G4-6304-01A-01-TS1.385679cf-f203-42d2-aae5-dc8502783bbd.svs']
        for image_file in image_files:

            num_log += 1

            dir_png = image_file


            predict_result = []
            tile_images = []
            tile_positions = []

            num = 0
            tile_dir = image_dir+'/{}/'.format(dir_png)
            tile_files = os.listdir(tile_dir)


            if  len(tile_files) > 7000 or len(tile_files) < 1000:
                print(f"Skipping {dir_png} due to insufficient tiles")
                continue
            high_low_dict = {"high_risk": 0, "low_risk": 0}
            four_stage_dict = {"stage one": 0, "stage two": 0, "stage three": 0, "stage four": 0}
            slice_num = 0
            for j in tqdm(tile_files, desc=f"Processing tiles in {dir_png}", unit="tile"):
                try:
                    num += 1
                    # if num > 200:
                    #     break
                    # Extract tile coordinates from filename
                    parts = j.split('_')[-1].replace('.png', '').split('-')
                    x_start, y_start, x_end, y_end = map(int, parts)

                    # Load tile image
                    image_path = os.path.join(image_dir, dir_png, j)

                    # image_path = './heat_map_images/{}/{}'.format(dir_png, j)
                    image = Image.open(image_path)

                    # 如果图像中白色像素点占比超过 80%，则跳过
                    if np.mean(np.array(image) > 240) > 0.7:
                        tile_images.append((image, 'stage one'))
                        tile_positions.append((x_start, y_start, x_end, y_end))
                        continue

                    slice_num += 1
                    # Perform prediction
                    probs = clip.detect_image(image, captions)
                    max_prob = max(probs[0])
                    description_match = captions[np.argmax(probs[0])]
                    if max_prob >= 0.6:
                        four_stage_dict[description_match] += 1
                    if description_match in ['stage three', 'stage four'] and max_prob >= 0.6:
                        high_low_dict["high_risk"] += 1
                        predict_result.append(description_match)
                        tile_images.append((image, description_match))
                        tile_positions.append((x_start, y_start, x_end, y_end))
                    else:
                        high_low_dict["low_risk"] += 1
                        predict_result.append('stage one')
                        description_match = 'stage one'
                        tile_images.append((image, description_match))
                        tile_positions.append((x_start, y_start, x_end, y_end))
                except Exception as e:
                    print(f"Error processing tile {j}: {e}")
                    continue
            print(image_file,":",high_low_dict)
            print(image_file,":",four_stage_dict)
            print(image_file,":",high_low_dict["high_risk"]/slice_num)
            stage_result_dict[image_file] = {
                "high_low_dict": high_low_dict,
                "four_stage_dict": four_stage_dict,
                "high_risk_ratio": high_low_dict["high_risk"]/slice_num,
                "slice_num": slice_num,
                "total_num": len(tile_files)
            }


            # Define mask colors for stages
            mask_colors = {
                'stage three': (0, 128, 0, 128),  # Red with transparency
                'stage four': (0, 128, 0, 128)  # Blue with transparency
            }

            # 动态计算原始图片大小
            # 找出偏移量
            min_x = min(pos[0] for pos in tile_positions)  # 最小的 x_start
            min_y = min(pos[1] for pos in tile_positions)  # 最小的 y_start

            # 调整原始图片大小
            max_x = max(pos[2] for pos in tile_positions)  # 最大的 x_end
            max_y = max(pos[3] for pos in tile_positions)  # 最大的 y_end
            original_size = (max_x - min_x, max_y - min_y)  # 原始图片大小（考虑偏移量）

            # 更新 overlay_mask 函数调用中的 tile_positions
            adjusted_tile_positions = [
                (x_start - min_x, y_start - min_y, x_end - min_x, y_end - min_y)
                for x_start, y_start, x_end, y_end in tile_positions
            ]

            # 根据实际需要调整输出图片大小
            output_size = (original_size[0] // 20, original_size[1] // 20)


            reconstructed_image = overlay_mask(original_size, tile_images, (256, 256), adjusted_tile_positions, mask_colors,
                                               output_size)

            output_dir = stage_dir + "/output"

            # output_dir = './output_stagePredict_images'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{dir_png}.png")
            reconstructed_image.save(output_path)
            print(f"Saved image to {output_path}")
            print("-" * 30)
            print()
        print(stage_dir,":")
        avg_high_risk_ratio = sum([stage_result_dict[key]["high_risk_ratio"] for key in stage_result_dict.keys()])/len(stage_result_dict.keys())
        print("avg_high_risk_ratio:",avg_high_risk_ratio)
        print(stage_result_dict)
        # 保存添加到原始文件中
        with open(stage_dir + '/stage_result.json', 'r') as f:
            stage_json = json.load(f)
            for key in stage_result_dict.keys():
                stage_json[key] = stage_result_dict[key]
        print(stage_json)
        with open(stage_dir + '/stage_result.json', 'w') as f:
            json.dump(stage_json, f)
        print("-" * 30)
        print()