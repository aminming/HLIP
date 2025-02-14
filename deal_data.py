import json
import pandas as pd
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# 加载JSON文件数据
def load_stage_data(files):
    stage_data = {}
    for i, file in enumerate(files, start=1):
        with open(file, 'r') as f:
            stage_json = json.load(f)
            # 删除TCGA-AC-A2FK-01Z-00-DX1.033F3C27-9860-4EF3-9330-37DE5EC45724.svs
            if 'TCGA-AC-A2FK-01Z-00-DX1.033F3C27-9860-4EF3-9330-37DE5EC45724.svs' in stage_json:
                del stage_json['TCGA-AC-A2FK-01Z-00-DX1.033F3C27-9860-4EF3-9330-37DE5EC45724.svs']
            # 重新保存json
            with open(file, 'w') as f:
                json.dump(stage_json, f)
            stage_data[f"stage_{i}"] = [
                case_info["high_risk_ratio"] for case_id, case_info in stage_json.items()
            ]
    return stage_data

# 文件路径
files = [

    r'F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage2\stage_result.json'
]

# 提取high_risk_ratio数据
stage_data = load_stage_data(files)
pass