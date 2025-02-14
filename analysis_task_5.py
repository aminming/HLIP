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
            stage_data[f"stage_{i}"] = [
                case_info["high_risk_ratio"] for case_id, case_info in stage_json.items()
            ]
    return stage_data

# 文件路径
files = [
    r'F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage1\stage_result.json',
    r'F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage2\stage_result.json',
    r'F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage3\stage_result.json',
    r'F:\chenhaobin\clip-pytorch\stagePredict_dataset\stage4\stage_result.json'
]

# 提取high_risk_ratio数据
stage_data = load_stage_data(files)
print(stage_data)
# 转为DataFrame
df = pd.DataFrame({
    "stage": [stage for stage, ratios in stage_data.items() for _ in ratios],
    "high_risk_ratio": [ratio for ratios in stage_data.values() for ratio in ratios]
})
# # 转为DataFrame，行为stage1、2、3、4，列为high_risk_ratio
# df = pd.DataFrame(stage_data)
# # 保存数据
# df.to_csv("F:\chenhaobin\clip-pytorch\stagePredict_dataset/task5_data.csv", index=False)

# 描述性统计
# 添加风险组分类
df['risk_group'] = df['stage'].apply(
    lambda x: 'high_risk' if x in ['stage_3', 'stage_4'] else 'low_risk'
)
summary_stats = df.groupby("stage")["high_risk_ratio"].agg(["mean", "std", "count"]).reset_index()
print("Summary Statistics:")
print(summary_stats)
print()

# 计算高风险和低风险的描述性统计
risk_summary_stats = df.groupby("risk_group")["high_risk_ratio"].agg(["mean", "std", "count"]).reset_index()
# 打印结果
print("Risk Group Summary Statistics:")
print(risk_summary_stats)
print()
import matplotlib.pyplot as plt
import seaborn as sns

# 箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x="stage", y="high_risk_ratio", data=df, palette="Set3")
plt.title("High Risk Ratio Distribution by Stage", fontsize=14)
plt.xlabel("Stage")
plt.ylabel("High Risk Ratio")
plt.show()

# 小提琴图
plt.figure(figsize=(8, 6))
sns.violinplot(x="stage", y="high_risk_ratio", data=df, palette="Set3", inner="point")
plt.title("High Risk Ratio Violin Plot by Stage", fontsize=14)
plt.xlabel("Stage")
plt.ylabel("High Risk Ratio")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


# 绘制箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x="risk_group", y="high_risk_ratio", data=df, palette="Set2")
plt.title("High Risk Ratio Distribution by Risk Group", fontsize=14)
plt.xlabel("Risk Group")
plt.ylabel("High Risk Ratio")
plt.show()
# 绘制小提琴图
plt.figure(figsize=(8, 6))
sns.violinplot(x="risk_group", y="high_risk_ratio", data=df, palette="Set2", inner="point")
plt.title("High Risk Ratio Violin Plot by Risk Group", fontsize=14)
plt.xlabel("Risk Group")
plt.ylabel("High Risk Ratio")
plt.show()


# 假设检验
from scipy.stats import ttest_ind, mannwhitneyu

# 分组
high_risk = df[df["stage"].isin(["stage_3", "stage_4"])]["high_risk_ratio"]
low_risk = df[df["stage"].isin(["stage_1", "stage_2"])]["high_risk_ratio"]

# t检验
t_stat, t_pval = ttest_ind(high_risk, low_risk, equal_var=False)
print(f"T-Test: t-statistic={t_stat:.3f}, p-value={t_pval:.3e}")
print()

# Mann-Whitney U检验
u_stat, u_pval = mannwhitneyu(high_risk, low_risk, alternative="two-sided")
print(f"Mann-Whitney U-Test: U-statistic={u_stat:.3f}, p-value={u_pval:.3e}")
print()
