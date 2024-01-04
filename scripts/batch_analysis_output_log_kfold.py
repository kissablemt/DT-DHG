import os
import re
from collections import defaultdict

# 定义目录结构
name = "10fold_k1"
root_dir = f"results/{name}"

auc_values = {}
hit10_values = {}

auc_values = defaultdict(list)
hit10_values = defaultdict(list)

def extract_key_name(subdir_name):
    # 提取子目录名称中的有用部分
    key_parts = re.findall(r"(\d+L_[a-zA-Z]+(?:v\d+)?(_ours)?)(?=_" + name + ")", subdir_name)
    return key_parts[0][0] if key_parts else None

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == "output.log":
            # 打开文件并读取内容
            with open(os.path.join(subdir, file), "r") as f:
                file_content = f.read()

                # 使用正则表达式提取包含多个fold的AUC值和Hit@10值
                auc_folds = re.findall(r"AUC (\d+\.\d+)", file_content)
                hit10_folds = re.findall(r"Hit@10 (\d+\.\d+)", file_content)

                if auc_folds and hit10_folds:
                    subdir_name = os.path.basename(subdir)
                    key = extract_key_name(subdir_name)

                    if key is not None:
                        for auc, hit10 in zip(auc_folds, hit10_folds):
                            auc_values[key].append(float(auc))
                            hit10_values[key].append(float(hit10))
                    else:
                        print(f"Invalid key found in {subdir}")
                else:
                    print(f"No AUC or Hit@10 found in {subdir}")

# 计算每个key对应的AUC和Hit@10平均值
auc_avg = {key: sum(auc_list) / len(auc_list) for key, auc_list in auc_values.items()}
hit10_avg = {key: sum(hit10_list) / len(hit10_list) for key, hit10_list in hit10_values.items()}

print("AUC values:", auc_avg)
print("Hit@10 values:", hit10_avg)

# 按目录名称排序
sorted_keys = sorted(auc_values.keys())

print("\n@@@@@@@@@@@@ Markdown @@@@@@@@@@@@\n")

# 生成Markdown表格
header = "| Metric | " + " | ".join(sorted_keys) + " |"
separator = "| --- " + " | --- " * len(sorted_keys) + "|"
auc_row = "| AUC | " + " | ".join([f"{auc_avg[k]:.4f}" for k in sorted_keys]) + " |"
hit10_row = "| Hit@10 | " + " | ".join([f"{hit10_avg[k]:.4f}" for k in sorted_keys]) + " |"


# 输出Markdown表格
print(header)
print(separator)
print(auc_row)
print(hit10_row)

# 生成LaTeX表格
header = r"\begin{tabular}{l" + "c" * len(sorted_keys) + r"}"
toprule = r"\toprule"
header_titles = "Metric & " + " & ".join([f"L{k}" for k in sorted_keys]) + r" \\"
midrule = r"\midrule"
auc_row = "AUC & " + " & ".join([f"{auc_avg[k]:.4f}" for k in sorted_keys]) + r" \\"
hit10_row = "Hit@10 & " + " & ".join([f"{hit10_avg[k]:.4f}" for k in sorted_keys]) + r" \\"
bottomrule = r"\bottomrule"
footer = r"\end{tabular}"

print("\n@@@@@@@@@@@@ LaTeX @@@@@@@@@@@@\n")

# 输出LaTeX表格
print(header)
print(toprule)
print(header_titles)
print(midrule)
print(auc_row)
print(hit10_row)
print(bottomrule)
print(footer)

print("\n@@@@@@@@@@@@ Excel @@@@@@@@@@@@\n")
# 生成逗号分隔的输出
output_lines = ["Model", "AUC", "Hit@10"]
for key in sorted_keys:
    output_lines[0] += f",{key}"
    output_lines[1] += f",{auc_avg[key]:.4f}"
    output_lines[2] += f",{hit10_avg[key]:.4f}"

# 打印制表符分隔的输出
print("\n".join(output_lines))