import os
import re

# 定义目录结构
root_dir = "results/v5_rank"

# 存储AUC和Hit@10值
auc_values = {}
hit10_values = {}

# 遍历目录结构
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == "output.log":
            # 打开文件并读取内容
            with open(os.path.join(subdir, file), "r") as f:
                file_content = f.read()

                # 使用正则表达式提取AUC值和Hit@10值
                auc = re.search(r"AUC (\d+\.\d+)", file_content)
                hit10 = re.search(r"Hit@10 (\d+\.\d+)", file_content)
                if auc and hit10:
                    key = int(re.search(r"L(\d+)", subdir).group(1))
                    auc_values[key] = auc.group(1)
                    hit10_values[key] = hit10.group(1)
                else:
                    print(f"No AUC or Hit@10 found in {subdir}")

print(auc_values)
print(hit10_values)

# 按目录名称排序
sorted_keys = sorted(auc_values.keys())

print("\n@@@@@@@@@@@@ Markdown @@@@@@@@@@@@\n")

# 生成Markdown表格
header = "| Metric | " + " | ".join([f"L{k}" for k in sorted_keys]) + " |"
separator = "| --- " + " | --- " * len(sorted_keys) + "|"
auc_row = "| AUC | " + " | ".join([auc_values[k] for k in sorted_keys]) + " |"
hit10_row = "| Hit@10 | " + " | ".join([hit10_values[k] for k in sorted_keys]) + " |"

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
auc_row = "AUC & " + " & ".join([auc_values[k] for k in sorted_keys]) + r" \\"
hit10_row = "Hit@10 & " + " & ".join([hit10_values[k] for k in sorted_keys]) + r" \\"
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
# 生成制表符分隔的输出
output_lines = ["AUC", "Hit@10"]
for key in sorted_keys:
    output_lines[0] += f",{auc_values[key]}"
    output_lines[1] += f",{hit10_values[key]}"

output = "\n".join(output_lines)
print(output)