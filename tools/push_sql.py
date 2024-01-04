import json
import os
from datetime import datetime
from pathlib import Path
import re
import pymongo

# 获取当前目录及其子目录下的所有文件夹
folders_list = list(Path("results/").glob("**/"))

# 获取所有结果的根目录
res_root_list = []
for f in folders_list:
    if (f / "output.log").exists() and (f / "args.json").exists():
        res_root_list.append(f)
        
# 定义正则表达式
pattern = re.compile(r'.* - INFO - (?P<metric>[\w@]+) (?P<value>[\d\.]+)')

data = []
for root in res_root_list:
    args_file = root / "args.json"
    log_file = root / "output.log"
    
    try:
        with open(args_file, "r") as f:
            args = json.load(f)
        res = dict(
            config=args["config"].split("/")[-2],
            dataset=args["dataset"],
            backbone=args["config"].split("/")[-1].replace(".yaml", ""),
            top_thr=args["top_thr"],
            n_layers=args["n_layers"],
            n_features=args["n_features"],
            init_method="Bo" if args["not_rand_feat"] else "Rand",
            negative_sample=args["negative_sample"],
            date=datetime.fromtimestamp(os.stat(args_file).st_ctime),
            path=str(root),
        )

        # 解析 output.log 文件
        with open(log_file, 'r') as f:
            content = f.read()
            # 使用正则表达式匹配指标和数值
            for match in pattern.finditer(content):
                metric = match.group('metric')
                value = float(match.group('value'))
                res[metric] = value
        
        data.append(res)
    except Exception as e:
        pass
print("Total: ", len(data))

# 连接MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")

# 获取数据库和集合对象
db = client["RankBern"]
col = db["results"]

# Drop the collection
col.drop()

# Insert data
result = col.insert_many(data)

# Check if all documents were inserted successfully
if result.acknowledged:
    print("All documents were inserted successfully.")
else:
    print("Some documents were not inserted.")

