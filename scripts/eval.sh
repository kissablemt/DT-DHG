#!/bin/bash

# 初始化变量和数组
CUSTOM_OUTPUT=""
TASKS=4
DATASETS=("Yamanishi" "Shao")
BACKBONES=("GCN" "SAGE" "GAT" "GATv2")
CONFIGS_ABS_DIR=""
INIT_DIMS=(16 32 64 128 256)
INIT_METHODS=("Rand")
NEGATIVE_SAMPLE=2
EPOCHS=5000
LR=0.001
TRAIN_RATIO=0.9
SEED=0
N_LAYERS=(5)
TOP_THRS=(0.05)
DEVICE="cuda:0"

# 获取当前脚本所在的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 项目根目录
ROOT_DIR="${SCRIPT_DIR}/.."

# 解析参数
while [ "$#" -gt 0 ]; do
  case "$1" in
    --top_thrs)
      IFS=',' read -ra TOP_THRS <<< "$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --n_layers)
      IFS=',' read -ra N_LAYERS <<< "$2"
      shift 2
      ;;
    --output)
      CUSTOM_OUTPUT="$2"
      shift 2
      ;;
    --datasets)
      IFS=',' read -ra DATASETS <<< "$2"
      shift 2
      ;;
    --backbones)
      IFS=',' read -ra BACKBONES <<< "$2"
      shift 2
      ;;
    --configs_abs_dir)
      CONFIGS_ABS_DIR="$2"
      shift 2
      ;;
    --init_dims)
      IFS=',' read -ra INIT_DIMS <<< "$2"
      shift 2
      ;;
    --init_methods)
      IFS=',' read -ra INIT_METHODS <<< "$2"
      shift 2
      ;;
    --negative_sample)
      NEGATIVE_SAMPLE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done


# 设置输出路径
# 获取当前时间作为前缀
current_time=$(date +"%m%d-%H%M%S")
OUTPUT="results/eval-${current_time}-${CUSTOM_OUTPUT}"
LOGS_DIR="logs/eval-${current_time}-${CUSTOM_OUTPUT}"

mkdir -p "${LOGS_DIR}"
mkdir -p "${OUTPUT}"


# 在脚本中使用参数
# 将信息同时输出到屏幕和$OUTPUT/params.txt文件
{
echo "Datasets: ${DATASETS[*]}"
echo "Tasks: $TASKS"
echo "Backbones: ${BACKBONES[*]}"
echo "Configs absolute directory: $CONFIGS_ABS_DIR"
echo "Init dimensions: ${INIT_DIMS[*]}"
echo "Init methods: ${INIT_METHODS[*]}"
echo "Top thresholds: ${TOP_THRS[*]}"
echo "N layer: ${N_LAYERS[*]}"
echo "Negative sample: $NEGATIVE_SAMPLE"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Train ratio: $TRAIN_RATIO"
echo "Seed: $SEED"
echo "Output: $OUTPUT"
echo "Logs: $LOGS_DIR"
echo "============================================="
} | tee "$OUTPUT/params.txt"




function run_task {
    name=$1
    yaml_absolute_path=$2
    dataset=$3
    n_features=$4
    sim2feat_flag=$5
    log_filename=$6
    n_layers=$7
    top_thr=$8

    nohup python -u train.py \
            --name ${name} \
            --output_dir ${OUTPUT} \
            --config ${yaml_absolute_path} \
            --dataset ${dataset} \
            --n_features ${n_features} \
            --hidden_dim ${n_features} ${sim2feat_flag} \
            --n_layers ${n_layers} \
            --epochs ${EPOCHS} \
            --lr ${LR} \
            --train_ratio ${TRAIN_RATIO} \
            --device ${DEVICE} \
            --negative_sample ${NEGATIVE_SAMPLE} \
            --top_thr ${top_thr} \
            --seed ${SEED}  > "${log_filename}" 2>&1 &
}


task_count=0
task_pids=()

for top_thr in "${TOP_THRS[@]}"; do
  for n_layers in "${N_LAYERS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for backbone in "${BACKBONES[@]}"; do
        for init_method in "${INIT_METHODS[@]}"; do
          for n_features in "${INIT_DIMS[@]}"; do
            name="${n_layers}_${dataset}_${backbone}_${n_features}_${init_method}_${top_thr}"
            yaml_absolute_path="${CONFIGS_ABS_DIR}/${backbone}.yaml"
            log_filename="${LOGS_DIR}/${name}.log"

            if [[ "${init_method}" == "Bo" ]]; then
              sim2feat_flag="--sim2feat"
            else
              sim2feat_flag=""
            fi
            # echo $sim2feat_flag

            # echo "${name}" "${yaml_absolute_path}" "${dataset}" "${n_features}" "${sim2feat_flag}" "${log_filename}" "${n_layers}" "${top_thr}"
            run_task "${name}" "${yaml_absolute_path}" "${dataset}" "${n_features}" "${sim2feat_flag}" "${log_filename}" "${n_layers}" "${top_thr}"
            echo "tail -f ${log_filename}"

            task_pids+=($!)
            let "task_count += 1"

            if [[ "${task_count}" -eq ${TASKS} ]]; then
              for pid in "${task_pids[@]}"; do
                wait ${pid}
              done
              task_count=0
              task_pids=()
            fi
            
            sleep 5s
          done
        done
      done
    done
  done
done

if [[ "${task_count}" -gt 0 ]]; then
  for pid in "${task_pids[@]}"; do
    wait ${pid}
  done
fi

echo "finish"
# sleep 1000000s