import signal
import subprocess
from flask import Flask, render_template, request, jsonify
import glob
import os
import time

app = Flask(__name__)

task_pid = {"PID": ["TASK_NAME", "TIME"]}

@app.route('/generate_command', methods=['POST'])
def generate_command():
    global task_pid
    params = request.form
    cur_time = time.strftime("%m%d-%H%M%S", time.localtime())
    log_file = "logs/" + cur_time + "-" + params['name'] + ".log"

    cmd = "python train.py "
    for key, value in params.items():
        if key in ['sim2feat', 'save_best', 'calc_auc', 'calc_aupr', 'calc_hit_at_10', 'tmp']:
            if value == 'on':
                cmd += f"--{key} "
        elif key == 'kfold':
            if int(value) > 0:
                cmd += f"--{key} {value} "
        elif key == 'train_ratio':
            if 'kfold' not in params or int(params['kfold']) == 0:
                cmd += f"--{key} {value} "
        elif key == "name":
            if value != "":
                cmd += f"--name {value} "
        elif key == "config":
            if value != "":
                cmd += f"--config configs/{value} "            
        else:
            cmd += f"--{key} {value} "
    
    cmd_list = cmd.split()

    with open(os.devnull, "w") as devnull:
        def preexec_function():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        log_file = "logs/" + cur_time + "-" + params['name'] + ".log"
        pid = subprocess.Popen(cmd_list, stderr=open(log_file, 'a'), preexec_fn=preexec_function).pid
        task_pid[str(pid)] = [params['name'], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    print("子进程的PID为：", pid)

    return jsonify(command=cmd)

@app.route('/kill-process', methods=['POST'])
def kill_process():
    global task_pid
    pid = request.form.get('pid')
    
    try:
        os.kill(int(pid), 9)  # SIGKILL
        if pid in task_pid:
            del task_pid[pid]
        return jsonify({'status': 'success', 'message': f'Process {pid} has been terminated.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/nvidia-smi')
def nvidia_smi():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used', '--format=csv'], capture_output=True, text=True)
    result_list = result.stdout.split('\n')[1:-1]
    result_data = []
    for item in result_list:
        result_data.append(dict(zip(['index', 'name', 'temperature', 'gpu_utilization', 'memory_utilization', 'memory_total', 'memory_free', 'memory_used'], item.split(', '))))
    
    process_result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv'], capture_output=True, text=True)
    process_list = process_result.stdout.split('\n')[1:-1]
    process_data = []
    for item in process_list:
        pid, used_gpu_memory = item.split(', ')
        name, date = task_pid.get(pid, ['Unknown', 'Unknown'])
        process_data.append({'pid': pid, 'used_gpu_memory': used_gpu_memory, 'name': name, 'date': date})
    
    return jsonify({'gpu_data': result_data, 'process_data': process_data})

@app.route('/')
def parameter_selection():
    # Get the list of config files
    config_files = [os.path.relpath(f, 'configs') for f in glob.glob('configs/**/*.yaml', recursive=True)]
    config_files.sort()

    # Render the form
    return render_template('parameter_selection.html', config_files=config_files)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
