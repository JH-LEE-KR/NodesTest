import re
import os
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, default='batch', help='partition name')
    parser.add_argument('--env_path', type=str, default='/data/jaeho/anaconda3/etc/profile.d/conda.sh', help='path of conda environment')
    parser.add_argument('--env_name', type=str, default='torch38gpu', help='name of conda environment')
    args = parser.parse_args()
    return args

def get_node_info():
    node_strings = os.popen('scontrol show nodes').read().strip().split('\n\n')
    nodes = dict()
    for node_string in node_strings:
        name=re.findall(r'NodeName=(\S+)', node_string)[0]
        gres=re.findall(r'Gres=gpu:(\d+)', node_string)[0]
        state=re.findall(r'State=(\S+)', node_string)[0]
        if state != 'IDLE':
            continue
        nodes[name] = gres
    
    return nodes

def create_code(nodes, args):
    for node, num_gpus in nodes.items():
        shutil.copytree('./code', f'./code_{node}')
        with open(f'./template.sh', 'r') as f1, open(f'./code_{node}/run.sh', 'w') as f2:
            for line in f1:
                if '#SBATCH --gres=gpu:' in line:
                    f2.write(f'#SBATCH --gres=gpu:{num_gpus}\n')
                elif '#SBATCH -p' in line:
                    f2.write(f'#SBATCH -p {args.partition}\n')
                elif '#SBATCH -w' in line:
                    f2.write(f'#SBATCH -w {node}\n')
                elif '--nproc_per_node' in line:
                    f2.write(f'        --nproc_per_node={num_gpus} \\\n')
                elif 'source' in line:
                    f2.write(f'source {args.env_path}\n')
                elif 'conda' in line:
                    f2.write(f'conda activate {args.env_name}\n')
                else:
                    f2.write(line)

def run_code(nodes):
    for node, _ in nodes.items():
        os.popen(f'cd ./code_{node} && sbatch run.sh')

def main():
    args = get_args()
    nodes = get_node_info()
    create_code(nodes, args)
    # run_code(nodes)

if __name__ == '__main__':
    main()
    exit(0)