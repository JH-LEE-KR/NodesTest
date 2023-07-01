import re
import os
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str, default='trinity', help='cluster name')
    args = parser.parse_args()
    return args

def get_node_info():
    node_strings = os.popen('scontrol show nodes').read().strip().split('\n\n')
    nodes = dict()
    for node_string in node_strings:
        name=re.findall(r'NodeName=(\S+)', node_string)[0]
        gres=re.findall(r'Gres=(\S+)', node_string)[0].split(':')[1]
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
                    if args.cluster == 'trinity':
                        partition = 'batch'
                    else:
                        partition = 'batch_grad'
                    f2.write(f'#SBATCH -p {partition}\n')
                elif '#SBATCH -w' in line:
                    f2.write(f'#SBATCH -w {node}\n')
                elif '--nproc_per_node' in line:
                    f2.write(f'        --nproc_per_node={num_gpus} \\\n')
                else:
                    f2.write(line)

def run_code(nodes):
    for node, _ in nodes.items():
        os.popen(f'cd ./code_{node} && sbatch run.sh')

def main():
    args = get_args()
    nodes = get_node_info()
    create_code(nodes, args)
    run_code(nodes)

if __name__ == '__main__':
    main()
    exit(0)