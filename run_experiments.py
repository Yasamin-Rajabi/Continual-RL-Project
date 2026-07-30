import subprocess
import argparse
import random
from tasks import tasks

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[
            "simple",
            "componet",
            "finetune",
            "prognet",
            "packnet",
            "cka-rl",
            "masknet",
            "cbpnet",
            "crelus"
        ],
        required=True,
    )
    # Added argument for switching fusion modes
    parser.add_argument("--fusion-mode", type=str, choices=["classic_cka", "weight_delta"], default="classic_cka")
    # Added argument to define the exact sequence of tasks (Defaults to 1 -> 3 -> 5)
    parser.add_argument("--task-sequence", type=int, nargs='+', default=[1, 3, 5], help="List of task IDs to run in sequence")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-run", default=False, action="store_true")
    parser.add_argument("--tag", type=str, default="Debug")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--pool_size", type=int, default=20)
    parser.add_argument("--encoder_from_base", action="store_true")
    return parser.parse_args()


args = parse_args()

# Use the specific task sequence
modes = args.task_sequence

def get_run_name(t_id):
    algo_name = args.algorithm if t_id > 0 or args.algorithm in ['packnet', 'prognet', 'cka-rl', 'masknet', 'cbpnet', 'crelus'] else 'simple'
    return f"task_{t_id}__{algo_name}__run_sac__{args.seed}"

for i, task_id in enumerate(modes):
    params = f"--model-type={args.algorithm} --task-id={task_id} --seed={args.seed} --tag={args.tag}"
    params += f" --fusion-mode={args.fusion_mode}"
    
    if args.debug:
        params += " --total-timesteps=50"
        params += " --learning_starts=5"
    else:
        params += " --total-timesteps=300000"
    if args.encoder_from_base:
        params += " --encoder-from-base"
    else:
        params += " --no-encoder-from-base"
    
    save_dir = f"agents/{args.tag}"
    params += f" --save-dir={save_dir}"
    params += f" --pool_size={args.pool_size}"

    if i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prognet", "cka-rl"]:
            params += " --prev-units"
            for prev_task_id in modes[:i]:
                params += f" {save_dir}/{get_run_name(prev_task_id)}"
        # single previous module
        elif args.algorithm in ["finetune", "packnet", "masknet", "cbpnet", "crelus"]:
            prev_task_id = modes[i-1]
            params += f" --prev-units {save_dir}/{get_run_name(prev_task_id)}"

    # Launch experiment
    cmd = f"python3 run_sac.py {params}"
    print(cmd)

    if not args.no_run:
        res = subprocess.run(cmd.split(" "))
        if res.returncode != 0:
            print(f"*** Process returned code {res.returncode}. Stopping on error.")
            quit(1)