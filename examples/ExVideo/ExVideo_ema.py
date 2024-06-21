import torch, os, argparse
from safetensors.torch import save_file


def load_pl_state_dict(file_path):
    print(f"loading {file_path}")
    state_dict = torch.load(file_path, map_location="cpu")
    trainable_param_names = set(state_dict["trainable_param_names"])
    if "module" in state_dict:
        state_dict = state_dict["module"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict_ = {}
    for name, param in state_dict.items():
        if name.startswith("_forward_module."):
            name = name[len("_forward_module."):]
        if name.startswith("unet."):
            name = name[len("unet."):]
        if name in trainable_param_names:
            state_dict_[name] = param
    return state_dict_


def ckpt_to_epochs(ckpt_name):
    return int(ckpt_name.split("=")[1].split("-")[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Gamma in EMA.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # args
    args = parse_args() 
    folder = args.output_path
    gamma = args.gamma

    # EMA
    ckpt_list = sorted([(ckpt_to_epochs(ckpt_name), ckpt_name) for ckpt_name in os.listdir(folder) if os.path.isdir(f"{folder}/{ckpt_name}")])
    state_dict_ema = None
    for epochs, ckpt_name in ckpt_list:
        state_dict = load_pl_state_dict(f"{folder}/{ckpt_name}/checkpoint/mp_rank_00_model_states.pt")
        if state_dict_ema is None:
            state_dict_ema = {name: param.float() for name, param in state_dict.items()}
        else:
            for name, param in state_dict.items():
                state_dict_ema[name] = state_dict_ema[name] * gamma + param.float() * (1 - gamma)
        save_path = ckpt_name.replace(".ckpt", "-ema.safetensors")
        print(f"save to {folder}/{save_path}")
        save_file(state_dict_ema, f"{folder}/{save_path}")
