import os, shutil, multiprocessing, time


def run_inference(script_path):
    output_path = os.path.join("data", script_path)
    for script in os.listdir(script_path):
        if not script.endswith(".py"):
            continue
        source_path = os.path.join(script_path, script)
        target_path = os.path.join(output_path, script)
        os.makedirs(target_path, exist_ok=True)
        cmd = f"python {source_path} > {target_path}/log.txt 2>&1"
        print(cmd)
        os.system(cmd)
        for file_name in os.listdir("./"):
            if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".mp4"):
                shutil.move(file_name, os.path.join(target_path, file_name))


def run_tasks_on_single_GPU(script_path, gpu_id, num_gpu):
    output_path = os.path.join("data", script_path)
    for script_id, script in enumerate(sorted(os.listdir(script_path))):
        if not script.endswith(".sh") and not script.endswith(".py"):
            continue
        if script_id % num_gpu != gpu_id:
            continue
        source_path = os.path.join(script_path, script)
        target_path = os.path.join(output_path, script)
        os.makedirs(target_path, exist_ok=True)
        if script.endswith(".sh"):
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} bash {source_path} > {target_path}/log.txt 2>&1"
        else:
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {source_path} > {target_path}/log.txt 2>&1"
        print(cmd, flush=True)
        os.system(cmd)


def run_train_multi_GPU(script_path):
    output_path = os.path.join("data", script_path)
    for script in os.listdir(script_path):
        if not script.endswith(".sh"):
            continue
        source_path = os.path.join(script_path, script)
        target_path = os.path.join(output_path, script)
        os.makedirs(target_path, exist_ok=True)
        cmd = f"bash {source_path} > {target_path}/log.txt 2>&1"
        print(cmd)
        os.system(cmd)
        time.sleep(3*60)
        


def run_train_single_GPU(script_path):
    processes = [multiprocessing.Process(target=run_tasks_on_single_GPU, args=(script_path, i, 8)) for i in range(8)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    # run_train_multi_GPU("examples/qwen_image/model_training/full")
    # run_train_single_GPU("examples/qwen_image/model_training/lora")
    # run_inference("examples/qwen_image/model_inference")
    # run_inference("examples/qwen_image/model_inference_low_vram")
    # run_inference("examples/qwen_image/model_training/validate_full")
    # run_inference("examples/qwen_image/model_training/validate_lora")
    run_train_single_GPU("examples/wanvideo/model_inference_low_vram")
