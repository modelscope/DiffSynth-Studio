import os, shutil, multiprocessing, time
NUM_GPUS = 7


def script_is_processed(output_path, script):
    return os.path.exists(os.path.join(output_path, script)) and "log.txt" in os.listdir(os.path.join(output_path, script))


def filter_unprocessed_tasks(script_path):
    tasks = []
    output_path = os.path.join("data", script_path)
    for script in sorted(os.listdir(script_path)):
        if not script.endswith(".sh") and not script.endswith(".py"):
            continue
        if script_is_processed(output_path, script):
            continue
        tasks.append(script)
    return tasks


def run_inference(script_path):
    tasks = filter_unprocessed_tasks(script_path)
    output_path = os.path.join("data", script_path)
    for script in tasks:
        source_path = os.path.join(script_path, script)
        target_path = os.path.join(output_path, script)
        os.makedirs(target_path, exist_ok=True)
        cmd = f"python {source_path} > {target_path}/log.txt 2>&1"
        print(cmd, flush=True)
        os.system(cmd)
        for file_name in os.listdir("./"):
            if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".mp4"):
                shutil.move(file_name, os.path.join(target_path, file_name))


def run_tasks_on_single_GPU(script_path, tasks, gpu_id, num_gpu):
    output_path = os.path.join("data", script_path)
    for script_id, script in enumerate(tasks):
        if script_id % num_gpu != gpu_id:
            continue
        source_path = os.path.join(script_path, script)
        target_path = os.path.join(output_path, script)
        os.makedirs(target_path, exist_ok=True)
        if script.endswith(".sh"):
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} bash {source_path} > {target_path}/log.txt 2>&1"
        elif script.endswith(".py"):
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python {source_path} > {target_path}/log.txt 2>&1"
        print(cmd, flush=True)
        os.system(cmd)


def run_train_multi_GPU(script_path):
    tasks = filter_unprocessed_tasks(script_path)
    output_path = os.path.join("data", script_path)
    for script in tasks:
        source_path = os.path.join(script_path, script)
        target_path = os.path.join(output_path, script)
        os.makedirs(target_path, exist_ok=True)
        cmd = f"bash {source_path} > {target_path}/log.txt 2>&1"
        print(cmd, flush=True)
        os.system(cmd)
        time.sleep(1)
        

def run_train_single_GPU(script_path):
    tasks = filter_unprocessed_tasks(script_path)
    processes = [multiprocessing.Process(target=run_tasks_on_single_GPU, args=(script_path, tasks, i, NUM_GPUS)) for i in range(NUM_GPUS)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def move_files(prefix, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    os.system(f"cp -r {prefix}* {target_folder}")
    os.system(f"rm -rf {prefix}*")


def test_qwen_image():
    run_inference("examples/qwen_image/model_inference")
    run_inference("examples/qwen_image/model_inference_low_vram")
    run_train_multi_GPU("examples/qwen_image/model_training/full")
    run_inference("examples/qwen_image/model_training/validate_full")
    run_train_single_GPU("examples/qwen_image/model_training/lora")
    run_inference("examples/qwen_image/model_training/validate_lora")
    

def test_wan():
    run_train_single_GPU("examples/wanvideo/model_inference")
    move_files("video_", "data/output/model_inference")
    run_train_single_GPU("examples/wanvideo/model_inference_low_vram")
    move_files("video_", "data/output/model_inference_low_vram")
    run_train_multi_GPU("examples/wanvideo/model_training/full")
    run_train_single_GPU("examples/wanvideo/model_training/validate_full")
    move_files("video_", "data/output/validate_full")
    run_train_single_GPU("examples/wanvideo/model_training/lora")
    run_train_single_GPU("examples/wanvideo/model_training/validate_lora")
    move_files("video_", "data/output/validate_lora")


def test_flux():
    run_inference("examples/flux/model_inference")
    run_inference("examples/flux/model_inference_low_vram")
    run_train_multi_GPU("examples/flux/model_training/full")
    run_inference("examples/flux/model_training/validate_full")
    run_train_single_GPU("examples/flux/model_training/lora")
    run_inference("examples/flux/model_training/validate_lora")


def test_z_image():
    run_inference("examples/z_image/model_inference")
    run_inference("examples/z_image/model_inference_low_vram")
    run_train_multi_GPU("examples/z_image/model_training/full")
    run_inference("examples/z_image/model_training/validate_full")
    run_train_single_GPU("examples/z_image/model_training/lora")
    run_inference("examples/z_image/model_training/validate_lora")


if __name__ == "__main__":
    test_z_image()
