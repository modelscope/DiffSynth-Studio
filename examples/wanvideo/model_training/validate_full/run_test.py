import multiprocessing, os


def run_task(scripts, thread_id, thread_num):
    for script_id, script in enumerate(scripts):
        if script_id % thread_num == thread_id:
            log_file_name = script.replace("/", "_") + ".txt"
            cmd = f"CUDA_VISIBLE_DEVICES={thread_id} python -u {script} > data/log/{log_file_name} 2>&1"
            os.makedirs("data/log", exist_ok=True)
            print(cmd, flush=True)
            os.system(cmd)
    

if __name__ == "__main__":
    scripts = []
    for file_name in os.listdir("examples/wanvideo/model_training/validate_full"):
        if file_name != "run_test.py":
            scripts.append(os.path.join("examples/wanvideo/model_training/validate_full", file_name))

    processes = [multiprocessing.Process(target=run_task, args=(scripts, i, 8)) for i in range(8)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Done!")