from diffsynth.diffusion.flow_match import FlowMatchScheduler
import argparse, json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DMD2 student sigmas and timesteps.")
    parser.add_argument("--scheduler_name", type=str, required=True, help="Name of sigma scheduler.")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Number of inference steps.")
    parser.add_argument("--extra_kwargs", type=str, default="{}", help="Extra parameters in set_timesteps, in JSON format.")
    args = parser.parse_args()
    scheduler = FlowMatchScheduler(args.scheduler_name)
    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, **json.loads(args.extra_kwargs))
    print(f'--dmd2_student_sigmas "{", ".join(map(str, scheduler.sigmas.tolist()))}" \\')
    print(f'--dmd2_student_timesteps "{", ".join(map(str, scheduler.timesteps.tolist()))}" \\')
