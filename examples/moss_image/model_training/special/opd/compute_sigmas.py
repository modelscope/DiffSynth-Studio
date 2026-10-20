from diffsynth.diffusion.flow_match import FlowMatchScheduler
import argparse, json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script of computing sigmas.")
    parser.add_argument("--scheduler_name", type=str, default="MOSS-Image", help="Name of sigma scheduler.")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps.")
    parser.add_argument("--extra_kwargs", type=str, default="{}", help="Extra parameters in `set_timesteps`. JSON format.")
    args = parser.parse_args()
    scheduler = FlowMatchScheduler(args.scheduler_name)
    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, **json.loads(args.extra_kwargs))
    print("sigmas:", scheduler.sigmas.tolist())
    print("timesteps:", scheduler.timesteps.tolist())
