def run_sampling(run_id: str) -> str:
    from .worker import run_sampling as run

    return run(run_id)

__all__ = ["run_sampling"]
