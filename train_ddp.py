"""Entry point for multi-GPU DDP training.

Run from the repository root with the project's virtualenv::

    python train_ddp.py --num-fields 5 --num-updates 5 --rollout-steps 32
"""

from coverage_env.rl.trainer.distributed import main

if __name__ == "__main__":
    main()
