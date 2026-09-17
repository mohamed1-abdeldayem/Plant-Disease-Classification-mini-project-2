import wandb

from plant_disease_mlops import settings


def setup_wandb() -> None:
    if settings.WANDB_KEY:
        wandb.login(key=settings.WANDB_KEY)
    else:
        wandb.login()
