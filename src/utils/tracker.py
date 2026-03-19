import wandb


class Tracker:
    def __init__(self, project, group, name, config):
        wandb.init(project=project, group=group, name=name, config=config)

    def log_metrics(self, metrics):
        wandb.log(metrics)

    def finish(self):
        wandb.finish()
