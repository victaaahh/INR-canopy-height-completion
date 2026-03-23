import wandb


class Tracker:
    def __init__(self, project, group, name, config):
        self.run = wandb.init(
            reinit="create_new", project=project, group=group, name=name, config=config
        )

    def log_metrics(self, metrics):
        self.run.log(metrics)

    def finish(self):
        self.run.finish()
