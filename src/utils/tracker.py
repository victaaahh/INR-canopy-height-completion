import wandb


class Tracker:
    def __init__(self, project, group, name, config):
        self.run = wandb.init(
            reinit="create_new", project=project, group=group, name=name, config=config
        )

    def log_metrics(self, metrics):
        self.run.log(metrics)

    def log_tables(self, dataframe_dict):
        table_dict = {
            name: wandb.Table(dataframe=df) for name, df in dataframe_dict.items()
        }
        self.run.log(table_dict)

    def log_images(self, image_dict: dict[str, list]):
        image_dict = {
            name: [wandb.Image(image) for image in imagelist]
            for name, imagelist in image_dict.items()
        }
        self.run.log(image_dict)

    def finish(self):
        self.run.finish()
