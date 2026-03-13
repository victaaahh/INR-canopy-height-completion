import trackio


class Tracker:
    def __init__(self, project, group, name, config):
        trackio.init(project=project, group=group, name=name, config=config)

    def log_metrics(self, metrics):
        trackio.log(metrics)

    def finish(self):
        trackio.finish()
