import wandb


class WandbMockup:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def log(self, *args, **kwargs):
        if self.enabled:
            wandb.log(*args, **kwargs)

    def finish(self):
        if self.enabled:
            wandb.finish()

    def init(self, *args, **kwargs):
        if self.enabled:
            wandb.init(*args, **kwargs)

    def Plotly(self, *args, **kwargs):
        if self.enabled:
            return wandb.Plotly(*args, **kwargs)
        else:
            return None
