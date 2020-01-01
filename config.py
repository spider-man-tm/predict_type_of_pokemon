import yaml


class Config:
    def __init__(self):
        self.num_epochs = None
        self.seed = None
        self.batch_size = None
        self.device = None
        self.batch_multiplier = None
        self.use_mixup = None
        self.load()

    def load(self):
        with open('./config/config.yml', 'r+') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            self.num_epochs = config.get('num_epochs')
            self.seed = config.get('seed')
            self.batch_size = config.get('batch_size')
            self.device = config.get('device')
            self.batch_multiplier = config.get('batch_multiplier')
            self.use_mixup = config.get('use_mixup')
