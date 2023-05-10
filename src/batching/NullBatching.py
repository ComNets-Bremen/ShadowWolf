from wolf_utils.ProgressBar import printProgressBar

"""
Does nothing, just return one batch with all images
"""
class NullBatching:

    def __init__(self, data, config):
        self.data = data
        self.config = config
        print("Running", __class__.__name__, "on", len(self.data), "images")

    def get_batches(self):
        return [self.data,]



