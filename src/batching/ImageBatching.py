import datetime
import PIL.Image
from wolf_utils.ProgressBar import printProgressBar

"""
Split a series of images into data subsets
"""
class ImageBatching:

    def __init__(self, data, config):
        self.data = data
        self.config = config
        print("Running", __class__.__name__, "on", len(self.data), "images")

    def get_batches(self):
        imgdb = []
        print("Building image database")
        printProgressBar(0, len(self.data), prefix = 'Progress:', suffix = 'Complete', length = 50)

        for frame_n, image in enumerate(self.data):
            printProgressBar(frame_n+1, len(self.data), prefix = 'Progress:', suffix = 'Complete', length = 50)
            img = PIL.Image.open(image)
            modifyDate = img.getexif()[306]
            dt_date = datetime.datetime.strptime(modifyDate, "%Y:%m:%d %H:%M:%S")
            imgdb.append({
                "image" : image,
                "date"  : modifyDate,
                "dt"    : dt_date,
                })

        imgdb.sort(key=lambda x:x["dt"])

        imgdb_batches = []
        current_subbatch = []
        lastframe = None
        for img in imgdb:
            if lastframe is None:
                lastframe = img
                current_subbatch.append(img["image"])
                continue
            if (img["dt"] - lastframe["dt"]).total_seconds() > self.config.getfloat("DEFAULT", "Batching_Timediff"):
                imgdb_batches.append(current_subbatch)
                current_subbatch = []
            current_subbatch.append(img["image"])
            lastframe = img
        imgdb_batches.append(current_subbatch)

        print("Database split into", len(imgdb_batches), "batches with a total of", len(self.data))
        return imgdb_batches



