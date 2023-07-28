# Some helpers for the PIL lib

## Is the given image in grayscale?
#
# Checks if the image is grayscale
def isGray(i):
    # An image is gray if all colors are the same. We check this using the
    # histogram.
    r, g, b = i.split()
    if r.histogram() == g.histogram() == b.histogram():
        return True
    return False


## Get all exif tags
#
# returns all exif tags from a given image
def getAllExif(i):
    import PIL.ExifTags
    exif_data_PIL = i._getexif()
    ret = []

    for k, v in PIL.ExifTags.TAGS.items():
        value = exif_data_PIL.get(k, None)

        if value is not None:
            ret.append((PIL.ExifTags.TAGS[k], k, value))
    return ret

