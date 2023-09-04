# Some helpers for the PIL lib

## Is the given image in grayscale?
#
# Checks if the image is grayscale
def is_gray(i):
    # An image is gray if all colors are the same. We check this using the
    # histogram.
    r, g, b = i.split()
    if r.histogram() == g.histogram() == b.histogram():
        return True
    return False


## Get all exif tags
#
# returns all exif tags from a given image
def get_all_exif(i):
    import PIL.ExifTags
    exif_data_PIL = i._getexif()
    ret = []

    for k, v in PIL.ExifTags.TAGS.items():
        value = exif_data_PIL.get(k, None)

        if value is not None:
            ret.append((PIL.ExifTags.TAGS[k], k, value))
    return ret


## Get all iptc tags
#
# return all iptc tags from a given image
def get_all_iptcs(i):
    from PIL import IptcImagePlugin

    iptc = IptcImagePlugin.getiptcinfo(i)
    ret = list()
    if iptc:
        for k, v in iptc.items():
            if isinstance(v, list):
                for val in v:
                    ret.append((":".join(map(str,k)), val.decode()))
            else:
                ret.append((":".join(map(str, k)), v.decode()))
    return ret
