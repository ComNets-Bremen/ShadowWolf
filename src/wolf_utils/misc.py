import unicodedata
import re

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def batch(iterable, n=4):
    """Return a batch of iterables
    Returns a batch of iterables with a given size
    """
    l = len(iterable)
    for ndx in range(0,l,n):
        yield iterable[ndx:min(ndx + n, l)]

def delta_time_format(seconds):
    """Return a meaningful string for the time left
    Returns a meaningful stream if the number of seconds is given (i.e. 3 minutes and 2 seconds)
    """
    ret = []
    left = seconds
    days = int(left / (60*60*24))
    left = left - (days*60*60*24)
    hours = int(left / (60*60))
    left = left - (hours*60*60)
    minutes = int(left / 60)
    left = left - (minutes*60)

    if days > 1:
        ret.append(f"{days} days")
    elif days > 0:
        ret.append(f"{days} day")

    if hours > 1:
        ret.append(f"{hours} hours")
    elif hours > 0:
        ret.append(f"{hours} hour")

    if minutes > 1:
        ret.append(f"{minutes} minutes")
    elif minutes > 0:
        ret.append(f"{minutes} minute")

    left = round(left)

    if left > 1:
        ret.append(f"{left} seconds")
    elif left > 0:
        ret.append(f"{left} second")

    return " ".join(ret)


