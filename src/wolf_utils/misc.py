import importlib
import unicodedata
import re

import cv2

import logging

logger = logging.getLogger(__file__)


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
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def getter_factory(dataclass, getter, *args, **kwargs):
    """Return a getter from a given dataclass

    Create an instance of a dataclass and return the corresponding getter.
    """
    logging.info(f"Getting \"{getter}\" for \"{dataclass}\"")
    module_path, class_name = dataclass.rsplit('.', 1)
    module = importlib.import_module(module_path, class_name)
    importlib.invalidate_caches()
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return getattr(instance, getter)


def delta_time_format(seconds):
    """Return a meaningful string for the time left
    Returns a meaningful stream if the number of seconds is given (i.e. 3 minutes and 2 seconds)
    """
    ret = []
    left = seconds
    days = int(left / (60 * 60 * 24))
    left = left - (days * 60 * 60 * 24)
    hours = int(left / (60 * 60))
    left = left - (hours * 60 * 60)
    minutes = int(left / 60)
    left = left - (minutes * 60)

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


def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=1,
          font_thickness=1,
          text_color=(0, 0, 0),
          text_color_bg=(255, 255, 255)
          ):
    """
    Draws a text with a background to a given position

    Parameters
    ----------
    img             The image to paint on
    text            The text
    pos             Position of the text
    font            The font (defaults to FONT_HERSHEY_PLAIN)
    font_scale      The scale of the font. Defaults to 1
    font_thickness  The thickness of the font. Defaults to 1
    text_color      The text color. Default: Black (0, 0, 0)
    text_color_bg   The background color. Default: White (255, 255. 255)

    Returns         The text size
    -------
    """

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size
