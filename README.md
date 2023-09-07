# ShadowWolf
Animals are hiding in the background? This toolchain tries to preprocess those images to make detections work more reliable.

# What can you find here?

## src

The main application. The idea is to write a complete flow which focusses on
detection of animals in the background.

Most computer vision tasks focus on nice, high quality images. The images are
mostly

* taken in good weather conditions.
* having good lightning conditions.
* containing sharp and good visible objects of interest.
* ...

For camera-trap images, this highly differs: Shaky images in bad weather taken
using dirty lenses are increasing the effort for good image recognitions.

**ShadowWolf** tries to deal with those challenges and to detect as many
objects of interests as possible.

## tools

Several tools for analysis etc.

## doc

A fundamental documentation of this work.

# Nice to know
## Database update in August 2023

As the structure has been changed in August 2023, you have to update the
database when re-running the old results with the script. The commands are:

    UPDATE detected_image SET source_datagetter = 'get_images' WHERE source_datagetter = 'getImages'
    UPDATE simple_label_split SET source_getter = 'get_cut_images' WHERE source_getter = 'getCutImages'
    UPDATE simple_label_split SET source_getter = 'get_images' WHERE source_getter = 'getImages'

# Authors

* Jens Dede, Sustainable Communication Networks, University of Bremen, 2023

