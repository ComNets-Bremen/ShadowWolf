Checkboard
==========

- Source: https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf
- From the Mobile Robot Programming Toolkit (MRPT) 
- License: new BSD license, https://docs.mrpt.org/reference/latest/license.html


Calibration
===========

1) Get the camera parameters: https://learnopencv.com/camera-calibration-using-opencv/
2) Remove the distortion: https://learnopencv.com/understanding-lens-distortion/

What is in here?
================

testimages
----------

Some images for several camera models

create_params.py
----------------

Create a set of parameters from given camera images

undistort_image.py
------------------

Remove the distortion from an image with the parameters file created with the
`create_params.py` script.

pickles
-------

A collection of pickles from our cameras.

camera-calibration-checker-board_9x7.pdf
----------------------------------------

The checkboard used for our images
