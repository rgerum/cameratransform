Camera
======

.. currentmodule:: cameratransform

.. autoclass:: Camera

Save/Load Functions
-------------------

.. automethod:: Camera.save
.. automethod:: Camera.load
.. autofunction:: load_camera

Transformations
---------------

.. note::
    This section only covers transformations from **image** coordinates to **space** coordinates for **gps** coordinates
    see section :ref:`gps`.

.. automethod:: Camera.imageFromSpace
.. automethod:: Camera.getRay
.. automethod:: Camera.spaceFromImage

Image Transformations
---------------------

.. automethod:: Camera.undistortImage
.. automethod:: Camera.getTopViewOfImage

Helper Functions
----------------

.. automethod:: Camera.distanceToHorizon
.. automethod:: Camera.getImageHorizon
.. automethod:: Camera.getImageBorder
.. automethod:: Camera.getCameraCone
.. automethod:: Camera.getObjectHeight
.. automethod:: Camera.generateLUT