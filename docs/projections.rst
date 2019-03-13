Projections
===========

.. tip::
    Projections transforms from the **image** to **camera** coordinate system. Parameters are :math:`f_x, f_y, c_x, c_y` and the image size in px.

This section describes the different projections which are available for the projection of the objects in the **camera
coordinate** system to the **image coordinates**.

In the **camera coordinate** system, the camera is positioned at (0,0,0) and is pointing in :math:`z` direction.

For each projection the projection formula is provided which allows to transform from the **camera coordinate** system (3D)
to the **image coordinate** system (2D). As information is lost from transforming from 3D to 2D, the back transformation
is not unique. All points that are projected on one image pixel lie on one line or ray. Therefore, for the
"backtransformation", only a ray can be provided. To obtain a point this ray has e.g. to be intersected with a plane in
the world.

The coordinates :math:`x_\mathrm{im},y_\mathrm{im}` represent a point in the image pixel coordinates, :math:`x,y,z` the
same point in the camera coordinate system. The center of the image is
:math:`c_x,c_y` and :math:`f_x` and :math:`f_y` are the focal lengths in pixel (focal length in mm divided by
the sensor width/height in mm times the image width/height in pixel), both focal lengths are the same for quadratic pixels on the sensor.

Parameters
----------
- ``focallength_x_px``, :math:`f_x`: the focal length of the camera relative to the width of a pixel on the sensor.
- ``focallength_y_px``, :math:`f_y`: the focal length of the camera relative to the height of a pixel on the sensor.
- ``center_x_px``, :math:`c_x`: the central point  of the image in pixels. Typically about half of the image width in pixels.
- ``center_y_px``, :math:`c_y`: the central point of the image in pixels. Typically about half of the image height in pixels.
- ``image_width_px``, :math:`\mathrm{im}_\mathrm{width}`: the width of the image in pixels.
- ``image_height_px``, :math:`\mathrm{im}_\mathrm{height}`: the height of the image in pixels.

Indirect Parameters
-------------------
- ``focallength_mm``, :math:`f_\mathrm{mm}`: the focal length of the camera in mm.
- ``sensor_width_mm``, :math:`\mathrm{sensor}_\mathrm{width}`: the width of the sensor in mm.
- ``sensor_height_mm``, :math:`\mathrm{sensor}_\mathrm{height}`: the height of the sensor in mm.
- ``view_x_deg``, :math:`\alpha_x`: the field of view in x direction (width) in degree.
- ``view_y_deg``, :math:`\alpha_y`: the field of view in y direction (height) in degree.

Functions
---------

.. currentmodule:: cameratransform

.. autoclass:: CameraProjection

.. automethod:: CameraProjection.imageFromCamera
.. automethod:: CameraProjection.getRay
.. automethod:: CameraProjection.getFieldOfView
.. automethod:: CameraProjection.focallengthFromFOV
.. automethod:: CameraProjection.imageFromFOV

Projections
-----------

All projections share the same interface, as explained above, but implement different image projections.

Rectilinear Projection
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RectilinearProjection

Cylindrical Projection
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CylindricalProjection


Equirectangular Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EquirectangularProjection

