.. _camorientation:

Camera Orientation
==================

.. tip::
    Projections transforms from the **camera** to the **space** coordinate system.
    Parameters are :math:`\alpha_\mathrm{tilt}, \alpha_\mathrm{roll}, \alpha_\mathrm{heading}` and the position :math:`x,y,z`.

Most cameras are not just heading down in z direction but have an orientation in 3D space. Therefore, the module `spatial`
allows to transform from **camera coordinates** (3D, origin at the camera, z=distance from camera) to the **space coordinates**
(3D, camera can have an arbitrary orientation).

Parameters
----------
- ``heading_deg``, :math:`\alpha_\mathrm{heading}`: the direction in which the camera is looking. (0°: the camera faces “north”, 90°: east, 180°: south, 270°: west)
- ``tilt_deg``, :math:`\alpha_\mathrm{tilt}`: the tilt of the camera. (0°: camera faces down, 90°: camera faces parallel to the ground, 180°: camera faces upwards)
- ``roll_deg``, :math:`\alpha_\mathrm{roll}`: the rotation of the image. (0°: camera image is not rotated (landscape format), 90°: camera image is in portrait format, 180°: camera is in upside down landscape format)
- ``pos_x_m``, :math:`x`: the x position of the camera.
- ``pos_y_m``, :math:`y`: the y position of the camera.
- ``elevation_m``, :math:`z`: the z position of the camera, or the elevation above the xy plane.

.. hint::
    When coming from a yaw-pitch-roll system, the pitch is defined differently than the tilt. Tilt 0 means looking straight
    down, while pitch 0 means looking straight ahead. Therefore, tilt_deg = pitch_deg + 90. Also the orientation of the
    roll is typically with an inverted sign: roll_deg = -roll_deg.

Transformation
--------------

.. currentmodule:: cameratransform

.. autoclass:: SpatialOrientation

.. automethod:: SpatialOrientation.cameraFromSpace
.. automethod:: SpatialOrientation.spaceFromCamera

.. autoclass:: SpatialOrientationYawPitchRoll
