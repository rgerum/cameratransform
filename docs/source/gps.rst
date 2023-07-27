.. _gps:

Earth Position (GPS)
====================

.. tip::
    Transforms from the **space** to the **gps** coordinate system.
    Parameters are :math:`\mathrm{lat}, \mathrm{lon}, \alpha_\mathrm{heading}`.

Often the camera is looking at landmarks where the GPS position is known or the GPS position of the camera itself is known
and the GPS position of landmarks has to be determined.

Parameters
----------
- ``lat``, :math:`\mathrm{lat}`: the latitude of the camera position.
- ``lon``, :math:`\mathrm{lon}`: the longitude of the camera position.
- ``heading_deg``, :math:`\alpha_\mathrm{heading}`: the direction in which the camera is looking, also used by the spatial orientation. (0°: the camera faces “north”, 90°: east, 180°: south, 270°: west)

Functions
---------

.. currentmodule:: cameratransform

.. automethod:: Camera.setGPSpos

.. automethod:: Camera.gpsFromSpace
.. automethod:: Camera.spaceFromGPS
.. automethod:: Camera.gpsFromImage
.. automethod:: Camera.imageFromGPS

Transformation
--------------

Distance
~~~~~~~~

.. autofunction:: getDistance

Bearing
~~~~~~~

.. autofunction:: getBearing

Move Distance
~~~~~~~~~~~~~

.. autofunction:: moveDistance

GPS - String Conversion
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: formatGPS

.. autofunction:: gpsFromString

