.. _coordinatesystems:

Coordinate Systems
==================

Cameratransform uses three different coordinate systems.

Image
-----
This 2D coordinate system defines a position on the image in pixel. The position X is between 0 and image_width_px and
the position y is between 0 and image_height_px. (0,0) is the top left corner of the image.

See also :ref:`camprojections`.

Space
-----
This 3D coordinate system defines an euclidean space in which the camera is positioned. Distances are in meter.
The camera is positioned at (pos_x_m, pos_y_m, elevation_m) in this coordinate system.
When heading_deg is 0, the camera faces in the positive y direction of this coordinate system.

See also :ref:`CamOrientation`.

GPS
---
This is a geo-coordinate system in which the camera is positioned. The coordinates are latitude, longitude and elevation.
The camera is positioned at (lat, lon). When heading_deg is 0, the camera faces north in this coordinate system.

This coordinate system shares the parameters elevation_m and the orientation angles (heading_deg, tilt_deg, roll_deg)
with the **Space** coordinate system to keep both coordinate systems aligned.

See also :ref:`gps`.

