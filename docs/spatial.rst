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

Transformation
--------------

**Class**: :py:class:`SpatialOrientation`

The orientation can be represented as a matrix multiplication in *projective coordinates*. First, we define rotation
matrices around the three angles: *tilt*, *roll*, *heading*:

.. math::
    R_{\mathrm{tilt}} &=
    \begin{pmatrix}
    1 & 0 & 0\\
    0 & \cos(\alpha_\mathrm{tilt}) & \sin(\alpha_\mathrm{tilt}) \\
    0 & -\sin(\alpha_\mathrm{tilt}) & \cos(\alpha_\mathrm{tilt}) \\
     \end{pmatrix}\\
     R_{\mathrm{roll}} &=
    \begin{pmatrix}
    \cos(\alpha_\mathrm{roll}) & \sin(\alpha_\mathrm{roll}) & 0\\
    -\sin(\alpha_\mathrm{roll}) & \cos(\alpha_\mathrm{roll}) & 0\\
    0 & 0 & 1\\
     \end{pmatrix}\\
     R_{\mathrm{heading}} &=
    \begin{pmatrix}
    \cos(\alpha_\mathrm{heading}) & \sin(\alpha_\mathrm{heading}) & 0\\
    -\sin(\alpha_\mathrm{heading}) & \cos(\alpha_\mathrm{heading}) & 0\\
    0 & 0 & 1\\
     \end{pmatrix}

And the position *x*, *y*, *z* (=elevation):

.. math::
    t &=
    \begin{pmatrix}
    x\\
    y\\
    -\mathrm{elevation}
     \end{pmatrix}

We combine the rotation matrices to a single rotation matrix and apply heading and tilt rotation to the translation vector:

.. math::
    R &=  R_{\mathrm{roll}} \cdot  R_{\mathrm{tilt}} \cdot  R_{\mathrm{heading}}\\
    T &= R_{\mathrm{tilt}} \cdot  R_{\mathrm{heading}} \cdot t\\

Finally, we compose the rotation and the translation to a single matrix in *projective coordinates*.

.. math::
    C_{\mathrm{extr.}} &=  \left(\begin{array}{c|c}
    R & T \\
    \hline
    0 & 1
    \end{array}
    \right)

which is equivalent to:

.. math::
    x_\mathrm{space} = R \cdot x_\mathrm{camera} + T
