Lens Distortions
================

.. tip::
    Lens distortion transforms from the **distorted** to **image**. Parameters are :math:`k_1, k_2, k_3` or :math:`a, b, c`.

As often the lenses of cameras do not provide a perfect projection on the image plane but introduce some distortions,
applications that work with images need to include the distortions of the lens. The distortions are mostly radial
distortions, but some use also skew and tangential components. CameraTransform does currently only allow for radial
distortion corrections.

To apply the distortion, the coordinates are first centered on the optical axis and scaled using a scale factor, e.g.
the focal length. Then the radial component of the coordinates is stretched or shrunken and the resulting coordinates
are scaled back to pixels and shifted to have 0,0 at the lower left corner of the image. The distortions are always
defined from the flat image to the distorted image. This means an undistortion of the image inverts the formulae.

As CameraTransform can includes the lens correction in the tool chain for projection from the image to the world or the
other way around, there is no need to render an undistorted version of each image the is used.

Brown Model
-----------

**Class**: :py:class:`BrownLensDistortion`

The most common distortion model is the Brown's distortion model. In CameraTransform, we only consider the radial part
of the model, as this covers all common cases and the merit of tangential components is disputed. This model relies on
transforming the radius with even polynomial powers in the coefficients :math:`k_1, k_2, k_3`. This distortion model is
e.g. also used by OpenCV or AgiSoft PhotoScan.

Adjust scale and offset of x and y to be relative to the center:

.. math::
    x' &= \frac{x-c_x}{f_x}\\
    y' &= \frac{y-c_y}{f_y}

Transform the radius from the center with the distortion:

.. math::
    r &= \sqrt{x'^2 + y'^2}\\
    r' &= r \cdot (1 + k_1 \cdot r^2 + k_2 \cdot r^4 + k_3 \cdot r^6)\\
    x_\mathrm{distorted}' &= x' / r \cdot r'\\
    y_\mathrm{distorted}' &= y' / r \cdot r'

Readjust scale and offset to obtain again pixel coordinates:

.. math::
    x_\mathrm{distorted} &= x_\mathrm{distorted}' \cdot f_x + c_x\\
    y_\mathrm{distorted} &= y_\mathrm{distorted}' \cdot f_y + c_y

ABC Model
---------

**Class**: :py:class:`ABCDistortion`

The ABC model is a less common distortion model, that just implements radial distortions. Here the radius is transformed
using a polynomial of 4th order. It is used e.g. in PTGui.

Adjust scale and offset of x and y to be relative to the center:

.. math::
    s &= 0.5 \cdot \mathrm{min}(\mathrm{im}_\mathrm{width}, \mathrm{im}_\mathrm{width})\\
    x' &= \frac{x-c_x}{s}\\
    y' &= \frac{y-c_y}{s}

Transform the radius from the center with the distortion:

.. math::
    r &= \sqrt{x^2 + y^2}\\
    r' &= d \cdot r + c \cdot r^2 + b \cdot r^3 + a \cdot r^4\\
    d &= 1 - a - b - c

Readjust scale and offset to obtain again pixel coordinates:

.. math::
    x_\mathrm{distorted} &= x_\mathrm{distorted}' \cdot s + c_x\\
    y_\mathrm{distorted} &= y_\mathrm{distorted}' \cdot s + c_y

