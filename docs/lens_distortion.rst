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

.. currentmodule:: CameraTransform

No Distortion
-------------

.. autoclass:: NoDistortion

Brown Model
-----------

.. autoclass:: BrownLensDistortion


ABC Model
---------

.. autoclass:: ABCDistortion
