from setuptools import setup

setup(name='CameraTransform',
      version="0.7",
      description='Projects coordinates from 2D to 3D and can fit camera parameters',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='MIT',
      packages=['CameraTransform'],
      install_requires=[
          'numpy',
          'scipy'
      ],
      extras_require={
        'projecting_top_view':  ["cv2", "matplotlib"],
        'exif_extraction':  ["pillow", "requests"]
      }
      )
