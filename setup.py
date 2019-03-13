from setuptools import setup

setup(name='cameratransform',
      version="0.7",
      description='Projects coordinates from 2D to 3D and can fit camera parameters',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='MIT',
      packages=['cameratransform'],
      install_requires=[
          'numpy',
          'scipy',
          'tqdm'
      ],
      extras_require={
        'projecting_top_view':  ["cv2", "matplotlib"],
        'exif_extraction':  ["pillow", "requests"]
      }
      )
