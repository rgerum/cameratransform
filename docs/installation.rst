Installation
============

Install with pip
----------------

The simplest way to install the package is using pip:

    ``pip install cameratransform``

Install from the repository
---------------------------

If you want to get a bleeding edge version, you can download the package form our bitbucket homepage.

First of all download the software from bitbucket:

`Download zip package <https://bitbucket.org/fabry_biophysics/cameratransform/get/tip.zip>`_

or clone with `Mercurial <https://www.mercurial-scm.org/>`_:

    ``hg clone https://bitbucket.org/fabry_biophysics/cameratransform``

Then open the installed folder and execute the following command in a command line:

    ``python setup.py install``

.. note::
    If you plan to update regularly, e.g. you have cloned repository, you can instead used ``python setup.py develop``
    that will not copy the the package to the python directory, but will use the files in place. This means that you don't
    have to install the package again if you update the code.
