seisplot
========

A utility for plotting `SEG-Y files <http://www.agilegeoscience.com/blog/2014/3/26/what-is-seg-y.html>`_. 

.. image:: https://img.shields.io/badge/status-alpha-orange.svg
    :target: #
    :alt: Project status

.. image:: https://img.shields.io/github/release/agile-geoscience/seisplot.svg
    :target: #
    :alt: Release

.. image:: https://img.shields.io/badge/python-3.4,_3.5-blue.svg
    :target: #
    :alt: Python version

.. image:: https://img.shields.io/badge/license-Apache_2.0-blue.svg
    :target: http://www.apache.org/licenses/LICENSE-2.0
    :alt: License


Installation
------------

If you don't already have a reliable Python installation, and know how to wield it, I recommend downloading and installing `Anaconda <https://www.continuum.io/downloads>`_.

Get this repo with ``git`` or by downloading the ZIP file, and enter that directory.

Make and enter a virtual environment::

    conda create -n seisplot --file package-list.txt
    source activate seisplot

Install one more dependency::

    conda install -c obspy obspy

Running
-------

Edit ``config.yaml`` to meet your requirements.

Run the script from the command line, for example::

    python seisplot.py </path/to/infile.sgy>
    
This will use ``config.yaml`` make a PNG file in the same location, and with the same basic filename. To use a specific config file with another name or location, and to specify the output filetype — use PDF or SVG for fully scalable vector graphics instead of a raster — add the ``--out`` part::

    python seisplot.py </path/to/infile.segy> --config config.yaml --out /path/to/result.pdf

You can specify a directory and `seisplot` will find files like ``*.sgy``, ``*.SGY``, ``*.segy``, and ``*.SEGY`` in that directory. If you add the recursive flag, ``-R`` to the command line, ``seisplot`` will descend into subdirectories looking for SEG-Y files.

With ``--out`` you can specify an output file and `seisplot` will honour the filetype if the ``matplotlib`` backend you are using supports it. If you specify a directory, all the outout files will go there, using the SEG-Y file's name as the main part of the filename (for example, `31-08.sgy` will give you ``31-08.png`` in the output directory.

As in all things, stains are optional.

Example
-------

.. image:: https://dl.dropboxusercontent.com/u/14965965/31_81_PR.stupid.png

Credits
-------

*Made with love and silliness by Evan and Matt at* `Agile <http://agilegeoscience.com>`_
