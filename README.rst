seisplot
========

If you don't already have a reliable Python installation, and know how to wield it, I recommend downloading and installing `Anaconda <https://www.continuum.io/downloads>`_.

Get this repo with ``git`` or by downloading the ZIP file, and enter that directory.

Make and enter a virtual environment::

    conda create -n seisplot --file package-list.txt
    source activate seisplot

Install one more dependency::

    conda install -c obspy obspy

Edit ``config.yaml`` to meet your requirements.

Run the script from the command line, for example::

    python seisplot.py </path/to/infile.sgy>
    
This will use ``config.yaml`` make a PNG file in the same location, and with the same basic filename. To use a specific config file with another name or location, and to specify the output filetype — use PDF or SVG for fully scalable vector graphics instead of a raster — add the ``--out`` part::

    python seisplot.py </path/to/infile.segy> --config config.yaml --out /path/to/result.pdf

.. image:: https://dl.dropboxusercontent.com/u/14965965/31_81_PR.png

*Made with love and silliness by* `Agile <http://agilegeoscience.com>`_
