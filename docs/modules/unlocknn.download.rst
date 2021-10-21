.. currentmodule:: unlocknn.download

Download example files
======================

The ``download`` module contains utilities for downloading pre-trained
models, using :func:`load_pretrained`, and example data, using :func:`load_data`.

.. autofunction:: load_pretrained

.. autofunction:: load_data

Adding new example files
------------------------

New pre-trained models and example data can be added to unlockNN by uploading
the serialized data to the GitHub repository. When doing so, make sure to add
metadata to the appropriate ``README.md`` file (in the ``data`` or ``models``
directories), as well as the :data:`AVAILABLE_MODELS` or :data:`AVAILABLE_DATA`
type variables in :mod:`unlocknn.download`.

For models, simply save the model, then compress the resulting folder to a
``tar.gz`` file format.

For data, unlockNN provides a convenience function for serializing pandas
``DataFrame`` s that contain pymatgen ``Structure`` s in a "structure" column:
:func:`save_struct_data`. The resulting output file can be uploaded to the repository.

.. autofunction:: save_struct_data
