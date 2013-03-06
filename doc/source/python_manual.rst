======
Manual
======

.. _mne-coreg-info:

MRI-Head Coregistration
=======================

Subjects with MRI
-----------------

In order to perform head mri coregistration in mne-python, the fiducials 
corresponding to the mri need to be specified in a file. If no such file 
exists, it can be created using the :func:`mne.gui.fiducials` gui.

After the fiducials have been saved, the coregistration can be performed using
:func:`mne.gui.coregistration`::

    >>> mne.gui.coregistration('path/to/raw.fif', subject='sample')

In the gui:

#. The position of head and mri are initially aligned using the nasion. The 
   nasion position will be held constant during the fitting procedure. Use the
   "Digitizer Position Adjustment" fields to adjust the nasion position of the 
   digitizer head shape relative to the mri.
#. Press "Fit" to estimate rotation parameters that minimize the distance from
   the digitizer points to the mri. Press "Fit Fiducials" to estimate rotation
   parameters that minimize the distance between the LAPs and RAPs. Manually 
   adjust the movement and rotation parameters   
#. Once a satisfactory coregistration is achieved, hit "Save trans" to save
   the trans file.


Subjects without MRI
--------------------

For subjects for which no structural mri is available, an average brain model 
can be substituted (for example, the *fsaverage* brain that comes with 
Freesurfer_.

To prepare the *fsaverage* brain for use with mne-python, run

    >>> mne.create_default_subject(subject='fsaverage')

Once this is done, the :func:`mne.gui.fit_mri_to_head` gui can be used to 
scale the *fsaverage* brain to better match another subject's head shape. The
gui is launched with::

    >>> mne.gui.fit_mri_to_head('path/to/raw.fif', s_from='fsaverage')

In the gui:

#. The position of head and mri are initially aligned using the nasion. The 
   nasion position will be held constant during the fitting procedure. Use the
   "Digitizer Position Adjustment" fields to adjust the nasion position of the 
   digitizer head shape relative to the mri.
#. In the "N Scaling Parameters" select whether to scale the brain with one 
   parameter (same scaling factor along all axes) or with three parameters
   (separate scaling factor for each axis).
#. Press the "Fit with Scaling" button to perform an automatic fit. The 
   procedure scales the mri and rotates the digitizer head shape around the 
   nasion to minimize the distance from each digitizer point to the mri.
#. Manually adjust the scale and rotation parameters until a satisfying fit is 
   achieved. You can also use "Fit with Scaling" again, using the current 
   values as starting point.
#. Once a satisfactory coregistration is achieved, hit "Save" to save the MRI
   as well as the trans file.


.. _Freesurfer: http://surfer.nmr.mgh.harvard.edu