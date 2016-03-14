.. _api:

.. currentmodule:: psychopy_ext


API reference
=============

----------
Experiment
----------
.. automodule:: psychopy_ext.exp

Experiment
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Experiment.__init__
   Experiment.set_logging
   Experiment.create_win
   Experiment.setup
   Experiment.before_exp
   Experiment.run
   Experiment.run_exp
   Experiment.after_exp
   Experiment.autorun
   Experiment.commit
   Experiment.commitpush

Task
~~~~
.. autosummary::
   :toctree: generated/

   Task.__init__
   Task.setup_task
   Task.create_fixation
   Task.create_stimuli
   Task.create_trial
   Task.create_exp_plan
   Task.wait_until_response
   Task.idle_event
   Task.feedback
   Task.wait
   Task.check_continue
   Task.set_autorun
   Task.set_TrialHandler
   Task.show_text
   Task.setup_task
   Task.get_blocks
   Task.before_task
   Task.run_task
   Task.after_task
   Task.before_block
   Task.run_block
   Task.after_block
   Task.run_trial
   Task.run_event
   Task.last_keypress
   Task.post_trial
   Task.get_behav_df
   Task.quit

Helper classes
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Datafile
   Event
   ThickShapeStim
   GroupStim
   MouseRespGroup
   SVG
   OrderedDict

Helper functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   default_computer
   set_paths
   get_behav_df
   combinations
   combinations_with_replacement
   try_makedirs
   signal_det
   invert_dict
   get_mon_sizes
   get_para_no
   get_unique_trials
   latin_square
   make_para
   weighted_choice


--------------------
fMRI analyses (beta)
--------------------
.. automodule:: psychopy_ext.fmri

Preproc
~~~~~~~
.. autosummary::
   :toctree: generated/

   Preproc.__init__
   Preproc.split_rp
   Preproc.gen_stats_batch

Analysis
~~~~~~~~
.. Analysis.plot_chunks
.. Analysis.genFakeData

.. autosummary::
   :toctree: generated/

   Analysis.__init__
   Analysis.run
   Analysis.get_fmri_df
   Analysis.get_behav_df
   Analysis.plot
   Analysis.run_method
   Analysis.get_mri_data
   Analysis.extract_samples
   Analysis.extract_labels
   Analysis.fmri_dataset
   Analysis.detrend
   Analysis.ds2evds
   Analysis.timecourse
   Analysis.signal
   Analysis.univariate
   Analysis.correlation
   Analysis.svm
   Analysis.plot_roi
   Analysis.plot_ds
   Analysis.read_csvs

Helper classes
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   GenHRF

Helper functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   plot_chunks
   plot_timecourse
   make_roi_pattern


------
Models
------

.. automodule:: psychopy_ext.models

.. autosummary::
   :toctree: generated/

Base Model class
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Model.get_teststim
   Model.train
   Model.test
   Model.dissimilarity
   Model._dis_simple
   Model._dis_gj_simple
   Model._dis_fast
   Model.input2array
   Model._prepare_im
   Model.compare

Pixel-wise model
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Pixelwise.run

GaborJet model
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   GaborJet.__init__
   GaborJet.run
   GaborJet.test
   GaborJet.dissimilarity
   GaborJet.compare

HMAX model
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   HMAX.__init__
   HMAX.run
   HMAX.train
   HMAX.test
   HMAX.get_gaussians
   HMAX.get_gaussians_matlab
   HMAX.get_gabors
   HMAX.get_circle
   HMAX.addZeros
   HMAX.get_S1
   HMAX.get_C1
   HMAX.get_S2
   HMAX.get_C2
   HMAX.get_VTU
   HMAX.compare

Caffe
~~~~~
.. autosummary::
   :toctree: generated/

   Caffe.__init__

Helper functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

    get_model

--------------------
Statistical analyses
--------------------
.. automodule:: psychopy_ext.stats

.. autosummary::
   :toctree: generated/

   df_fromdict
   nan_outliers

---------------
Pretty plotting
---------------

.. automodule:: psychopy_ext.plot

--------------
User interface
--------------

.. automodule:: psychopy_ext.ui

.. autosummary::
   :toctree: generated/

   Choices
   Control.__init__
   Control.run_builtin
   Control.cmd
   Control.app
