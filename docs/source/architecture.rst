Basic architecture
==================

    - A Project (accessible via the command line or graphic user interface)
        - Experiment
            - Setup
                - Logging (:func:`~psychopy_ext.exp.Experiment.set_logging()`)
                - Create :class:`psychopy.visualWindow` (:func:`~psychopy_ext.exp.Experiment.create_win()`)                
                - Create stimuli (:func:`~psychopy_ext.exp.Experiment.create_stimuli()`)
                - Create trial (:func:`~psychopy_ext.exp.Experiment.create_trial()`)
                    - Event
                        - Duration
                        - Display (which stimulus type is presented)
                        - Default function (what to do during the event)
                - Create trial list (:func:`~psychopy_ext.exp.Experiment.create_trialList()`)
            - Show instructions
            - Loop over trials
                - Loop over events                    
                    - Execute the default function
                        - update stimuli
                    - Register responses and their timing (with respect to the onset of the trial)                    
                - Record accuracy and response time in the trialList (:func:`~psychopy_ext.exp.Experiment.postTrial()`); can be adjusted if necessary, e.g., calculate accuracy
        - Analysis
