.. _architecture:

Basic architecture
==================

Functions in **bold** are mandatory to define. Everything else *just works*.

- A Project (accessible via the command line or graphic user interface)

  - **Control interface:** :class:`~psychopy_ext.ui.Control`
  
    - Choices of things to do: :class:`~psychopy_ext.ui.Choices`
      - **path to modyle to run**, e.g., 'scripts.main'
      - name of experiment
      - alias to access experiment via command line
      - order of possible tasks (methods) to run
      
    - **Initialization**
    
      - Docstring to describe the experiment for participants
      - **Call parent class:** ``super(MyExp, self).__init__()``
      - Define global parameters, like ``self.stim_size = 3``
      - Redefine computer parameters, usually ``self.computer.valid_responses``
      - Redefine ``self.paths``
      - Define tasks in the experiment, if any: ``self.tasks = []``
      
    - Setup: :func:`~psychopy_ext.exp.Experiment.setup()`          
    
      - Collect run time information: :func:`psychopy.info.RunTimeInfo()`
      - Set random seed
      - Set logging: :func:`~psychopy_ext.exp.Experiment.set_logging()`
      - Create :class:`psychopy.visualWindow`: :func:`~psychopy_ext.exp.Experiment.create_win()`
      
    - Define tasks (inherit from :class:`~psychopy_ext.exp.Task` or define within the experiment class if there is a single task only)
    
      - Initialization
      
        - Docstring to describe the task for participants
        - Call parent class: ``super(MyExp, self).__init__()``
        - Define which column in ``self.exp_plan`` determines splitting into blocks (``self.blockcol``)
        - Define global parameters, like ``self.stim_size = 3``
        - Redefine computer parameters, usually ``self.computer.valid_responses``
        - Define data file name
        
      - Setup: :func:`~psychopy_ext.exp.Task.setup_task()`
      
        - Inherit properties from parent
        - **Create stimuli:** :func:`~psychopy_ext.exp.Task.create_stimuli()`
        
          - Create fixation: :func:`~psychopy_ext.exp.Experiment.create_fixation()`
          - Define your own stimuli
          
        - **Create trial:** :func:`~psychopy_ext.exp.Task.create_trial()`
        
          - Define events: :class:`~psychopy_ext.exp.Experiment.Event`
          
            - Duration
            - Display (which stimulus type is presented)
            - Default function (what to do during the event)
            
              - Do nothing: :func:`~psychopy_ext.exp.Task.idle_event()`
              - Wait until response: :func:`~psychopy_ext.exp.Task.wait_until_response()`
              - Give feedback: :func:`~psychopy_ext.exp.Task.feedback()`
              - Or define your own
              
        - **Create experimental plan:** :func:`~psychopy_ext.exp.Task.create_exp_plan()`
        - Determine if stimulus presentation should be controlled by global timing
        - Set up auto run
        - Split into blocks: :func:`~psychopy_ext.exp.Task.get_blocks()`  
                
    - Run: :func:`~psychopy_ext.exp.Experiment.run()`
    
      - Show instructions: :func:`~psychopy_ext.exp.Experiment.before_exp()`
      - Loop over tasks: :func:`~psychopy_ext.exp.Task.run_task()`
      
        - Show instructions: :func:`~psychopy_ext.exp.Task.before_task()`
        - Loop over blocks: :func:`~psychopy_ext.exp.Task.before_task()`
        
          - Show instructions: :func:`~psychopy_ext.exp.Task.before_block()`
          - Loop over trials: :func:`~psychopy_ext.exp.Task.run_trial()`
          
            - Loop over events: :func:`~psychopy_ext.exp.Task.run_event()`
                                
              - Execute the default function
              - Register responses and their timing (with respect to the onset of the trial)     
                             
            - Record accuracy and response time: :func:`~psychopy_ext.exp.Task.post_trial()`
            
          - Wait between blocks: :func:`~psychopy_ext.exp.Task.after_block()`
          
        - Wait between tasks: :func:`~psychopy_ext.exp.Task.after_task()`
        
      - Show a "thank you" at the end: :func:`~psychopy_ext.exp.Experiment.after_exp()`
      - Push data to a repository
      
  - Analysis
  
    - Get data: :func:`~psychopy_ext.exp.get_behav_df()`    
    - Aggregate data: :func:`~psychopy_ext.stats.aggregate()` or :func:`~psychopy_ext.stats.accuracy()`
    - Plot:
    
      - Initialize plot: :class:`~psychopy_ext.plot.Plot`
      - Plot: :func:`~psychopy_ext.plot.Plot.plot()`
      - Show plot: :func:`~psychopy_ext.plot.Plot.show()`
      
  - Simulation:
  
    - Get images of stimuli
    - Choose a model from :mod:`~pshychopy_ext.models`
    - Run it, e.g., :func:`~psychopy_ext.models.Model.run()`
    
