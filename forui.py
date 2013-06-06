
        # https://groups.google.com/forum/?fromgroups=#!topic/wxpython-users/blIdwQ2YV0w
        #from multiprocessing import Process, Queue
        #q = Queue() # create a queue object to talk to the window
        #p = Process(target=processQueue, args=(q,)) # create the window process
        #p.start() # start the window process
        #start_app(q)  # create our app
        #p.join() # wait for the child process to finish

        # Two options: launch the child processes from a
        #thread that is already separate from the wx thread.  Another way would
        #be to create your child processes before creating the wx.App object,
        #perhaps using a multiprocessing.Pool object so you can send them tasks
        #to do later if needed.