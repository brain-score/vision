.. _Tutorial:

========
Tutorial
========

About
=====
Brain-Score is a novel way to measure how “brain-like" artificial neural networks are,
from recurrent NNs to Spiking NNs and everything in between. It was developed by my colleagues
Martin Schrimpf and James DiCarlo, amongst others, at MIT, and I (Mike) was recently hired in
June to maintain and add features to the code base. In a nutshell, Brain-Score attempts to map
how similar models are to the brain, and gives a score (usually ranging from 0 to 1) on various
brain regions and behavioral benchmarks. This guide is a quickstart for researchers and tinkerers
alike that tries to outline the setup, submission, and feedback system for users.

Overview
========

Brain-Score sounds great! How do I get started? What do I do next? How do I submit?
--------------------------------------------------------------------

We will give you guidance below to try and walk you
through the entire process, end-to-end. Start here if you have already
(or would like to create) a model to submit and get a Brain-Score.
This guide will walk you through from scratch to receiving a
submission score from the website. Completion of the tutorial
should take about an hour, more or less.

What does Brain-Score look like, exactly?
--------------------------------------------------------------------
The main code for Brain-Score is contained and hosted on Github (https://github.com/brain-score)
on various repos, free for anyone to fork,
clone, or download. The main page has 10 repos (9 visible to non-devs)
that make up the entire source code base- this can be overwhelming, but
fear not. I will walk you through what each does, as well as explain that
you most likely will only need a few to get a model up and scored. It is
good to see, however, the structure of how Brain-Score operates with its
various repos. They are:

1. ``brain-score``: the heart of the code that runs analysis and comparisons.
2. ``sample-model-submission``: template and examples for model submissions.
3. ``candidate_models``: various pre-trained models/models that you can submit.
4. ``model-tools``: helper functions that translate from machine learning models
   to brain models to be tested on brain-score.
5. ``brainio_collection``: the repo that packages and collects the stimuli/data.
6. ``brainio_base``: repo that contains various data structures for BrainIO.
7. ``result_caching``: a helper repo to store the results of function calls so they can
   be re-used without re-computing.
8. ``brain-score.web``: website front and back end.
9. ``Brainio_contrib`` (archived): used in the past to contribute stimuli and datasets
   (now part of ``brainio_collection``).

Which repos will I use?
-----------------------
When we get to the install guide, I will show you exactly how to
clone/fork repos for your own project in the easiest way possible.
But for now, you will mainly only need the ``sample-model-submission`` repo.

How do I get a Brain-Score for my model?
----------------------------------------

Brain-Score currently has two “ways" to score your model:

1. *On your machine*:  your model is scored on one or more benchmarks (a benchmark is a
   standard to compare against, like Majaj-Hong’s 2015 paper on IT results (which you
   can read more about here: https://brain-score.readthedocs.io/en/latest/modules/benchmarks.html). This is useful if you would like to test and see if your
   model is ready to be submitted and run against all benchmarks, or if you want a quick
   and dirty way to score on a single benchmark, like V4 or IT. This is mainly used to
   test and/or optimize your model before submitting to the main site.
2. *Remote*: your model is run on Brain-Score’s website, and contains scores against all the
   local benchmarks, plus many others (we avoid sharing this code with users to
   avoid overfitting). The (hierarchical) mean of all these benchmark scores is the
   Brain-Score itself. This has more overhead to get started, but is worth it, as you
   can set up pull requests to our main code package (i.e, suggest changes) if you find
   errors and have the packages/dependencies automatically installed.

Why do you recommend installing and submitting the way outlined in this guide? In other words, why should I do it your way?
------------------------------------------------------------------------------

A reasonable question, and it is always good to be skeptical. The short answer
is that using an IDE like Pycharm and virtual environments
drastically cuts the error rate for install down, as well as makes the whole
process of installing dependencies easier. Using a VENV also helps with headaches
caused by clashes between Anaconda and PIP, and Pycharm
(or another IDE like VScode) takes care of that for you.

Do I have to read/do this entire tutorial to submit a model?
------------------------------------------------------------

No - You can skip to the quickstart guide below, if you do not
wish to read/do this entire tutorial, but we recommend it in
order to get the most out of brain-score and avoid errors that might pop up along the way.



Quickstart
==========
In this section, we will provide a quick and easy way to get Brain-Score
up and running. This is mainly for those who do not have the time to read
or do the whole tutorial, or for those who just want to go ahead and submit
a model quickly; however, we do officially recommend that you read and do the
entire tutorial, as it will help you in the future. This section also does not
have pictures, which the other more lengthy sections below do. In this case,
we will submit a version of AlexNet from Pytorch’s library; the main steps are outlined below:

1. Get a copy of  the ``sample-model-submission`` repo from our Github. If you are wondering
   about what the various repos on the main Brain-Score github do, check out the Overview section above.
   Here is the command for a terminal ::
     git clone https://github.com/brain-score/sample-model-submission.git
2. Install the dependencies via pip. (Make sure to switch into the ``sample-model-submission`` folder
   that was created in step one above when you cloned the repo). You can read more about this in the
   Install Brain-Score Repos and Dependencies section below. Again, the command for the terminal is ::
     pip install .
3. Specify the model to test. You can do this a few ways, but in this case, we can
   just use the model defined in ``pytorch.py``. More info on this step can be found in
   the Submitting a Model to Brain-Score.org Part 1: Preparing the Model section of
   this guide. In essence, you need to implement the various functions outlined in
   the ``sample-model-submission/models/base_models.py`` file.
4. Test the model on your machine. You can do this simply by hitting “run"
   on the ``pytorch.py`` file (in this case) or the file where you implemented
   the ``base_models.py`` template. If you followed the steps correctly, you should
   receive a message on the Python console indicating that you are ready to submit.
5. Submit the package to brain-score.org. This step is slightly brittle as of now,
   and is easy to do wrong; I recommend skipping down to ``Submitting a Model to
   Brain-Score.org Part 2: Upload`` section to see the structure of the zip file that
   our site needs in order to process your submission.
6. That’s it! Read more below to get a better idea of the process, or to help fix bugs that might come up.



Install Brain-Score Repos and Dependencies
==========================================
In this section, I will show you how to get packages installed and dependencies
linked in order to run setup for submission and scoring.

1. Download PyCharm (https://www.jetbrains.com/pycharm/download/#section=windows) or another IDE.
   *Note: you do not have to use Pycharm per se, but we recommend it, and this guide will show*
   *you how to integrate Brain-Score with it. It is the easier way.*
   If you do not have experience with Pycharm, here’s a nice tutorial: https://www.tutorialspoint.com/pycharm/index.htm.
   Again, we recommend and like Pycharm, but this tutorial is neutral in the sense that you can use
   any IDE, as the steps are very similar for other environments, but this document will
   feature Pycharm screenshots.
2. Once Pycharm (or your own IDE) is set up, we will start the install of Brain-Score
   and its various repos. First, in your file explorer, make a new file on your desktop
   or favorite place to save things. I personally made a folder called ``brainscore-brief``
   in my ``/desktop`` folder. Create a new project, and your IDE should ask you for a location
   to create said project. I recommend setting up the path to be the newly created folder
   from above, in my case the path is ::
     /Users/mike/desktop/brainscore-brief
   Your IDE will create a Python interpreter for the project (the piece of code that
   tells the computer how to run various Python commands) by setting up a Virtual Environment
   for you automatically. A venv is handy because installing the dependencies that Brain-Score
   needs will not conflict with other packages on your computer if you use a venv.
   To the left on your screen, you will see your folder brainscore-brief that is the
   project root. If you click to expand it, then you will see an orange folder marked ``venv``
   that contains all the venv files and whatnot. I would not mess with the ``venv`` folder or
   download anything in there. Again, your IDE will most likely be different if you do not use
   Pycharm, but the main points still hold.
3. Next, we are going to clone the repos we need in order to get the code from Github for Brain-Score.
   The easiest way to do this is to install Git on your computer from: https://git-scm.com/download/win.
   Once this is installed, open up your terminal and navigate into the ``brainscore-brief``
   folder. In my case, the commands are ::
     cd desktop -> cd brainscore-brief

   After you are in this folder,
   run::
     git clone https://github.com/brain-score/sample-model-submission.git
   This will copy our sample-model-submission code from Github into your local machine to run later on.
   Switching back to your IDE’s file explorer, you should now see a folder called ``sample-model-submission``
   in your project folder. Clicking on/expanding this will show you the various files and
   programs that are in our collection for the ``sample-model-submission`` repo.
   You can see the various folders in the image below: the top level ``brainscore-brief``
   is the folder that we created a few steps ago. The next level ``sample-model-submission``
   is the repo cloned from our Github. You should now see something akin to below when you
   look at your version on your machine:

    .. image:: /docs/source/modules/tutorial_screenshots/sms.png
       :width: 600

4. We will now install the pip packages that our code needs to run: things like ``scipy`` and
   ``imageio`` , etc. In your IDE or using your main computer terminal, switch into your root
   directory, in this case ``brainscore-brief``. Navigate into the repo directory,
   ``sample-model-submission``, using the command ::
     cd sample-model-submission
   (which should be one level down from the original created folder/directory).
   Once you are in this ``sample-model-submission`` repo,
   run the command below  (note the period; this tells pip to install all the dependencies you will
   need: a nice and handy way to do this) ::
     pip install .
   In Pycharm, you can check to make sure these dependencies were installed correctly
   by going into ::
     Pycharm -> settings (preferences on Mac) -> project: brainscore-brief -> project interpreter
   where you will see a list of around 100 packages like ``toml``, ``xarray``, and
   ``Keras-preprocessing``. (Note: installing all the dependencies will take around 2-5 mins
   on your machine, depending on the hardware/internet). A different IDE will most likely
   have a similar feature, but this tutorial gives Pycharm as an example.
5. Congrats! You now have completed the hardest part of install.
   Also remember before running the pip command, make sure to navigate
   using terminal into the correct folder using the ::
     cd sample-model-submission
   command to ensure it is installed in the right place- otherwise you get error #1
   in the ``Common Errors: Setup`` section. Feel free to explore the various
   files and get a feel for them.

That’s it! You have downloaded and retrieved all of the files you need to submit a model!
Take a break and go get some lunch or some donuts. If you get an error that is not
listed/resolved below, reach out to us at MIT and we can (most likely) help:

- msch@mit.edu
- mferg@mit.edu
- cshay@mit.edu

Submit a Model to Brain-Score.org Part 1: Preparing the Model
=============================================================

By now you should have the ``sample-model-submission`` repo cloned and
the dependencies installed. It is now time to prepare your model to be
submitted! In this part we will submit a standard, generic form of AlexNet
(implemented in Pytorch) in order to get a feel for the submission process.
In Part 3 I will show you how to submit a custom Pytorch model, which is
most helpful for those that want to submit their own model.

1. Navigate, using your IDE’s Project Window (usually the left side of the
   screen that shows all the folders/files), into the
   ``sample-model-submission/examples/pytorch.py`` Python file.
   If you did the above steps correctly, you will be able to simply
   hit run on this file and the "prepping" service will commence.
   What does that mean? The code in this file downloads, prepares, and
   "mock scores" your model on a benchmark of choice, in order to ensure
   everything works correctly for the main Brain-Score site submission.
   It is like a check: if all goes well running this code, then your model
   is ready to submit to the site to be scored. (*Note: the first time running
   the ``pytorch.py`` file will take a bit, because you have to download the model
   (AlexNet in this case) weights/models from the site. This took about 15 mins
   for me on my Macbook (around 5 on my desktop PC), and only occurs the first
   time you download a pretrained model. Make sure your computer does not time out,
   or the download process can halt and you might have to re-run the file, which
   is a pain I am too familiar with unfortunately…*)
2. If this works correctly, then you will get a message on the Python console
   declaring::
     Test successful, you are ready to submit!
   and you can jump down below to Part 2, but I recommend
   reading the rest of the steps to understand what’s going on.
   A common error regarding SSL might happen at this point and is #2 on the
   ``Common Errors: Setup`` section, so check that out if you get that error.
3. Explore Further: navigate to ``sample-model-submission/models/base_models.py`` using
   the project explorer. You will see that this is basically a blank version of the
   ``pytorch.py`` file, and serves as a template to make new models to submit. The ``pytorch.py``
   file that you just successfully ran is an instance of this template, and this template
   declares how models must be structured to be scored. For now, we will just submit the
   AlexNet model as is.




Submitting a Model to Brain-Score.org Part 2: Upload
====================================================

If you made it this far, you are ready to upload your AlexNet model
and get a Brain-Score! In a nutshell, all this step ensues is zipping
the folder and making sure the files to submit are in the right place.

1. Right now, the working code we have confirmed is ready to submit is
   in the ``pytorch.py`` file. This file is mainly an example file, and
   thus we do not really want to submit it - instead, we are going to
   make a copy of it, rename it, and submit *that* version.
2. Before we do this, it is best to go ahead and make a folder in the
   root ``brainscore-brief`` directory to house all your submissions.
   This way, you can have a nice place to keep your submissions and
   reference them later if need be. For example, I made one called
   ``my_model_submissions`` located inside the project root (brainscore-brief),
   as seen below:

    .. image:: /docs/source/modules/tutorial_screenshots/mms.png
      :width: 600

3. We are now going to make the sub-folders necessary for submission.
   In general, the submission package will be a zip folder with a few things in
   it. It is important to get the folder “levels" right, or the website will not
   be able to parse the submission package and start running the correct code.
   So, this step in the guide is just about building this submission package.
   See below for the breakdown of zip file we will submit, with the various levels of the folders. Note the
   *two* ``\__init__.py`` files in both the ``models`` folder and root: ::


    my_alexnet_submission (main folder)
        models (subfolder)
            base_models.py
            __init__.py
        __init__.py
        setup.py

4. Now we will start making the submission package. In your ``my_model_submissions`` folder,
   create a new folder (that we will eventually zip to submit) called ``my_alexnet_submission``.
   In that newly created folder, create (yet another) folder called ``models``.
   You can see we are building the package up as explained above.
   Your IDE file/project explorer should look something like this below at this point:
    .. image:: /docs/source/modules/tutorial_screenshots/subfolders.png
      :width: 600

5. Next, we are going to add the ``setup.py`` file into the ``my_alexnet_submission`` folder.
   There are a few ways to do this, but the easiest is just to navigate into the
   ``sample_model_submission`` folder, and you will see a ``setup.py`` file there. We are going
   to copy that and place it inside of the ``my_alexnet_submission`` folder. You should
   be able to do this by just right clicking the file, copying, and then pasting inside
   the correct folder. It is important to paste it inside the
   ``my_alexnet_submission folder``, in order for it to be placed in the right spot.
   Your project should now look similiar to this:
    .. image:: /docs/source/modules/tutorial_screenshots/setup.png
      :width: 600

6. After this we will make the ``__init__.py`` file and place it inside the same folder as ``setup.py`` above.
   This ``__init__.py`` file is basically just a blank Python file that the submission
   needs in order to run. So, the easiest way to do this is to use your IDE to create
   a new Python file inside the correct folder: in Pycharm, you can do this by highlighting
   the ``my_alexnet_submission`` folder by clicking it, and it will be shown in blue.
   From there::
    right click -> new -> Python file
   Name this file ``__init__.py`` and
   click enter.  In the file, hit a new line (enter) so that the ``__init__.py`` file is not blank.
   Your package should look akin to this:
    .. image:: /docs/source/modules/tutorial_screenshots/init_py.png
      :width: 600
7. We are almost done! Copy the ``__init__.py`` file you just made and place that
   *additional* copy inside the ``models`` folder. Finally, we want to add the
   actual submission to the package. There are a few ways to do this, but
   for now we are just going to copy the code from ``pytorch.py`` into a blank
   Python file. Create a new Python file called ``base_models.py`` (note: the
   creation process is identical to how you created the ``__init__.py`` file above,
   just make sure this file is created inside the models folder), and paste the
   code from ``pytorch.py`` into there. This creates another instance of the
   ``base_models.py`` file, filled with ``pytorch.py`` ’s code, which is in this case the
   AlexNet model. You are basically done at this point, and your final package
   should look akin to the picture below. Remember, the actual model is now contained
   in the ``models/base_models.py`` file, and that is what is getting run on our site to get a score for you.
    .. image:: /docs/source/modules/tutorial_screenshots/final_submit.png
      :width: 600
8. You are now ready to submit! Zip the folder named ``my_alexnet_subission``,
   navigate to http://www.brain-score.org/profile/, log in/create a new account,
   and submit the model! Usually (depending on how busy the time of year is)
   it will take around 1 hour or so to score, but might take longer. If you
   do not see a score within 24 hours, contact us and we can send you
   (soon you will have access to this yourself)
   the error logs to resubmit. You have now successfully submitted a model!
   Congrats, and we look forward to having more submissions from you.
   In the future, you can just copy the submission package and paste
   in your code into ``models/base_models.py``, and it should work (which
   is why we had you make that whole package in the first place!)
9. *Experiment Further*: Try submitting different pretrained models,
   like ``vgg16``, ``Inception``, or ``RESNET``, that are available from
   Pytorch. Keras works the same way, and for the most part this
   tutorial can be followed to use ``keras.py`` instead of ``pytorch.py``; if you
   use keras, just remember to add it to the ``setup.py`` file. Finally,
   when using a different pretrained model (or a custom model) see the
   section below on layer selection to get the layers needed in the actual model,
   which will be different according to the model used.



Submitting a Model to Brain-Score.org Part 3: Custom model (Optional)
=====================================================================

At this point, I would say that you are pretty comfortable with the submission,
and hopefully you have submitted at least one model and gotten a score.
So, in this section, I will skip some of the parts that are common with
submitting a custom model (vs. something like AlexNet), and just focus on what is different.

1. In short, submitting a custom model is not that difficult
   for those that have already submitted a model like AlexNet
   and have a submission package ready. If you have not done this,
   I highly recommend going through this tutorial beforehand, or else you will
   encounter some errors along the way.
2. The entire package we submit will be the same as a pretrained model,
   but with the ``models/base_models.py`` file different (as the model itself is different).
   So, I would recommend just copying the ``my_alexnet_submission`` folder,
   pasting it into the ``my_model_submissions`` folder, and renaming it to something
   like ``my_custom_submission``. This will take care of all the tricky
   submission stuff, and you can just focus on implementing the actual model inside ``models/base_models.py``.
3. Now the fun part: scoring a model that you create! In this guide we will be implementing
   a light-weight Pytorch model and submitting that. All this entails is adding
   a little bit of extra stuff to ``models/base_models.py``.
4. The easiest way to do this is to simply copy all the code in the file
   from here, and I can walk you through the important stuff that is necessary
   to understand how to submit a custom model. It is, in a nutshell, just a
   slightly more complicated version of the original ``base_models.py`` template
   in the ``sample-model-submissions`` folder.
5. The first is the imports: you will most likely need all of them that
   the code above has listed. If you try to run the above code in Google Colab
   (which is basically a Google version of Jupyter Notebooks), it will not
   run (due to packages not being installed), and is just for visual
   purposes only; copy and paste the code into your ``models/base_models.py`` file.
   Next, you see the class definition of the custom model in Pytorch, lines 19 - 35.
   Line 39 deals with preprocessing, line 40 is the ``PytorchWrapper`` that
   converts a model into a neuroscience-ready network to run benchmarks on,
   and lines 41 - 43 are the layers of the network that will be scored.
   These usually are all the layers, or you can just pick ones you specifically
   want. You will need all of this, and most likely will only change the
   actual layer names based on the network/what you want scored.
6. Lines 47-55 are just the name of the model, and should be replaced
   with whatever you want to call your model. Lines 59-73 tell the
   code what to score, and you most likely will not have to
   change this. Lines 76-89 is a layer function that simply returns a
   list of the layers to consider, and will probably be identical to line 43.
   Lines 92-98 deal with ``bibtex``, and you can replace this with your ``bibtex``
   if your model has been published. Lastly, lines 101-104 are the main driver
   code, and you shouldn't need to modify this.
7. That’s it! You can change the actual model in lines 19-35, just make sure you
   change the layer names as well. Run your ``models/base_models.py`` file,
   and you should get the following message indicating you are good to submit::
    Test successful, you are ready to submit!
   At this point, all that is left is to zip the ``my_custom_submission`` folder
   and actually submit on our site! At this point, if you run into any errors,
   check out the ``Common Errors: Submission section`` of this guide, and if you can’t
   find a solution, feel free to email us!
|
|
.. image:: /docs/source/modules/tutorial_screenshots/mit_logo.png
    :width: 300
    :height: 200
    :align: center
|
.. image:: /docs/source/modules/tutorial_screenshots/mibr_logo.png
    :width: 300
    :height: 200
    :align: center
|
.. image:: /docs/source/modules/tutorial_screenshots/bcs2.jpg
    :width: 300
    :height: 75
    :align: center



Common Errors: Setup
====================

Below are some common error that you might encouinter while setting up
this project or doing this tutorial. We will add more soon!

1. When running ``pip install .``, you get a message
   from the terminal like::
     Directory '.' is not installable. Neither 'setup.py' nor 'pyproject.toml' found.
   *Cause*: Not running ``pip install .`` in the right
   directory: most likely you are in the original ``brainscore-brief`` folder we created,
   and not the ``sample_model_submission`` sub-folder that is the repo we should be in.

   *Fix*: if you are in the main ``brainscore-brief``
   folder, simply run::
    cd sample_model_submission
   and then rerun
   the ::
    pip install .
   command. This navigates to the correct sample_model_submission subfolder and
   installs the packages where they are supposed to be.
   More generally: make sure you are in the ``sample_model_submission`` folder
   (and not its parent or child folder) before you run the pip command above. This should fix the error.

2. After install while running ``pytorch.py``
   for the first time, you get::
    ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1076)
   *Cause*: Pytorch’s backend. The SSL certificate for downloading a pre-trained model has expired
   from their end and pytorch should renew soon (usually ~4 hrs)

   *Fix*: If you can’t wait, add the following lines of code to your ``pytorch.py``
   (or whatever file is using the pretrained Pytorch models): *Note: Pycharm might throw a warning about this
   line, but you can disregard)*::
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context



Common Errors: Submission
=========================

1. It has been 24 hours since I submitted my model, and I have not gotten a score? What happened?

   *Cause*: There are many issues that could cause this.

   *Fix*:  If it happens, email ``mferg@mit.edu`` and I can check the logs
   and tell you what happened. You will, very soon, be able to log in and check the logs yourself,
   so stay tuned!



Frequently Asked Questions
==========================

1. **What are all the numbers on the Brain-Score site?**

   As of now on the leaderboard (Brain-Score), there are 6 numbers
   that your model would get: ``average``, ``V1``, ``V2``, ``V4``, ``IT``, and ``Behavioral``.
   Each one of these is a set of benchmarks that tests how "brain-like"
   your model is to various cognitive and neural data- in essence,
   it is a measure of how close the model is to the brain.
   Models are also tested on "Engineering" benchmarks which are non-brain,
   typically machine learning measures that the brain measures can be related
   to (e.g. more V1-like → more robust to image perturbations).

2. **What is the idea behind Brain-Score? Where can I learn more?**

   The website is a great place to start, and for those who really
   want to dive deep, I would recommend reading the technical paper(https://www.biorxiv.org/content/10.1101/407007v1)
   and the perspective paper (https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X)
   that outline the idea and the inner workings of how Brain-Score operates.

3. **I was looking at the code and I found an error in the code/docs/etc. How can I contribute?**

   Right now, the easiest way would be to fork (make a copy of the Brain-Score
   project repos in your own Github) our brain-score repos,
   edit your version, and submit a pull request (PR) to merge it
   into our master branch. We will have to confirm that PR, but will thank you for contributing!

4. **I really like Brain-Score, and I have some ideas that I would love to
   talk to someone about. How do I get in touch/who do I talk to?**

   Martin Schrimpf, the main creator of Brain-Score, would be a great place to start.
   Chris Shay, the DiCarlo Lab manager, can also help, and if you need to
   talk to Jim DiCarlo himself you can reach out as well.  All contact
   info is on the lab website: http://dicarlolab.mit.edu/

5. **I am a neuroscientist/cognitive scientist/cognitive-AI-neuro-computational-systems-scientist
   and would love to talk theory or contribute to benchmarks, as I have collected data or
   have theoretical questions. What should I do?**

   I would reach out to Martin, Chris, or Jim directly, via the lab website as stated above.

6. **Is there any reward for reaching the top overall brain-score? Or even a top
   score on the individual benchmarks?**

   We hope to set up a dedicated competition in the near future, but we
   monitor the site and if you get a top score, we will know and reach out.
   If you are local and get the top average score, we might even buy you a beer if you’re nice to us :)
