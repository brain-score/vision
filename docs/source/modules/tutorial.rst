.. _Tutorial:
.. _benchmarks: https://brain-score.readthedocs.io/en/latest/modules/benchmarks.html

========
Tutorial
========

About
=====
Brain-Score is a platform that measures how brain-like candidate models are.
It is agnostic to the model class as long as models can make neural and/or
behavioral predictions in response to visual stimuli. In a nutshell, Brain-Score evaluates
the similarity to brain regions in the primate ventral stream as well as behavioral outputs,
and gives a score (usually ranging from 0 to 1) on these various
brain regions and behavioral benchmarks. This guide is a tutorial for researchers and tinkerers
alike that outlines the setup, submission, and feedback system for users.


Quickstart
==========
In this section, we will provide a quick and easy way
to get your model(s) ready for submission. This is mainly for those who do not have the time to read
or do the whole tutorial, or for those who just want to go ahead and submit
a model quickly; however, we recommend referring back to this tutorial,
especially if you encounter errors. This section also does not
have pictures, which the other more lengthy sections below do. As an example,
we will submit a version of AlexNet from Pytorch’s library; the main steps are outlined below:

1. Clone a copy of  the ``sample-model-submission`` repo from from
   https://github.com/brain-score/sample-model-submission. If you are wondering
   about what the various repos on the main Brain-Score github do, check out the `Overview`_ section below.
   Here is the command for a terminal: ::
     git clone https://github.com/brain-score/sample-model-submission.git
2. Install the dependencies via pip. (Make sure to switch into the ``sample-model-submission`` folder
   that was created in step one above when you cloned the repo). You can read more about this in the
   `Install Brain-Score Repos and Dependencies`_ section below. The command for a terminal is: ::
     pip install .
3. Specify the model to test. In this example, we can
   just use the model defined in ``pytorch.py``, located in the ``examples`` folder
   of this repository. More info on this step can be found in
   the `Submitting a Model to Brain-Score.org Part 1: Preparing the Model`_ section of
   this guide. In essence, you need to implement the functions outlined in
   the ``sample-model-submission/models/base_models.py`` file.
4. Test the model on your machine. You can do this simply by running the ``pytorch.py`` file
   (again, located in the ``examples`` folder)
   or the file where you implemented
   the ``base_models.py`` template. If you followed the steps correctly, you should
   receive a message on the Python console indicating that you are ready to submit.
5. Submit to ``brain-score.org``. This step is slightly brittle as of now,
   and is easy to do wrong; we recommend skipping down to
   `Submitting a Model to Brain-Score.org Part 2: Upload`_
   section to see the structure of the zip file that
   our site needs in order to process your submission.
   That’s it! Read more below to get a better idea of the process, or to help fix bugs that might come up.


Overview
========

Brain-Score sounds great! How do I get started? What do I do next? How do I submit?
--------------------------------------------------------------------

This tutorial walks you through the entire process, end-to-end.
Start here if you have already
(or would like to create) a model to submit and get a Brain-Score.
This guide will walk you through from downloading our code to receiving a
submission score from the website. Completion of the tutorial
should take about 15-20 minutes.

What does Brain-Score look like, exactly?
--------------------------------------------------------------------
The main code for Brain-Score is contained and hosted on Github (https://github.com/brain-score)
on various repos, free for anyone to fork,
clone, or download. The main page has 10 repos (9 visible to non-devs)
that make up the entire source code base- this can be overwhelming, but
fear not. We will walk you through what each does, as well as explain that
you most likely will only need one to get a model up and scored. It is
good to see, however, the structure of how Brain-Score operates with its
various repos. They are:

1. ``brain-score``: the heart of the code that runs analysis and comparisons.
2. ``sample-model-submission``: template and examples for model submissions.
3. ``candidate_models``: various pre-trained models/models that have already been scored.
4. ``model-tools``: helper functions that translate from machine learning models
   to brain models to be tested on brain-score.
5. ``brainio_collection``: the repo that packages and collects the stimuli/data.
6. ``brainio_base``: repo that contains various data structures for ``BrainIO``.
7. ``result_caching``: a helper repo to store the results of function calls so they can
   be re-used without re-computing.
8. ``brain-score.web``: website front and back end.
9. ``Brainio_contrib`` (archived): used in the past to contribute stimuli and datasets
   (now part of ``brainio_collection``).

Which repo(s) will I use?
-----------------------
When we get to the install guide, we will show you exactly how to
clone/fork a repo for your own project in the easiest way possible.
But for now, you will mainly only need the ``sample-model-submission`` repo.

How do I get a Brain-Score for my model?
----------------------------------------
When you submit a model to our website, it is scored against all
availible benchmarks (e.g. neural predictivity on IT recordings
from Majaj*, Hong* et al. 2015; see benchmarks_ for more details). The (hierarchical) mean of
individual benchmark scores is the Brain-Score itself.





Before submitting your model, you might want to get a quick sense of its performance;
to that end, we provide *public* benchmarks that you can run locally, which are different subsets
of the larger benchmark dataset. This is mainly used to optimize your model before
submission, or if you want to score models locally on publicly available data.
*Note: a submission is the only way to score models on private evaluation data.*





Why do you recommend installing and submitting the way outlined in this guide? In other words, why should I do it your way?
------------------------------------------------------------------------------

A reasonable question, and it is always good to be skeptical. The short answer
is that using an IDE like Pycharm or VSCode along with virtual environments
drastically cuts the error rate for install down, as well as makes the whole
process of installing dependencies easier. Using a venv also helps with headaches
caused by clashes between Anaconda and PIP, and Pycharm
(or another IDE like VSCode) takes care of that.

Do I have to read/do this entire tutorial to submit a model?
------------------------------------------------------------

No - You can just read the `Quickstart`_ section, if you do not
wish to read/do this entire tutorial. However, we recommend referring back to this
tutorial to help with errors that might pop up along the way.




Install Brain-Score Repos and Dependencies
==========================================
In this section, we will show you how to get packages installed and dependencies
linked in order to run setup for submission and scoring.

1. Download PyCharm (https://www.jetbrains.com/pycharm/download) or another IDE.
   *Note: you do not have to use Pycharm per se, but we recommend it, and this guide will show*
   *you how to integrate Brain-Score with it.*
   If you do not have experience with Pycharm, here’s a nice tutorial: https://www.tutorialspoint.com/pycharm/index.htm.
   Again, we recommend and like Pycharm, but this tutorial is neutral in the sense that you can use
   any IDE, as the steps are very similar for other environments, but this document will
   feature Pycharm screenshots.
2. Once Pycharm (or your own IDE) is set up, we will start the install of Brain-Score
   and its various repos. First, in your file explorer, make a new file on your desktop
   or favorite place to save things. I personally made a folder called ``brainscore-brief``
   in my ``/desktop`` folder. Create a new project, and your IDE should ask you for a location
   to create said project. We recommend setting up the path to be the newly created folder
   from above, in my case the path is ::
     /Users/mike/desktop/brainscore-brief
   Your IDE will create a Python interpreter for the project (the piece of code that
   tells the computer how to run various Python commands) by setting up a Virtual Environment
   for you automatically. A venv is handy because installing the dependencies that Brain-Score
   needs will not conflict with other packages on your computer if you use a venv.
   To the left on your screen, you will see your folder ``brainscore-brief`` that is the
   project root. If you click to expand it, then you will see an orange folder marked ``venv``
   that contains all the venv files and whatnot. I would not mess with the ``venv`` folder or
   download anything in there. Again, your IDE will most likely be different if you do not use
   Pycharm, but the main points still hold.
3. Next, we are going to clone the repo we need from Github for Brain-Score.
   The easiest way to do this is to install Git on your computer from (for Windows): https://git-scm.com/download/win.
   On Mac, Git should already be insalled, but if not, visit https://git-scm.com/download/mac.
   Once this is installed, open up your terminal and navigate into the ``brainscore-brief``
   folder. In my case, the commands are ::
     cd desktop
     cd brainscore-brief

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

    .. image:: tutorial_screenshots/sms.png
       :width: 600

4. We will now install the pip packages that our code needs to run: things like ``scipy`` and
   ``imageio`` , etc. In your IDE, or using your main computer terminal, switch into your root
   directory, in this case ``brainscore-brief``. Navigate into the repo directory,
   ``sample-model-submission``, using the command ::
     cd sample-model-submission
   (which should be one level down from the original created folder/directory).
   Once you are in this ``sample-model-submission`` repo,
   run the command below  (note the period; this tells pip to install all the dependencies you will
   need, a nice and handy way to do this). ::
     pip install .
   In Pycharm, you can check to make sure these dependencies were installed correctly
   by going into ::
     Pycharm -> settings (preferences on Mac) -> project: brainscore-brief -> project interpreter
   where you will see a list of around 100 packages like ``toml``, ``xarray``, and
   ``Keras-preprocessing``. *(Note: installing all the dependencies will take around 2-5 mins
   on your machine, depending on the hardware/internet)*. A different IDE will most likely
   have a similar feature, but this tutorial uses Pycharm.
5. Congrats! You now have completed the hardest part of install.
   Also remember before running the pip command, make sure to navigate
   using terminal into the correct folder using the ::
     cd sample-model-submission
   command to ensure it is installed in the right place- otherwise you get error #1
   in the `Common Errors: Setup`_ section. Feel free to explore the various
   files and get a feel for them.

That’s it! You have downloaded and retrieved all of the files you need to submit a model!
Take a break and go get some lunch or some donuts. If you get an error that is not
listed/resolved below, reach out to us at MIT and we can (most likely) help:

- msch@mit.edu
- mferg@mit.edu
- cshay@mit.edu

Submitting a Model to Brain-Score.org Part 1: Preparing the Model
=============================================================

By now you should have the ``sample-model-submission`` repo cloned and
the dependencies installed. It is now time to prepare your model to be
submitted! In this part we will submit a standard, generic form of AlexNet
(implemented in Pytorch) in order to get a feel for the submission process.
In Part 3 we will show you how to submit a custom Pytorch model, which is
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
   this file will take a bit, because you have to download the model
   (AlexNet in this case) weights as well as ImageNet validation images (for PCA initialization).
2. If this works correctly, then you will get a message on the Python console
   declaring::
     Test successful, you are ready to submit!
   and you can jump down below to Part 2, but we recommend
   reading the rest of the steps to understand what’s going on.
   A common error regarding SSL might happen at this point and is #2 on the
   `Common Errors: Setup`_ section, so check that out if you get that error.
3. Explore Further: navigate to ``sample-model-submission/models/base_models.py`` using
   the project explorer. You will see that this is basically a blank version of the
   ``pytorch.py`` file, and serves as a template to make new models to submit. The ``pytorch.py``
   file that you just successfully ran is an instance of this template, and this template
   declares how models must be structured to be scored. For now, we will just submit the
   AlexNet model as is.




Submitting a Model to Brain-Score.org Part 2: Upload
====================================================

If you made it this far, you are ready to upload your AlexNet model
and get a Brain-Score! In a nutshell, this step is simply zipping
the folder and making sure the files to submit are in the right place.

1. Right now, the working code we have confirmed is ready to be submitted is
   in the ``pytorch.py`` file. This file is mainly an example file, and
   thus we do not really want to submit it - instead, we are going to
   make a copy of it, rename it, and submit *that* version.
2. Before we do this, it is best to go ahead and make a folder in the
   root ``brainscore-brief`` directory to house all your submissions.
   This way, you can have a nice place to keep your submissions and
   reference them later if need be. For example, I made one called
   ``my_model_submissions`` located inside the project root (``brainscore-brief``),
   as seen below:

    .. image:: tutorial_screenshots/mms.png
      :width: 600

3. We are now going to make the sub-folders necessary for submission.
   In general, the submission package will be a zip folder with a few things in
   it. It is important to get the folder “levels" right, or the website will not
   be able to parse the submission package and start running the correct code.
   So, this step in the guide is just about building this submission package.
   See below for the breakdown of zip file we will submit, with the various levels of the folders. Note the
   *two* ``__init__.py`` files in both the ``models`` folder and root: ::


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
    .. image:: tutorial_screenshots/subfolders.png
      :width: 600

5. Next, we are going to add the ``setup.py`` file into the ``my_alexnet_submission`` folder.
   There are a few ways to do this, but the easiest is just to navigate into the
   ``sample_model_submission`` folder, and you will see a ``setup.py`` file there. We are going
   to copy that and place it inside of the ``my_alexnet_submission`` folder. You should
   be able to do this by just right clicking the file, copying, and then pasting inside
   the correct folder. It is important to paste it inside the
   ``my_alexnet_submission`` folder, in order for it to be placed in the right spot.
   Your project should now look similiar to this:
    .. image:: tutorial_screenshots/setup.png
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
   Your package should now look akin to this:
    .. image:: tutorial_screenshots/init_py.png
      :width: 600
7. We are almost done! Copy the ``__init__.py`` file you just made and place that
   *additional* copy inside the ``models`` folder. Finally, we want to add the
   actual submission to the package. There are a few ways to do this, but
   for now we are just going to copy the code from ``pytorch.py`` into a blank
   Python file. Create a new Python file called ``base_models.py`` (the
   creation process is identical to how you created the ``__init__.py`` file above,
   just make sure this file is created inside the models folder), and paste the
   code from ``pytorch.py`` into there. This creates another instance of the
   ``base_models.py`` file, filled with ``pytorch.py`` ’s code, which is in this case the
   AlexNet model. You are basically done at this point, and your final package
   should look similiar to the picture below. Remember, the actual model is now contained
   in the ``models/base_models.py`` file, and that is what is getting run on our site to get a score for you.
    .. image:: tutorial_screenshots/final_submit.png
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


Submitting a Model to Brain-Score.org Part 3: Custom model (Optional)
=====================================================================

At this point, I would say that you are pretty comfortable with the submission,
and hopefully you have submitted at least one model and gotten a score.
So, in this section, we will skip some of the parts that are common with
submitting a custom model (vs. something like AlexNet), and just focus on what is different.

1. In short, submitting a custom model is not that difficult
   for those that have already submitted a model like AlexNet
   and have a submission package ready. If you have not done this,
   we highly recommend going through this tutorial beforehand, or else you will
   encounter some errors along the way.
2. The entire package we submit will be the same as a pretrained model,
   but with the ``models/base_models.py`` file different (as the model itself is different).
   So, we would recommend just copying the ``my_alexnet_submission`` folder,
   pasting it into the ``my_model_submissions`` folder, and renaming it to something
   like ``my_custom_submission``. This will take care of all the tricky
   submission stuff, and you can just focus on implementing the actual model inside ``models/base_models.py``.
3. Now the fun part: scoring a model that you create! In this section we will be implementing
   a light-weight Pytorch model and submitting that. All this entails is adding
   a little bit of extra stuff to ``models/base_models.py``.
4. The easiest way to do this is to simply copy all the code in the block below,
   and we can walk you through the important stuff that is necessary
   to understand how to submit a custom model. It is, in a nutshell, just a
   slightly more complicated version of the original ``base_models.py`` template
   in the ``sample-model-submissions`` folder. The code is listed below ::

    # Custom Pytorch model from:
    # https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

    from model_tools.check_submission import check_models
    import numpy as np
    import torch
    from torch import nn
    import functools
    from model_tools.activations.pytorch import PytorchWrapper
    from brainscore import score_model
    from model_tools.brain_transformation import ModelCommitment
    from model_tools.activations.pytorch import load_preprocess_images
    from brainscore import score_model

    """
    Template module for a base model submission to brain-score
    """

    # define your custom model here:
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
            self.relu1 = torch.nn.ReLU()
            linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
            self.linear = torch.nn.Linear(int(linear_input_size), 1000)
            self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = self.relu2(x)
            return x


    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(identifier='my-model', model=MyModel(), preprocessing=preprocessing)

    # actually make the model, with the layers you want to see specified:
    model = ModelCommitment(identifier='my-model', activations_model=activations_model,
                            # specify layers to consider
                            layers=['conv1', 'relu1', 'relu2'])


    # The model names to consider. If you are making a custom model, then you most likley want to change
    # the return value of this function.
    def get_model_list():
        """
        This method defines all submitted model names. It returns a list of model names.
        The name is then used in the get_model method to fetch the actual model instance.
        If the submission contains only one model, return a one item list.
        :return: a list of model string names
        """

        return ['my-model']


    # get_model method actually gets the model. For a custom model, this is just linked to the
    # model we defined above.
    def get_model(name):
        """
        This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
        containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
        keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
        wrappers.
        :param name: the name of the model to fetch
        :return: the model instance
        """
        assert name == 'my-model'

        # link the custom model to the wrapper object(activations_model above):
        wrapper = activations_model
        wrapper.image_size = 224
        return wrapper


    # get_layers method to tell the code what layers to consider. If you are submitting a custom
    # model, then you will most likley need to change this method's return values.
    def get_layers(name):
        """
        This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
        layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
        faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
        size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
        model, the layer name are for instance dot concatenated per module, e.g. "features.2".
        :param name: the name of the model, to return the layers for
        :return: a list of strings containing all layers, that should be considered as brain area.
        """

        # quick check to make sure the model is the correct one:
        assert name == 'my-model'

        # returns the layers you want to consider
        return  ['conv1', 'relu1', 'relu2']

    # Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
    # model has been published, or leave the empty return value if there is no publication to refer to.
    def get_bibtex(model_identifier):
        """
        A method returning the bibtex reference of the requested model as a string.
        """

        # from pytorch.py:
        return ''

    # Main Method: In submitting a custom model, you should not have to mess with this.
    if __name__ == '__main__':
        # Use this method to ensure the correctness of the BaseModel implementations.
        # It executes a mock run of brain-score benchmarks.
        check_models.check_base_models(__name__)




5. The first is the imports: you will most likely need all of them that
   the code above has listed. If you try to run the above code in Google Colab
   (which is basically a Google version of Jupyter Notebooks), it will not
   run (due to packages not being installed), and is just for visual
   purposes only; copy and paste the code into your ``models/base_models.py`` file.
   Next, you see the class definition of the custom model in Pytorch, followed by model
   preprocessing, the ``PytorchWrapper`` that
   converts a base model into an activations model to extract activations from,
   and the ModelCommitment to convert the activations model into a BrainModel to run on the benchmarks.
   We usually test the layers at the outputs of blocks, but this choice is up to you.
   You will need all of this, and most likely will only change the
   actual layer names based on the network/what you want scored.
6. Next is the function for "naming" the model, and should be replaced
   with whatever you want to call your model. The next function tells the
   code what to score, and you most likely will not have to
   change this. This is followed by a layer function that simply returns a
   list of the layers to consider.
   Next is is the ``bibtex`` method, and you can replace this with your ``bibtex``
   if your model has been published. Lastly, the concluding lines contain and call
   the ``__main__`` method, and you shouldn't need to modify this.
7. That’s it! You can change the actual model in the class definition, just make sure you
   change the layer names as well. Run your ``models/base_models.py`` file,
   and you should get the following message indicating you are good to submit::
    Test successful, you are ready to submit!
   At this point, all that is left is to zip the ``my_custom_submission`` folder
   and actually submit on our site! If you run into any errors,
   check out the `Common Errors: Submission`_ section of this guide, and if you can’t
   find a solution, feel free to email us!

Common Errors: Setup
====================

Below are some common errors that you might encounter while setting up
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
   command. This navigates to the correct ``sample_model_submission`` subfolder and
   installs the packages where they are supposed to be.
   More generally: make sure you are in the ``sample_model_submission`` folder
   (and not its parent or child folder) before you run the pip command above. This should fix the error.

2. After install while running ``pytorch.py``
   for the first time, you get::
    ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1076)
   *Cause*: Pytorch’s backend. The SSL certificate for downloading a pre-trained model has expired
   from their end and Pytorch should renew soon (usually ~4 hrs)

   *Fix*: If you can’t wait, add the following lines of code to your ``pytorch.py``
   (or whatever file is using the pretrained Pytorch models): *Note: Pycharm might throw a warning about this
   line, but you can disregard)*::
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context



Common Errors: Submission
=========================

1. It has been 24 hours since I submitted my model, and I have not gotten a score? What happened?

   *Cause*: There are many issues that could cause this.

   *Fix*:  If it happens, email ``mferg@mit.edu`` and we can check the logs
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
   want to dive deep, we would recommend reading the technical paper(https://www.biorxiv.org/content/10.1101/407007v1)
   and the perspective paper (https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X)
   that outline the idea and the inner workings of how Brain-Score operates.

3. **I was looking at the code and I found an error in the code/docs/etc. How can I contribute?**

   Right now, the easiest way would be to fork (make a copy of the Brain-Score
   project repos in your own Github) our Brain-Score repos,
   edit your version, and submit a pull request (PR) to merge it
   into our master branch. We will have to confirm that PR, but will thank you for contributing!

4. **I really like Brain-Score, and I have some ideas that I would love to
   talk to someone about. How do I get in touch/who do I talk to?**

   Martin Schrimpf, the main creator of Brain-Score, would be a great place to start.
   Chris Shay, the DiCarlo Lab manager, can also help, and if you need to
   talk to Jim DiCarlo himself you can reach out as well.  We will also be
   creating a mailing list soon, so stay tuned. All contact
   info is on the lab website: http://dicarlolab.mit.edu/

5. **I am a neuroscientist/cognitive scientist/cognitive-AI-neuro-computational-systems-scientist
   and would love to talk theory or contribute to benchmarks, as I have collected data or
   have theoretical questions. What should I do?**

   I would reach out to Martin, Chris, or Jim directly, via the lab website as stated above.

6. **Is there any reward for reaching the top overall Brain-Ccore? Or even a top
   score on the individual benchmarks?**

   We hope to set up a dedicated competition in the near future, but we
   monitor the site and if you get a top score, we will know and reach out.
   If you are local and get the top average score, we might even buy you a beer if you’re nice to us :)
