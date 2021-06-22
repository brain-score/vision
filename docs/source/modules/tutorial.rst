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

- **brain-score**: the heart of the code that runs analysis and comparisons.
- **sample-model-submission**: template and examples for model submissions.
- **candidate_models**: various pre-trained models/models that you can submit.
- **model-tools**: helper functions that translate from machine learning models
  to brain models to be tested on brain-score.
- **brainio_collection**: the repo that packages and collects the stimuli/data.
- **brainio_base**: repo that contains various data structures for BrainIO.
- **result_caching**: a helper repo to store the results of function calls so they can
  be re-used without re-computing.
- **brain-score.web**: website front and back end.
- **Brainio_contrib (archived)**: used in the past to contribute stimuli and datasets
  (now part of **brainio_collection**).

Which repos will I use?
-----------------------
When we get to the install guide, I will show you exactly how to
clone/fork repos for your own project in the easiest way possible.
But for now, you will mainly only need the **sample-model-submission** repo.

How do I get a Brain-Score for my model?
----------------------------------------

Brain-Score currently has two “ways" to score your model:

- *On your machine*:  your model is scored on one or more benchmarks (a benchmark is a
  standard to compare against, like Majaj-Hong’s 2015 paper on IT results (which you
  can read more about here: https://brain-score.readthedocs.io/en/latest/modules/benchmarks.html). This is useful if you would like to test and see if your
  model is ready to be submitted and run against all benchmarks, or if you want a quick
  and dirty way to score on a single benchmark, like V4 or IT. This is mainly used to
  test and/or optimize your model before submitting to the main site.
- *Remote*: your model is run on Brain-Score’s website, and contains scores against all the
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

- Get a copy of  the **sample-model-submission** repo from our Github. If you are wondering
  about what the various repos on the main Brain-Score github do, check out the Overview section above.
  Here is the command for a terminal:
    git clone https://github.com/brain-score/sample-model-submission.git
- Install the dependencies via pip. (Make sure to switch into the **sample-model-submission** folder
  that was created in step one above when you cloned the repo). You can read more about this in the
  Install Brain-Score Repos and Dependencies section below. Again, the command for the terminal is:
    pip install .
- Specify the model to test. You can do this a few ways, but in this case, we can
  just use the model defined in **pytorch.py**. More info on this step can be found in
  the Submitting a Model to Brain-Score.org Part 1: Preparing the Model section of
  this guide. In essence, you need to implement the various functions outlined in
  the **sample-model-submission/models/base_models.py** file.
- Test the model on your machine. You can do this simply by hitting “run"
  on the **pytorch.py** file (in this case) or the file where you implemented
  the **base_models.py** template. If you followed the steps correctly, you should
  receive a message on the Python console indicating that you are ready to submit.
- Submit the package to brain-score.org. This step is slightly brittle as of now,
  and is easy to do wrong; I recommend skipping down to **Submitting a Model to
  Brain-Score.org Part 2: Upload** section to see the structure of the zip file that
  our site needs in order to process your submission.
- That’s it! Read more below to get a better idea of the process, or to help fix bugs that might come up.



Install Brain-Score Repos and Dependencies
==========================================
In this section, I will show you how to get packages installed and dependencies
linked in order to run setup for submission and scoring.

- Download PyCharm (https://www.jetbrains.com/pycharm/download/#section=windows) or another IDE.
  *Note: you do not have to use Pycharm per se, but we recommend it, and this guide will show*
  *you how to integrate Brain-Score with it. It is the easier way.*
  If you do not have experience with Pycharm, here’s a nice tutorial: https://www.tutorialspoint.com/pycharm/index.htm.
  Again, we recommend and like Pycharm, but this tutorial is neutral in the sense that you can use
  any IDE, as the steps are very similar for other environments, but this document will
  feature Pycharm screenshots.
- Once Pycharm (or your own IDE) is set up, we will start the install of Brain-Score
  and its various repos. First, in your file explorer, make a new file on your desktop
  or favorite place to save things. I personally made a folder called **brainscore-brief**
  in my **/desktop** folder. Create a new project, and your IDE should ask you for a location
  to create said project. I recommend setting up the path to be the newly created folder
  from above, in my case the path is:
    /Users/mike/desktop/brainscore-brief
  Your IDE will create a Python interpreter for the project (the piece of code that
  tells the computer how to run various Python commands) by setting up a Virtual Environment
  for you automatically. A venv is handy because installing the dependencies that Brain-Score
  needs will not conflict with other packages on your computer if you use a venv.
  To the left on your screen, you will see your folder brainscore-brief that is the
  project root. If you click to expand it, then you will see an orange folder marked **venv**
  that contains all the venv files and whatnot. I would not mess with the **venv** folder or
  download anything in there. Again, your IDE will most likely be different if you do not use
  Pycharm, but the main points still hold.
- Next, we are going to clone the repos we need in order to get the code from Github for Brain-Score.
  The easiest way to do this is to install Git on your computer from: https://git-scm.com/download/win.
  Once this is installed, open up your terminal and navigate into the **brainscore-brief**
  folder. In my case, the commands are:
    cd desktop -> cd brainscore-brief

  After you are in this folder,
  run:
    git clone https://github.com/brain-score/sample-model-submission.git
  This will copy our sample-model-submission code from Github into your local machine to run later on.
  Switching back to your IDE’s file explorer, you should now see a folder called **sample-model-submission**
  in your project folder. Clicking on/expanding this will show you the various files and
  programs that are in our collection for the **sample-model-submission** repo.
  You can see the various folders in the image below: the top level **brainscore-brief**
  is the folder that we created a few steps ago. The next level **sample-model-submission**
  is the repo cloned from our Github. You should now see something akin to below when you
  look at your version on your machine:

   .. image:: C:\Users\Mike\Desktop\MIT\Brain-Score\brain-score_local\docs\source\modules\tutorial_screenshots\image1.png
      :width: 600

- We will now install the pip packages that our code needs to run: things like **scipy** and
  **imageio**, etc. In your IDE or using your main computer terminal, switch into your root
  directory, in this case **brainscore-brief**. Navigate into the repo directory,
  **sample-model-submission**, using the command
    cd sample-model-submission
  (which should be one level down from the original created folder/directory).
  Once you are in this brain-score repo,
  run the command below  (note the **.** This tells pip to install all the dependencies you will
  need: a nice and handy way to do this)
    pip install .
  In Pycharm, you can check to make sure these dependencies were installed correctly
  by going into
    Pycharm -> settings (preferences on Mac) -> project: brainscore-brief -> project interpreter
  where you will see a list of around 100 packages like **toml**, **xarray**, and
  **Keras-preprocessing**. (Note: installing all the dependencies will take around 2-5 mins
  on your machine, depending on the hardware/internet). A different IDE will most likely
  have a similar feature, but this tutorial gives Pycharm as an example.
- Congrats! You now have completed the hardest part of install.
  Also remember before running the pip command, make sure to navigate
  using terminal into the correct folder using the
    cd sample-model-submission
  command to ensure it is installed in the right place- otherwise you get error #1
  in the **Common Errors: Setup** section. Feel free to explore the various
  files and get a feel for them.

That’s it! You have downloaded and retrieved all of the files you need to submit a model!
Take a break and go get some lunch or some donuts. If you get an error that is not
listed/resolved below, reach out to us at MIT and we can (most likely) help:

- msch@mit.edu
- mferg@mit.edu
- cshay@mit.edu

