.. _model_tutorial:

.. _technical paper: https://www.biorxiv.org/content/10.1101/407007v1
.. _perspective paper: https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X
.. _Pycharm: https://www.jetbrains.com/pycharm/download
.. _introduction: https://www.tutorialspoint.com/pycharm/index.htm
.. _sample-model-submission: https://github.com/brain-score/sample-model-submission
.. _github: https://github.com/brain-score
.. _github repository: https://github.com/brain-score/vision
.. _windows: https://git-scm.com/download/win
.. _mac: https://git-scm.com/download/mac
.. _profile: http://www.brain-score.org/profile/

==============
Model Tutorial
==============

The Brain-Score platform aims to yield strong computational models of the ventral stream.
We enable researchers to quickly get a sense of how their model scores against
standardized brain and behavior benchmarks on multiple dimensions and facilitate
comparisons to other state-of-the-art models. At the same time, new brain
data can quickly be tested against a wide range of models to determine how
well existing models explain the data.

In particular, Brain-Score Vision evaluates
the similarity to neural recordings in the primate visual areas as well as behavioral outputs,
with a score (ranging from 0 "not aligned" to 1 "aligned at noise ceiling") on these various
brain and behavioral benchmarks. This guide is a tutorial for researchers and tinkerers
alike that outlines the setup, submission, and common issues for users.


Quickstart
==========
In this section, we will provide a quick and easy way
to get your model(s) ready for submission. This is mainly for those who do not have the time to read
or do the whole tutorial, or for those who just want to go ahead and submit
a model quickly; however, we recommend referring back to this tutorial,
especially if you encounter errors. This section also does not
have pictures, which the other more lengthy sections below do. As an example,
we will submit a version of AlexNet from Pytorch’s library; the main steps are outlined below:

1. Make a new directory in ``brainscore_vision/models``, e.g. ``brainscore_vision/models/mymodel``.
   We refer to this as a new *model plugin*.
2. Specify the dependencies in ``brainscore_vision/models/mymodel/setup.py``.
3. In the ``brainscore_vision/models/mymodel/__init__.py``, implement the model such that it follows the :ref:`interface`
   and register it to the ``brainscore_vision.model_registry``:
   ``model_registry['myalexnet'] = lambda: ModelCommitment(identifier='myalexnet', ...)``
4. In the ``brainscore_vision/models/mymodel/test.py``, write unit tests for your model and make sure they pass locally.
   You might for instance want to test that
   ``score(model_identifier='myalexnet', benchmark_identifier='MajajHong2015public.IT-pls')`` returns a reasonable score.
5. Submit to ``brain-score.org``. You can do this by either opening a pull request on the `Github repository`,
   or by submitting a zip file with your plugin on the website.
   That’s it! Read more below to get a better idea of the process, or to help fix bugs that might come up.


Common Errors: Setup
====================

Below are some common errors that you might encounter while setting up
this project or doing this tutorial. We will add more soon!

1. When running ``pip install .``, you get a message
   from the terminal like::
     Directory '.' is not installable. Neither 'setup.py' nor 'pyproject.toml' found.
   *Cause*: Not running ``pip install .`` in the right directory:
   most likely you are in the plugin folder we created,
   and not the top-level folder containing ``brainscore_vision`` we should be in.

   *Fix*: if you are in the plugin directory ``brainscore_vision/models/mymodel``, simply run::
    cd ../../../
   and then rerun
   the ::
    pip install .
   command. This navigates to the correct top-level folder and
   installs the packages where they are supposed to be.
   More generally: make sure you are in the top-level folder containing ``brainscore_vision``
   (and not its parent or child folder) before you run the pip command above. This should fix the error.

2. After implementing a pytorch model and running ``score`` for the first time, you get::
    ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1076)
   *Cause*: Pytorch’s backend. The SSL certificate for downloading a pre-trained model has expired
   from their end and Pytorch should renew soon (usually ~4 hrs)

   *Fix*: If you can’t wait, add the following lines of code to your plugin:
   (*Note that Pycharm might throw a warning about this line)*::
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context



Common Errors: Submission
=========================

1. It has been 24 hours since I submitted my model, and I have not gotten a score? What happened?

   *Cause*: There are many issues that could cause this.

   *Fix*:  If it happens, please open an issue on ``https://github.com/brain-score/vision/issues/new``
   and we can check the logs and tell you what happened. If it is really urgent, additionally send us an email.
   You will, hopefully soon, be able to log in and check the logs yourself, so stay tuned!



Frequently Asked Questions
==========================

1. **What are all the numbers on the Brain-Score site?**

   As of now on the leaderboard (Brain-Score), there are many scores that your model would obtain.
   These are sub-divided into ``neural`` and ``behavioral`` scores which themselves are further hierarchically organized.
   Each one of these is a set of benchmarks that tests how "brain-like"
   your model is to various cognitive and neural data -- in essence,
   it is a measure of how similar the model is to the brain's visual system.
   Models are also tested on "Engineering" benchmarks which do not include biological data
   but typically test against ground truth, often for a machine learning benchmark.
   These are often to the brain and behavioral scores (e.g. more V1-like → more robust to image perturbations).

2. **What is the idea behind Brain-Score? Where can I learn more?**

   The website is a great place to start, and for those who want to dive deep,
   we recommend reading the `perspective paper`_ and the `technical paper`_
   that outline the idea and the inner workings of how Brain-Score operates.

3. **I was looking at the code and I found an error in the code/docs/etc. How can I contribute?**

   The easiest way would be to fork the repository
   (make a copy of the Brain-Score `Github repository` locally and/or in your own Github),
   make the necessary edits there,
   and submit a pull request (PR) to merge it into our master branch.
   We will have to confirm that PR, and thank you for contributing!

4. **I really like Brain-Score, and I have some ideas that I would love to
   talk to someone about. How do I get in touch?**

   Make an issue ``https://github.com/brain-score/vision/issues/new``, or send us an email!
   We will also be creating a mailing list soon, so stay tuned.

5. **Is there any reward for reaching the top overall Brain-Score? Or even a top
   score on the individual benchmarks?**

   We sometimes run competitions (e.g. ``https://www.brainscoreworkshop.com/``).
   A top Brain-Score result is also a great way to show the goodness of your model and market its value to the community.
