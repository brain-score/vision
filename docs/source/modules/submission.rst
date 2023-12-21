Submission System
------------------

To submit a model or benchmark to the Brain-Score leaderboard, Brain-Score provides an automated submission system.
This allows us to publicly score new models on existing benchmarks and score existing models against new benchmarks.

.. image:: docs/source/modules/brainscore_submission.png
    :width: 400px
    :align: center
    :height: 200px
    :alt: submission system diagram

Users can submit models or benchmarks as a zip file via the `Brain-Score website <#Brain-Score Website>`_ or as a pull request via GitHub.
Appropriately configured zip file submissions are converted by `Jenkins <#Jenkins>`_ (`create_github_pr`) into a pull request. 
If tests pass, the PR is automatically merged into the codebase by a GitHub Action (`automerge_plugin-only_prs`).
If tests fail on a website submission, the submitter will be notified via email.
Note: While we currently do not support private code submissions, this feature is under development.

If a newly merged PR contains new models or benchmarks,
a separate Github Action (`score_new_plugins`) triggers the scoring runs which are handled by Jenkins
and run on the `OpenMind compute cluster <#Openmind>`_.


Submission System Components
############################

- **Brain-Score Website**:

    The `website <www.brain-score.org>`_ (`GitHub <https://github.com/brain-score/brain-score.web>`_) handles our front 
    end. It is implemented using Django and also accesses the database instance.

    The website is hosted via Amazon Elastic Beanstalk. There are two instances on AWS EB:

    - `brain-score-web-dev <brain-score-web-dev.us-east-2.elasticbeanstalk.com>`_: This is our dev website environment.
      It is the same as prod but uses the dev database.

    - `brain-score-web-prod <brain-score-web-prod.us-east-2.elasticbeanstalk.com>`_: Our production branch
        that uses the prod database. This is what end users see, as the above URL is forwarded to brain-score.org
- **Jenkins**:

    Jenkins is a continuous integration tool, which we use to automatically run 
    project unittests and the scoring process for models of the brain. `Jenkins is running on Braintree <http://braintree.mit.edu:8080/>`_ 
    - DiCarlo lab's internal server. Jenkins defines different jobs and executes different tasks. The task for a new submission is
    triggered via the website and the unittest tasks are triggerd by GitHub web hooks. Once the jobs are triggered,
    Jenkins runs a procedure to execute the tests or scoring and communicate the results back to the user (via email)
    or back to GitHub.

-  **OpenMind**

    As scoring submissions is a computationally and memory expensive process, we cannot execute model scoring on small
    machines. We submit jobs to Openmind, a computer cluster operated by MIT BCS. The big advantage of Openmind is its
    queuing system, which allows to define detailed resource requirements. Jobs are executed once their requested
    resources are available. The Jenkins related contents are stored on ``/om2/group/dicarlo/jenkins``. This directory
    contains a script for model submission (`score_model.sh`), for unittests (`unittests_brainscore.sh`), and for plugin-specific
    tests (`unittests_plugins.sh`). The scripts are executed in Openmind jobs and are responsible for fully installing a conda 
    environment, executing the process, and shutting everything down again. For scoring runs, results are stored in the database 
    and sent to the user via email. For unit and plugin tests, results are reported back to Github.


- **Postgres database:**

    Our database, hosted on Amazon AWS, contains all displayed score and submission data, along with much more user and
    model data/metadata. Our AWS account contains three database instances:
     
    - Prod (brainscore-prod-ohio-cred): This database is used in production mode, containing real user's data. This
      database should not be altered for development until features have been tested and vetted on Dev.
    - Dev (brainscore-1-ohio-cred): A development database, which can be used to develop new database dependent
      features. Nothing will break when the database schema here is changed; it is periodically updated to match Prod.
    - Test (brainscore-ohio-test): The database used for executing tests. Jenkins also executes unittests of all
      Brain-Score projects and should use this database for testing.

    The names in parantheses are used in brain-score to load database credentials for the different databases.
    Just change the name and another database is used. Databases are automatically snapshotted every 7 days, and
    devs can restore snapshots at any time.






What to do
#################


...when changing the database schema
************************************
The current schema is depicted `here
<https://github.com/brain-score/brain-score/blob/master/brainscore_vision/docs/source/modules/db_schema.uml>`_.


When the database schema has to be changed, use the `Brain-Score.web <https://github.com/brain-score/brain-score.web>`_
project, along with django commands, to adjust the tables (in `benchmark/models.py`). The schema also has to be updated
in `core <https://github.com/brain-score/core/blob/main/brainscore_core/submission/database_models.py>`_. Once changes
are made locally, follow `these instructions <https://github.com/brain-score/brain-score.web/blob/master/deployment.md#to-deploy>`_ to
apply those migrations to the correct databases. All needed changes to the database (dev or prod) should be done with
Django via migrations. During development, work with the dev database (secret `brainscore-1-ohio-cred`); when your
tests pass on the test database (`brainscore-ohio-test`) they are ready for the PR. Once the PR is approved and test
cases run, the PR can be merged. Finally, apply those migrations to the prod database via the link above.


...changing the submission process
**********************************
In addition to the main job for scoring submission (`score_plugins`), Jenkins contains a second job (`dev_score_plugins`),
which can be used to test new submission code. It is also a good idea instead of checking out the Brain-Score master
branch, as it is done in the default job, to checkout your development branch instead. This way you can run a whole
submission without harming the "production" job. This is accomplished already by a duplicate of score_models.sh for dev,
aptly named dev_score_models.sh. That script is what is run on Jenkins's dev environment. Once the development job runs
successfully, the code can be merged to master and will be run "in production".

Scoring Process Description
###########################
For scoring submitted files, we install the Brain-Score framework on Openmind and run the scoring process.