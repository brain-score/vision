Given a JSON of NWB metadata formatted according to the provided template, the `DataFactory` class in `generate_data_files.py` converts the user data into a Brain-Score dataset by generating the contents of the `__init__.py`, `data_packaging.py`, and `test.py` files in the proper format, using helper functions from `create_assembly.py`, `dandi_to_stimulus_set.py`, and `extract_nwb_data.py`. 

If the user uploads a zip file of their Dandiset in addition to the JSON file, the script will read the NWB file contained in the Dandiset. If the user does not upload their Dandiset, the script will attempt to stream from the DANDI archive using the `DandiAPIClient` and the Dandiset ID provided in the JSON. 

Jenkins workflow
The `brainscore_nwb_converter` Jenkins job is triggered when the user submits their JSON value and optional local Dandiset to the NWB conversion page of the website. This job sets up the environment, including the Brain-Score DANDI API key, Jenkins API variables, and AWS environment variacles. It parses the contents of the provided JSON file and outputs a JSON string, to be passed in when it calls the DataFactory class in the EC2 instance. Once the final zip file of the Brain-Score dataset is generated, `brainscore_nwb_converter` triggers `create_github_pr`, passing in this zip file as well as the original `submission.config` from the website.

The `create_github_pr` job runs as usual, but does not add the `automerge-pr` label to the Github PR for data plugins.


User requirements:
    1. Electrode data is stored in the NWB `electrodes` field 
    2. PSTH data is stored in the NWB `scratch` field, with `PSTHs_QualityApproved_ZScored_SessionMerged` as the key
    3. JSON file contains all fields in the template, in the specified format

Steps to convert NWB data to a Brain-Score dataset (website):
    1. Make sure all user requirements above are satisifed
    2. Make sure Dandiset is either public or shared with the Brain-Score DANDI account
    3. On the Brain-Score website, navigate to the NWB conversion page from the central profile
    4. Upload a JSON file containing the fields in the example template
    5. (Optional) Upload a zip file of the Dandiset 
    6. View Brain-Score dataset on the vision Github repository


Currently, running the script locally is not supported. The script automatically generates the files in the `/home/ubuntu/vision/brainscore_vision/nwb_integration/` Jenkins directory, but this can be addressed by requiring an `exp_path` field in the JSON provided by the user if they choose to submit a local Dandiset. Also, the `user_json` parameter passed in must be in the form of a JSON string. 

Steps to convert NWB data to a Brain-Score dataset (local):
    1. Follow the Brain-Score tutorial for environment setup
    2. Set up `DANDI_API_KEY` environment variable
    3. Set up AWS credentials
    4. Create JSON containing required fields (view template on Brain-Score website)
    5. Convert JSON file into JSON string
    6. Call DataFactory class with JSON string as parameter
    7. Submit generated local zip file on Brain-Score website
