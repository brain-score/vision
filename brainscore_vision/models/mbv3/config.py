"""Remote host configuration."""
from os import getenv, path
from dotenv import load_dotenv


# Load environment variables from .env
basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, ".env"))

# Read environment variables
host = getenv("REMOTE_HOST")
username = getenv("REMOTE_USERNAME")
password = getenv("REMOTE_PASSWORD")
ssh_key_filepath = getenv("SSH_KEY_FILEPATH")
remote_path = getenv("REMOTE_PATH")

local_file_directory = 'data'
