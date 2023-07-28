import logging

import boto3
from botocore.exceptions import ClientError

_logger = logging.getLogger(__name__)


class UniqueKeyDict(dict):
    def __init__(self, reload=False, **kwargs):
        super().__init__(**kwargs)
        self.reload = reload

    def __setitem__(self, key, *args, **kwargs):
        if key in self:
            raise KeyError("Key '{}' already exists with value '{}'.".format(key, self[key]))
        super(UniqueKeyDict, self).__setitem__(key, *args, **kwargs)

    def __getitem__(self, item):
        value = super(UniqueKeyDict, self).__getitem__(item)
        if self.reload and hasattr(value, 'reload'):
            _logger.warning(f'{item} is accessed again and reloaded')
            value.reload()
        return value


def get_secret(secret_name, region_name='us-east-2'):
    session = boto3.session.Session()
    _logger.info("Fetch secret from secret manager")
    try:
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name,
        )
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            _logger.error("The requested secret " + secret_name + " was not found")
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            _logger.error("The request was invalid due to:", e)
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            _logger.error("The request had invalid params:", e)
    except Exception as e:
        _logger.error("The request failed with:", e)
    else:
        # Secrets Manager decrypts the secret value using the associated KMS CMK
        # Depending on whether the secret was a string or binary, only one of these fields will be populated
        _logger.info(f'Secret {secret_name} successfully fetched')
        if 'SecretString' in get_secret_value_response:
            _logger.info("Inside string response...")
            return get_secret_value_response['SecretString']
        else:
            _logger.info("Inside binary response...")
            return get_secret_value_response['SecretBinary']

