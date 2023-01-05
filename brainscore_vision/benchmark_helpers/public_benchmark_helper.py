import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

import brainio
import logging
from brainio.fetch import BotoFetcher

_logger = logging.getLogger(__name__)


def list_public_assemblies():
    all_assemblies = brainio.list_assemblies()
    public_assemblies = []
    for assembly in all_assemblies:
        assy_model = brainio.lookup.lookup_assembly(assembly)
        if assy_model['location_type'] != 'S3':
            _logger.warning(f"Unknown location_type in assembly {assy_model}")
            continue
        probe_fetcher = _ProbeBotoFetcher(location=assy_model['location'], local_filename='probe')  # filename is unused
        if probe_fetcher.has_access():
            public_assemblies.append(assembly)
    return public_assemblies


class _ProbeBotoFetcher(BotoFetcher):
    def has_access(self):
        s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
        obj = s3.Object(self.bucketname, self.relative_path)
        try:
            # noinspection PyStatementEffect
            obj.content_length  # probe
            return True
        except ClientError:
            return False
