import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

import brainio_collection
from brainio_collection.fetch import BotoFetcher


def list_public_assemblies():
    all_assemblies = brainio_collection.list_assemblies()
    all_assemblies = ['movshon.FreemanZiemba2013.private'] + all_assemblies
    public_assemblies = []
    for assembly in all_assemblies:
        access = True
        # https://github.com/brain-score/brainio_collection/blob/a7a1eed2afafa0988d2b9da76091b3f61942e4d1/brainio_collection/fetch.py#L208
        assy_model = brainio_collection.assemblies.lookup_assembly(assembly)
        for store_map in assy_model.assembly_store_maps:
            probe_fetcher = _ProbeBotoFetcher(location=store_map.assembly_store_model.location,
                                              unique_name=store_map.assembly_store_model.unique_name)
            if not probe_fetcher.has_access():
                access = False
                break
        if access:
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
