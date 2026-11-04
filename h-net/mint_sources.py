"""Write the sources file that prepare_bytes.py reads: links that expire to language-model's v6 corpora in R2, with
each file's SHA-256 from its corpus's manifest. Run it under language-model's runs/storage-control/r2.py, which gives
the credentials. It prints only the files' keys and sizes, and writes the file readable by its owner alone.

Usage: python h-net/mint_sources.py OUTPUT [--expires SECONDS]
"""

import argparse
import hashlib
import json
import os

import boto3
from botocore.config import Config

PREFIX = "data/pretrain-v6"
FILES = {"stable": ("train_ja", "train_en", "validation_ja", "validation_en"), "decay": ("train_ja", "train_en")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    parser.add_argument("--expires", type=int, default=7200, help="seconds the links last")
    arguments = parser.parse_args()
    s3 = boto3.client("s3", endpoint_url=os.environ["LM_R2_ENDPOINT"], region_name="auto",
                      config=Config(signature_version="s3v4"))
    bucket = os.environ["LM_R2_BUCKET"]

    def link(key):
        return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                         ExpiresIn=arguments.expires)

    def read(key):
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    sources = {}
    manifests = {phase: json.loads(read(f"{PREFIX}/corpus-{phase}/manifest.json")) for phase in FILES}
    for phase, names in FILES.items():
        sources[phase] = {}
        for name in names:
            key = f"{PREFIX}/corpus-{phase}/{name}.bin"
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            sources[phase][name] = {"url": link(key), "sha256": manifests[phase]["files"][f"{name}.bin"]["sha256"]}
            print(json.dumps({"key": key, "bytes": size}))
    key = f"{PREFIX}/corpus-stable/tokenizer.json"
    digest = hashlib.sha256(read(key)).hexdigest()
    if digest != manifests["stable"]["tokenizer_sha256"]:
        raise SystemExit("the tokenizer in R2 is not the one the stable corpus names")
    sources["tokenizer"] = {"url": link(key), "sha256": digest}
    print(json.dumps({"key": key, "sha256_matches_manifest": True}))
    descriptor = os.open(arguments.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(sources, output)


if __name__ == "__main__":
    main()
