#!/usr/bin/env python

"""
This script was used to upload IMC data to Zenodo
upon the release of the manuscript as a preprint.

After peer-review, the entries were updated to match
the accepted manuscript using the `_upload_update.py` script.

"""

import sys, json, requests, hashlib
import typing as tp

import pandas as pd

# from src.analysis import get_files
# from src.types import Path

secrets_file = Path("~/.zenodo.auth.json").expanduser()
secrets = json.load(open(secrets_file))
zenodo_json = Path("metadata/zenodo.deposition.proc.json")
api_root = "https://zenodo.org/api/"
headers = {"Content-Type": "application/json"}
kws = dict(params=secrets)

dep: tp.Dict[str, tp.Any]
bucket_url: str


def main():
    global dep, bucket_url
    # Test connection
    req = requests.get(api_root + "deposit/depositions", **kws)
    assert req.ok

    # Get a new bucket or load existing
    if not zenodo_json.exists():
        req = requests.post(
            api_root + "deposit/depositions",
            json={},
            **kws,
        )
        json.dump(req.json(), open(zenodo_json, "w"))
    dep = json.load(open(zenodo_json, "r"))
    # dep = {"id": 5719157}

    # renew the metadata:
    dep = get()

    # Add metadata
    authors_meta = pd.read_csv("metadata/authors.csv")
    if (
        "creators" not in dep["metadata"]
        or len(dep["metadata"]["creators"]) != authors_meta.shape[0]
    ):
        metadata = json.load(open("metadata/zenodo_metadata.json"))
        authors = authors_meta[["name", "affiliation", "orcid"]].T.to_dict()
        authors = [v for k, v in authors.items()]
        metadata["metadata"]["creators"] = authors
        r = requests.put(
            api_root + f"deposit/depositions/{dep['id']}",
            data=json.dumps(metadata),
            headers=headers,
            **kws,
        )
        assert r.ok

    # Upload files
    bucket_url = dep["links"]["bucket"] + "/"
    # 'https://zenodo.org/api/files/aa463788-ab41-4816-9818-8d526ac43ebe/'

    files = Path("processed").glob("*/*/*.*")
    files = sorted(
        [
            f
            for f in files
            if f.as_posix().endswith(("_full.csv", "_full.tiff", "_mask.tiff"))
        ]
    )

    # # Upload TIFF files
    for file in files:
        upload(file.as_posix(), file.as_posix())

    # Upload Metadata
    dep = get()
    df = pd.DataFrame(dep["files"])
    df["filename"] = "data/" + df["filename"]
    df["download_link"] = df["links"].apply(lambda x: x["download"])
    meta = df.set_index("filename").drop("links", axis=1)
    meta.to_csv("metadata/zenodo_metadata.csv")
    upload("metadata/zenodo_metadata.csv", "metadata.csv")


def get() -> tp.Dict[str, tp.Any]:
    return requests.get(api_root + f"deposit/depositions/{dep['id']}", **kws).json()


def get_file_md5sum(filename: str, chunk_size: int = 8192) -> str:
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def upload(file: str, target: str = None, refresh: bool = False) -> None:
    if target is None:
        target = file
    if refresh:
        exists = [x["filename"] for x in get()["files"]]
    else:
        try:
            exists = dep["existing_files"]
        except KeyError:
            exists = []
    if target in exists:
        print(f"File '{file}' already uploaded.")
        return
    print(f"Uploading '{file}'.")
    with open(file, "rb") as handle:
        r = requests.put(bucket_url + target, data=handle, **kws)
    assert r.ok, f"Error uploading file '{file}': {r.json()['message']}."
    print(f"Successfuly uploaded '{file}'.")

    f = r.json()["checksum"].replace("md5:", "")
    g = get_file_md5sum(file)
    assert f == g, f"MD5 checksum does not match for file '{file}'."
    print(f"Checksum match for '{file}'.")


def delete(file: str, refresh: bool = False) -> None:
    print(f"Deleting '{file}'.")
    if refresh:
        files = get()["files"]
    else:
        files = dep["files"]
    file_ids = [f["id"] for f in files if f["filename"] == file]
    # ^^ this should always be one but just in case
    for file_id in file_ids:
        r = requests.delete(
            api_root + f"deposit/depositions/{dep['id']}/files/{file_id}", **kws
        )
    assert r.ok, f"Error deleting file '{file}', with id '{file_id}'."


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
