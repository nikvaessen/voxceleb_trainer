#!/usr/bin/python
# -*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

import argparse
import multiprocessing
import os
import pathlib
import subprocess
import pathlib
import pdb
import hashlib
import time
import glob
import tarfile
import threading

from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile

## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description="VoxCeleb downloader")

parser.add_argument("--save_path", type=str, default="data", help="Target directory")
parser.add_argument("--user", type=str, default="user", help="Username")
parser.add_argument("--password", type=str, default="pass", help="Password")

parser.add_argument(
    "--download", dest="download", action="store_true", help="Enable download"
)
parser.add_argument(
    "--extract", dest="extract", action="store_true", help="Enable extract"
)
parser.add_argument(
    "--convert", dest="convert", action="store_true", help="Enable convert"
)
parser.add_argument(
    "--augment",
    dest="augment",
    action="store_true",
    help="Download and extract augmentation files",
)

args = parser.parse_args()


## ========== ===========
## MD5SUM
## ========== ===========
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


## ========== ===========
## Download with wget
## ========== ===========
def download(args, lines):
    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split("/")[-1]

        path = pathlib.Path(args.save_path) / outfile

        ## Download files
        out = subprocess.call(
            f"wget {url} --user {args.user} --password {args.password} -O {str(path)}",
            shell=True,
        )
        if out != 0:
            raise ValueError(
                "Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website."
                % url
            )

        ## Check MD5
        md5ck = md5(str(path))
        if md5ck == md5gt:
            print("Checksum successful %s." % outfile)
        else:
            raise Warning("Checksum failed %s." % outfile)


## ========== ===========
## Concatenate file parts
## ========== ===========
def concatenate(args, lines):
    for line in lines:
        infile = line.split()[0]
        outfile = line.split()[1]
        md5gt = line.split()[2]

        infile_path = pathlib.Path(args.save_path) / infile
        outfile_path = pathlib.Path(args.save_path) / "concat" / outfile
        outfile_path.parent.mkdir(parents=True, exist_ok=True)

        ## Concatenate files
        out = subprocess.call(
            f"cat {infile_path} > {outfile_path}",
            shell=True,
        )

        ## Check MD5
        md5ck = md5(str(outfile_path))
        if md5ck == md5gt:
            print("Checksum successful %s." % outfile)
        else:
            raise Warning("Checksum failed %s." % outfile)


## ========== ===========
## Extract zip files
## ========== ===========
def full_extract(args, fname):
    print("Extracting %s" % fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(args.save_path)
    elif fname.endswith(".zip"):
        path = pathlib.Path(fname)
        with ZipFile(fname, "r") as zf:
            zf.extractall(args.save_path)


## ========== ===========
## Partially extract zip files
## ========== ===========
def part_extract(args, fname, target):
    print("Extracting %s" % fname)
    with ZipFile(fname, "r") as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, args.save_path)
            # pdb.set_trace()
            # zf.extractall(args.save_path)


## ========== ===========
## Convert
## ========== ===========
def convert_file(fname):
    outfile = fname.replace(".m4a", ".wav")
    out = subprocess.call(
        f"ffmpeg -y -i {str(fname)} -ac 1 -vn -acodec pcm_s16le -ar 16000 {str(outfile)} >/dev/null 2>/dev/null",
        shell=True,
    )
    if out != 0:
        raise ValueError(f"Conversion failed {str(fname)}")


def convert(args):
    files = pathlib.Path(args.save_path).rglob("*.m4a")
    files = [f for f in files]
    files = sorted(files)

    print(f"Converting {len(files)} files from AAC to WAV")
    with tqdm(total=len(files)) as pbar, multiprocessing.Pool(8) as workers:
        for file in files:
            workers.apply_async(
                convert_file,
                args=(str(file),),
                error_callback=lambda x: print(x),
                callback=lambda _: pbar.update(1),
            )

        workers.close()
        workers.join()


## ========== ===========
## Split MUSAN for faster random access
## ========== ===========
def split_musan(args):
    files = glob.glob("%s/musan/*/*/*.wav" % args.save_path)

    audlen = 16000 * 5
    audstr = 16000 * 3

    for idx, file in enumerate(files):
        fs, aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace("/musan/", "/musan_split/"))[0]
        os.makedirs(writedir)
        for st in range(0, len(aud) - audlen, audstr):
            wavfile.write(writedir + "/%05d.wav" % (st / fs), fs, aud[st : st + audlen])

        print(idx, file)


## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":

    if not os.path.exists(args.save_path):
        raise ValueError("Target directory does not exist.")

    f = open("lists/fileparts.txt", "r")
    fileparts = f.readlines()
    f.close()

    f = open("lists/files.txt", "r")
    files = f.readlines()
    f.close()

    f = open("lists/augment.txt", "r")
    augfiles = f.readlines()
    f.close()

    if args.augment:
        download(args, augfiles)
        part_extract(
            args,
            os.path.join(args.save_path, "rirs_noises.zip"),
            [
                "RIRS_NOISES/simulated_rirs/mediumroom",
                "RIRS_NOISES/simulated_rirs/smallroom",
            ],
        )
        full_extract(args, os.path.join(args.save_path, "musan.tar.gz"))
        split_musan(args)

    if args.download:
        download(args, fileparts)

    if args.extract:
        concatenate(args, files)

        for file in files:
            full_extract(args, os.path.join(args.save_path, "concat", file.split()[1]))

        save_path = pathlib.Path(args.save_path)
        out = subprocess.call(
            f"mv {str(save_path/'dev'/'aac')} {str(save_path / 'aac')} && rmdir {str(save_path / 'dev')}",
            shell=True,
        )
        out = subprocess.call(
            f"mv {str(save_path / 'wav')} {str(save_path / 'voxceleb1')}", shell=True
        )
        out = subprocess.call(
            f"mv {str(save_path / 'aac')} {str(save_path / 'voxceleb2')}", shell=True
        )

    if args.convert:
        convert(args)
