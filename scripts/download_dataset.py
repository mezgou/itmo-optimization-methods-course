from __future__ import annotations

import argparse

from optlib.datasets import download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Google Drive dataset by file id.")
    parser.add_argument("file_id")
    parser.add_argument("dest")
    args = parser.parse_args()
    path = download(args.file_id, args.dest)
    print(path)


if __name__ == "__main__":
    main()
