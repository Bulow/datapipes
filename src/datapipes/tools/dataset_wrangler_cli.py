import argparse
from typing import Optional
from pathlib import Path
from datapipes.tools import dataset_wrangler


def main(argv: Optional[list[str]] = None) -> int:
    print(f"{__name__}: {argv = }")

    parser = argparse.ArgumentParser(
        prog="DatasetWrangler",
        description="Compress, and safely move datasets",
    )
    parser.add_argument(
        "--in",
        dest="default_input_dir",
        default=str(Path.home()),
        help="Default input folder (optional).",
    )
    parser.add_argument(
        "--out",
        dest="default_output_dir",
        default=str(Path.home() / "Datasets"),
        help="Default output folder (optional).",
    )


    args = parser.parse_args(argv or [])
    

    dataset_wrangler(
        default_input_dir=args.default_input_dir,
        default_output_dir=args.default_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
