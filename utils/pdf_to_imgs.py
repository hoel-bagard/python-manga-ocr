import shutil
from argparse import ArgumentParser
from pathlib import Path

try:
    from pdf2image import convert_from_path
except ModuleNotFoundError:
    print("To use this script you need to pip install 'pdf2image'.")
    print("Note: you might need to install other dependencies, check the package's github page.")
    exit()


def name_generator(padding=4):
    """Generate the image file names for pdf2image (not sure what it's expecting, this is good enough)."""
    while True:
        yield "page"


def main():
    parser = ArgumentParser(description="Script to convert manga pdfs to jpgs.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset (can contain multiple pdfs).")
    parser.add_argument("output_path", type=Path, help="Path to the output folder.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_dir: Path = args.output_path

    pdf_files_paths = list(data_path.rglob("*.pdf"))
    nb_files = len(pdf_files_paths)

    for i, pdf_path in enumerate(pdf_files_paths, start=1):
        msg = f"Processing image {pdf_path.name}    ({i}/{nb_files})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        rel_path = pdf_path.relative_to(data_path)
        output_path = output_dir / rel_path.stem
        output_path.mkdir(exist_ok=True, parents=True)
        images = convert_from_path(pdf_path, dpi=200, output_folder=output_path, output_file=name_generator())  # noqa


if __name__ == "__main__":
    main()
