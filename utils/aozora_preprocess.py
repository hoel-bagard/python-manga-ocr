import shutil
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description="Script to remove extra information from aozora data.")
    parser.add_argument("data_path", type=Path, help="Path to the folder with the txts.")
    parser.add_argument("--output_path", "-o", type=Path,
                        help="Path to the output folder. defaults to './data/preprocessed_text'.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_dir: Path = args.output_path if args.output_path else Path('./data/preprocessed_text')

    txt_files_paths = list(data_path.rglob("*.txt"))
    nb_files = len(txt_files_paths)

    for i, txt_path in enumerate(txt_files_paths, start=1):
        msg = f"Processing image {txt_path.name}    ({i}/{nb_files})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        rel_path = txt_path.relative_to(data_path)
        output_path = output_dir / rel_path.stem
        output_path.mkdir(exist_ok=True, parents=True)

        with open(txt_path, "r") as f:
            content = f.read()

        # TODO regex
        # ［＃ここから２字下げ］
        # Until second -------------------------------------------------------
        # After 底本：「


if __name__ == "__main__":
    main()
