# Python Manga OCR

## Get data
You can get the ブラックジャックによろしく manga [here](https://densho810.com/free/).\
Unzip them and use the `utils/pdf_to_imgs.py` script to get images (you need to install dependencies for that).

## Installation
### Install the requirements

## Usage
```
python manga_ocr.py <path to input image> -o <path to output image>
```
Example:
```
python manga_ocr.py data/manga_page_hd.jpeg -o data/ocr_result.png
```

Result:
| Input | Output |
|    :---:      |     :---:     |
| ![input](/data/manga_page_hd.jpeg?raw "Manga page") | ![output](/data/ocr_result.png?raw "OCR output")|



## TODOs
- [ ] The RLSA implementation follows the paper, but the paper was made for horizontal text.


## Misc
Took some ideas/code from:
- https://github.com/Kocarus/Manga-Translator-TesseractOCR/blob/master/locate_bubbles.py
- https://github.com/johnoneil/MangaTextDetection

Not yet but maybe later:
- https://github.com/leminhyen2/Sugoi-Manga-OCR/blob/main/backendServer/removeFurigana.py
