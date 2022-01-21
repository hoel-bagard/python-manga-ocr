# Python Manga OCR

## Installation
### Install Tesseract
#### Arch
On arch you can use:
```
sudo pacman -S tesseract python-pytesseract tesseract-data-jpn
```
This will install tesseract. However it will not install the vertical trained data. You can get it (and install it) with:
```
sudo wget https://github.com/tesseract-ocr/tessdata/raw/main/jpn_vert.traineddata -P /usr/share/tessdata/
```

Notes (from the old project):\
The trained data for Japanese is also available [here](https://github.com/tesseract-ocr/tessdata/raw/4.00/jpn.traineddata) if needed. The list of all the trained data is [here](https://tesseract-ocr.github.io/tessdoc/Data-Files.html).
The current trained data is from [here](https://github.com/tesseract-ocr/tessdata_fast/blob/main/script/Japanese.traineddata) (rename it to jpn.traineddata).
