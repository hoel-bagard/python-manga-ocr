# Python Manga OCR

## Installation
### Install Tesseract
#### Arch
On arch you can use:
```
sudo pacman -S tesseract python-pytesseract tesseract-data-jpn
```

This will install tesseract and the basic Japanese data.\
However it will not install the vertical trained data. You can get it (and install it) with one of the two following commands. I'm currently using the second one.
```
sudo wget https://github.com/tesseract-ocr/tessdata/raw/main/jpn_vert.traineddata -P /usr/share/tessdata/
sudo wget https://github.com/tesseract-ocr/tessdata_best/raw/main/jpn_vert.traineddata -P /usr/share/tessdata/
```
