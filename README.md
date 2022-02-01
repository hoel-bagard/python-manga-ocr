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

## TODO
Try to use tesseract directly instead of pytesseract ?


## Misc
Took some ideas/code from:\
https://github.com/Kocarus/Manga-Translator-TesseractOCR/blob/master/locate_bubbles.py\
https://github.com/johnoneil/MangaTextDetection\
https://github.com/johnoneil/MangaTextDetection/blob/master/clean_page.py\


Not yet but maybe later:\
https://github.com/leminhyen2/Sugoi-Manga-OCR/blob/main/backendServer/removeFurigana.py
