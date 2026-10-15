# Week 3 - Text Classification (20 Newsgroups)

1. `pip install scikit-learn pandas numpy matplotlib python-docx`
2. `python text_classification.py`   -> trains/evaluates 4 models, writes `outputs/results.json` + 4 figures
3. `python build_report.py`          -> writes `outputs/Week3_Report.docx`, `Week3_Report.pdf` (needs LibreOffice, else Save As PDF in Word) and `description.txt`

Use your own data: `python text_classification.py --csv my_data.csv` (columns: `text,label`).
Google Colab: upload both .py files, then run `!pip install python-docx` and the two commands above with `!python`.

## Advanced version
After step 2, run `python text_classification_advanced.py` (adds feature ablation, stacking ensemble, repeated CV, bootstrap CI, McNemar test, ROC-AUC, PR + learning curves).
Options: `--all20` (all 20 classes), `--embeddings` (Sentence-BERT + LR; needs `pip install sentence-transformers scipy`).
Then run `python build_report.py` - Section 7.4 is added automatically.
