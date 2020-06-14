Here the data lives. Named `input` as in Kaggle for convenience.

Data download:

```
kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification
kaggle datasets download -d kashnitsky/jigsaw-multilingual-toxic-test-translated
kaggle datasets download -d ma7555/jigsaw-train-translated
kaggle datasets download -d miklgr500/jigsaw-train-multilingual-coments-google-api
kaggle datasets download -d ludovick/jigsawtanslatedgoogle

mkdir multilingual-train
unzip jigsaw-train-multilingual-coments-google-api.zip -d ./multilingual-train/

mkdir multilingual-train-ludovick
unzip jigsawtanslatedgoogle.zip -d multilingual-train-ludovick

unzip \*.zip
rm *.zip

```