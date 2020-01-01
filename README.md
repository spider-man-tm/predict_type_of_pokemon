Name: Predicting type of Pokemon
====

### Overview
- Neural networks of predicting Pokemon types

### Usage
```bash
# Image collection
# -t: Selection of directory
# -g: Selection of Pokemon's generation
python img_search.py -t test -g 1
python img_search.py -t train -g 2
python img_search.py -t train -g 3
...

# make csv
python make_csv.py

# run train
python train.py

# run eval
python eval.py
```

### Install
```
pip install torch==1.1.0
pip install efficientnet_pytorch==0.5.1
pip install albumetations==0.4.2
```

### Pick Up Figure
#### F1 Score

![F1 Score](https://github.com/spider-man-tm/predict_type_of_pokemon/blob/master/export_figure/f1_score.png)

#### Predict of Spider man

|タイプ|出力
|---|---
|はがね |52%
|あく　|26%
|エスパー |23%
|むし　|8%
|水 |1%

![Spiderman](https://github.com/spider-man-tm/readme_figure/blob/master/spider-man.jpg)