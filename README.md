# BUPT-ML-FinalProject
[twitter-sentiment-analysis-self-driving-cars](https://www.kaggle.com/competitions/twitter-sentiment-analysis-self-driving-cars)

## Project Structure
formed by [Dir Tree Noter](http://dir.yardtea.cc/)
```
BUPT-ML-FinalProject
├─ README.md
├─ requirements.txt
├─ data
│     ├─ finetune
│     ├─ preprocessed
│     │     ├─ test.csv
│     │     └─ train.csv
│     └─ release
│          ├─ sample.csv
│          ├─ test.csv
│          └─ train.csv
├─ data_preprocess.ipynb
├─ finetune.py
└─ finetune.sh
```

## Usage
1. 在 Kaggle 中注册账号，在 Account 中找到 API，并点击 `Create New API Token`
2. 将自动下载的 `Kaggle.json` 文件移动到 `C:\Users\YourUserName\.kaggle` 文件夹中（若无请自行新建）
3. 执行命令 `kaggle competitions download -c twitter-sentiment-analysis-self-driving-cars` 下载原始数据
4. 执行命令 `pip install -r requirements.txt` 配置环境
5. 在 `jupyter notebook` 中运行 `data_preprocess.ipynb` 进行数据预处理
6. 执行命令 `bash finetune.sh` 进行预训练
