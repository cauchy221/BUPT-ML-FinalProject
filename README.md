# BUPT-ML-FinalProject
[twitter-sentiment-analysis-self-driving-cars](https://www.kaggle.com/competitions/twitter-sentiment-analysis-self-driving-cars)

## Project Structure
formed by [Dir Tree Noter](http://dir.yardtea.cc/)
```
BUPT-ML-FinalProject
├─ README.md
├─ 设计思路.pdf
├─ data
│    ├─ preprocessed
│    │    ├─ big
│    │    └─ small
│    └─ release
│           ├─ origin.csv
│           ├─ sample.csv
│           ├─ test.csv
│           └─ train.csv
├─ data_preprocess.ipynb
├─ finetune.ipynb
├─ output
└─ research
     ├─ Twitter_Sentiment_Analysis_Using_Machine_Learning_Algorithms_A_Case_Study.pdf
     └─ 机器学习方法调研.pdf
```

## Usage
1. 在 Kaggle 中注册账号，在 Account 中找到 API，并点击 `Create New API Token`
2. 将自动下载的 `Kaggle.json` 文件移动到 `C:\Users\YourUserName\.kaggle` 文件夹中（若无请自行新建）
3. 执行命令 `kaggle competitions download -c twitter-sentiment-analysis-self-driving-cars` 下载原始数据
4. 配置环境
5. 在 `jupyter notebook` 中运行 `data_preprocess.ipynb` 进行数据预处理
6. 在 `jupyter notebook` 中运行 `finetune.ipynb` 进行训练、验证和测试

## Result
在此飞书文档中，通过 Google Driver 保存了个别模型参数对应的 `.bin` 文件
https://dd80km7uu3.feishu.cn/sheets/shtcnQVZM6cBtPYLI1AqAVEo8ue
