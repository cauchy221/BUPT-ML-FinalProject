# BUPT-ML-FinalProject
[twitter-sentiment-analysis-self-driving-cars](https://www.kaggle.com/competitions/twitter-sentiment-analysis-self-driving-cars)

## Project Structure
formed by [Dir Tree Noter](http://dir.yardtea.cc/)
```
BUPT-ML-FinalProject
├─ README.md
├─ data
│    ├─ preprocessed
│    │    ├─ big
│    │    │    ├─ dev.csv
│    │    │    └─ train.csv
│    │    └─ small
│    │           ├─ dev.csv
│    │           ├─ test.csv
│    │           └─ train.csv
│    └─ release
│           ├─ origin.csv
│           ├─ sample.csv
│           ├─ test.csv
│           └─ train.csv
├─ data_preprocess.ipynb
├─ finetune_v1.py
├─ finetune_v1.sh
├─ finetune_v2.ipynb
├─ output
│    ├─ big_lr1e-5
│    │    └─ submit.csv
│    ├─ big_lr2e-5
│    │    └─ submit.csv
│    └─ small_lr1e-5
│           └─ submit.csv
└─ research
       ├─ Twitter_Sentiment_Analysis_Using_Machine_Learning_Algorithms_A_Case_Study.pdf
       ├─ 机器学习方法调研.md
       └─ 机器学习方法调研.pdf
```

## Usage
1. 在 Kaggle 中注册账号，在 Account 中找到 API，并点击 `Create New API Token`
2. 将自动下载的 `Kaggle.json` 文件移动到 `C:\Users\YourUserName\.kaggle` 文件夹中（若无请自行新建）
3. 执行命令 `kaggle competitions download -c twitter-sentiment-analysis-self-driving-cars` 下载原始数据
4. 配置环境
5. 在 `jupyter notebook` 中运行 `data_preprocess.ipynb` 进行数据预处理
6. 在 `jupyter notebook` 中运行 `finetune.ipynb` 进行训练、验证和测试


## Note
- `finetune_v1.py` 代码有问题
- `random_seed=42`
- `small` 数据集非常拉，可以直接不看了


## Result
|train-eval-test|学习率|private score|public score|
|---------------|------|----------|--------------|
|8:1:1|1e-5|0.93061|0.93047|
|8:1:1|2e-5|0.93877|0.93660|
|8:1:1|3e-5|0.94489|0.93251|
|8:2:0|1e-5|NULL|NULL|
|8:2:0|2e-5|NULL|NULL|
|8:2:0|3e-5|NULL|NULL|


## TODO
1. [DONE] 按照各类数据的比例划分训练集和测试集，而不是随机划分
2. [DONE] 添加其它 [Twitter Self-Driving Car](https://data.world/crowdflower/sentiment-self-driving-cars) 情感分析数据进行扩充
3. [DONE] 进一步的数据预处理
4. 数据可视化
5. 其他方法
    - LSTM 预计能达到 [85%](https://link.springer.com/chapter/10.1007/978-981-15-5113-0_40)
    - SVM 预计能达到 [90%](https://core.ac.uk/download/pdf/144738815.pdf)
