# BP-neural-network
a really simple version

- just 3 layers, **input layer, hiden layer, output layer**
- just **sigmoid** activaation function
- download train data and test data from [kaggle](https://www.kaggle.com/c/3004/download/train.csv)
- you can choose **BGD** or **SGD**
- you can set **learning rate**, **loss threshold**, **batch**, **max iteration** etc.
- you can train mnist like this:
```cplus
bp_network bp(vector<int>{784,100,10});

bp.read_mnist("train.csv");
bp.data_normalization();
bp.train_SGD();
bp.predict();
```
- **1000** items to train, **200** items to test, accuracy>**0.9**
