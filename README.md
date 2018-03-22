# BP-neural-network
a really simple version

- just 3 layers, input layer, hiden layer, output layer
- just sigmoid activaation function
- download train data and test data from kaggle
- you can train mnist like this:
```
bp_network a(vector<int>{784,100,10});

a.read_mnist("train.csv");
a.data_normalization();
a.train_SGD(0.01, 10, 0.01, 100);
a.predict();
```
- 1000 items to train, 200 items to test, accuracy>0.9
