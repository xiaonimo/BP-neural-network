#include <iostream>
#include <vector>
#include "mat.h"
#include "bp_network.h"

int main() {
    bp_network a(vector<int>{784,100,10});

    a.read_mnist("train.csv");
    a.data_normalization();
    a.train_SGD(0.01, 10, 0.01, 100);
    a.predict();
}
