#include "bp_network.h"

void bp_network::read_mnist(string filename) {
    int train_data_num = 2000;
    int test_data_num = 200;
    train_X = Mat(train_data_num, 784, 0);
    train_Y = Mat(train_data_num, 10, 0);
    test_X = Mat(test_data_num, 784, 0);
    test_Y = Mat(test_data_num, 10, 0);

    ifstream fin(filename);

    for (int i=0; i<train_data_num; ++i) {
        string line;
        getline(fin, line);
        istringstream lin(line);
        string str;
        int j = 0;
        while (getline(lin, str, ',')) {
            double val = 0;
            stringstream ss(str);
            ss >> val;
            if (!j) {//第一列是label
                train_Y.m[i][val] = 1;
                j++;
            } else {
                train_X.m[i][j-1] = val;
                j++;
            }
        }
    }

    for (int i=0; i<test_data_num; ++i) {
        string line;
        getline(fin, line);
        istringstream lin(line);
        string str;
        int j = 0;
        while (getline(lin, str, ',')) {
            double val = 0;
            stringstream ss(str);
            ss >> val;
            if (!j) {//第一列是label
                test_Y.m[i][val] = 1;
                j++;
            } else {
                test_X.m[i][j-1] = val;
                j++;
            }
        }
    }
    fin.close();
    cout << "read mnist data finished" << endl;
}

int bp_network::argmax(vector<double> x) {
    assert(x.size() > 0);
    double max_val = x[0];
    int index = 0;
    for (int i=1; i<int(x.size()); ++i) {
        if (x[i] < max_val) continue;
        max_val = x[i];
        index = i;
    }
    return index;
}

void bp_network::data_normalization() {
    train_X = train_X/255.0;
    test_X = test_X/255.0;
    cout << "data nomallized finished" <<endl;
}

void bp_network::create_sigmoid_table() {
    double _step = 40.0/4000.0;
    for (int i=0; i<4000; ++i) {
        sigmoid_table[i] = 1.0/(1.0+exp(20.0-i*_step));
        //cout <<setprecision(20)<< i << "\t" << sigmoid_table[i] <<endl;
    }
    cout << "create sigmoid table finished" <<endl;
}

double bp_network::sigmoid(double x) {
    if (x<=-20) return 0.00000000000001;
    else if (x>=20) return 0.99999999999999;
    else {
        return sigmoid_table[(x+20)*100];
    }
}

Mat bp_network::sigmoid(Mat b) {
    Mat res(b.rows, b.cols, 0);
    for (int i=0; i<b.rows; ++i ) {
        for (int j=0; j<b.cols; ++j) {
            res.m[i][j] = sigmoid(b.m[i][j]);
        }
    }
    return res;
}

