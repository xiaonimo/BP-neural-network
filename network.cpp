#include "bp_network.h"
#include "mat.h"


bp_network::bp_network(vector<int> l) {
    assert(l.size() == 3);//目前是输入层，一个隐含层，输出层。
    layers_info = l;
    w1 = Mat(l[0], l[1], 0);
    w2 = Mat(l[1], l[2], 0);

    b1 = Mat(1, l[1], 0);
    b2 = Mat(1, l[2], 0);

    d_w1 = Mat(l[0], l[1], 0);
    d_w2 = Mat(l[1], l[2], 0);

    d_b1 = Mat(1, l[1], 0);
    d_b2 = Mat(1, l[2], 0);

    init_weights();

    i_input = Mat(1, l[0], 0);
    i_output = Mat(1, l[0], 0);

    h_input = Mat(1, l[1], 0);
    h_output = Mat(1, l[1], 0);

    o_input = Mat(1, l[2], 0);
    o_output = Mat(1, l[2], 0);

    sigmoid_table = vector<double>(4000, 0);
    create_sigmoid_table();
}

void bp_network::init(Mat& b, int seed) {
    mt19937 gen(seed);
    normal_distribution<double> normal(0, 0.01);

    for (int i=0;i<b.rows;++i) {
        for (int j=0;j<b.cols;++j) b.m[i][j] = normal(gen);
    }
}

void bp_network::init_weights() {
    init(w1, 1);
    init(w2, 2);
    init(b1, 3);
    init(b2, 4);
}

void bp_network::forward_flow() {
    //输入层没有神经元
    i_input = X;
    i_output = i_input;

    //输入层->隐含层
    h_input = i_output*w1 + b1;

    //隐含层激活函数
    h_output = sigmoid(h_input);

    //隐含层->输出层
    o_input = h_output*w2 +b2;

    //输出层激活函数
    o_output = sigmoid(o_input);
}

void bp_network::set_X(vector<double> inp) {
    assert(int(inp.size()) == layers_info[0]);
    X = Mat(inp);
}

void bp_network::set_Y(vector<double> outp) {
    assert(int(outp.size())==layers_info.back());
    Y = Mat(outp);
}

void bp_network::show_output() {
    cout << "o_output is:" <<endl;
    o_output.print();
}

void bp_network::backword_flow() {
    //第二层d_d2  (y'-y)*f'(net)
    //第二层d_w2  (y'-y)*f'(net)*x
    d_b2 = (o_output-Y).mul(o_output.mul(o_output*(-1) + 1));
    d_w2 = h_output.inverse()*d_b2;

    //第一层d_d1
    //第一层d_w1
    Mat tmp1 = o_output-Y;

    Mat tmp2 = o_output.mul(o_output*(-1)+1);

    Mat tmp3 = w2*(tmp1.mul(tmp2)).inverse();

    Mat tmp4 = h_output.mul(h_output*(-1)+1);

    d_b1 = tmp3.inverse().mul(tmp4);

    d_w1 = i_output.inverse()*d_b1;
}

void bp_network::update_weights() {
    w1 = w1-(d_w1*step);
    w2 = w2-(d_w2*step);
    b1 = b1-(d_b1*step);
    b2 = b2-(d_b2*step);
}


void bp_network::train_SGD(double _loss, int _itr, double _step, int _max_itr_batch) {
    step = _step;
    min_loss = _loss;
    max_itr_all = _itr;
    max_itr_batch = _max_itr_batch;

    default_random_engine gen;
    uniform_int_distribution<int> r(0, train_X.rows-1);//用于SGD过程中，随机生成样本

    for (int k=0; k<max_itr_all; ++k) {
        for (int i=0; i<train_X.rows; ++i) {
            int _index = r(gen);
            set_X(train_X.m[_index]);
            set_Y(train_Y.m[_index]);

            train_loss();
            cout << "SGD\tcur_itr_batch(" << i << "/" << train_X.rows << ")\tcur_itr_all(" << k << "/" << max_itr_all << ")\tloss:" << cur_loss << endl;
        }
    }
}

void bp_network::train_BGD(int _batch, double _loss, int _itr, double _step, int _max_itr_batch) {
    batch = _batch;
    min_loss = _loss;
    max_itr_all = _itr;
    step = _step;
    max_itr_batch = _max_itr_batch;

    int batches = train_X.rows/batch+1;
    for (int k=0; k<max_itr_all; ++k) {//总迭代次数
        for (int i=0; i<batches; ++i) {//batch
            cur_loss = DBL_MAX_10_EXP;
            max_itr_batch = 100;
            int __itr = 0;
            while (cur_loss > _loss || __itr <max_itr_batch) {//batch内迭代
                __itr++;//保证最小迭代数
                Mat loss_mat(1, layers_info[2], 0);//batch范围内的loss值
                Mat batch_d_w1(layers_info[0], layers_info[1], 0);
                Mat batch_d_b1(1, layers_info[1], 0);
                Mat batch_d_w2(layers_info[1], layers_info[2], 0);
                Mat batch_d_b2(1, layers_info[2], 0);

                for (int j=i*batch; j<(i+1)*batch && j<train_X.rows; ++j) {
                    set_X(train_X.m[j]);
                    set_Y(train_Y.m[j]);

                    forward_flow();
                    loss_mat = loss_mat + (o_output-Y).square();
                    backword_flow();

                    batch_d_w1 = batch_d_w1 + d_w1/batch;
                    batch_d_b1 = batch_d_b1 + d_b1/batch;
                    batch_d_w2 = batch_d_w2 + d_w2/batch;
                    batch_d_b2 = batch_d_b2 + d_b2/batch;
                }
                cur_loss = loss_mat.sum()/batch;
                d_w1 = batch_d_w1;
                d_b1 = batch_d_b1;
                d_w2 = batch_d_w2;
                d_b2 = batch_d_b2;
                update_weights();
            }
            cout << "BGD\tcur_itr_batch(" << i << "/" << batches << ")\tcur_itr_all(" << k << "/" << max_itr_all << ")\tloss:" << cur_loss << endl;
        }
    }
}

void bp_network::train_itr() {
    cur_itr_batch = 1;

    while (cur_itr_batch < max_itr_batch) {
        forward_flow();
        backword_flow();
        update_weights();
    }
}

void bp_network::train_loss() {
    cur_itr_batch = 0;
    cur_loss = DBL_MAX_10_EXP;

    while (cur_loss > min_loss && cur_itr_batch<max_itr_batch) {
        cur_itr_batch++;
        forward_flow();
        backword_flow();
        update_weights();
        cur_loss = (o_output-Y).square().sum();
    }
}

void bp_network::predict() {
    int correct_answer = 0;
    for (int i=0; i<test_X.rows; ++i) {
        set_X(test_X.m[i]);
        forward_flow();
        int pred_res = argmax(o_output.m[0]);
        int real_res = argmax(test_Y.m[i]);

        correct_answer += (pred_res==real_res);

        cout << i << "\t" << pred_res << "/" << real_res << "\t" <<correct_answer/double(i+1) << endl;
    }
}


