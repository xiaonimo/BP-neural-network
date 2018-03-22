#ifndef BP_NETWORK_H
#define BP_NETWORK_H

#include <vector>
#include <string>
#include <random>
#include <cfloat>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "mat.h"

using namespace std;


class bp_network {
public:
    bp_network(){}
    bp_network(vector<int>);

    //loss阈值，迭代次数，学习率
    void train_SGD(double _loss=0.1, int _itr=10, double _step=0.01, int _max_itr_batch=100);
    void train_BGD(int _batch=10, double _loss=1, int _itr=10, double _step=0.01, int _max_itr_batch=100);
    void predict();                 //预测
    void data_normalization();      //数据归一化
    void read_mnist(string);        //读取从kaggle下载的csv格式数据

public:
    Mat w1;                 //输入层->隐含层的权重矩阵
    Mat w2;                 //隐含层->输出层的权重矩阵
    Mat b1;                 //隐含层的bias
    Mat b2;                 //输出层的bias

    Mat d_w1;               //w1的梯度
    Mat d_b1;               //b1的梯度
    Mat d_w2;               //w2的梯度
    Mat d_b2;               //b2的梯度

    Mat train_X;            //训练样本
    Mat train_Y;            //训练样本的label
    Mat test_X;             //测试样本
    Mat test_Y;             //测试样本的label
    Mat X;                  //当前样本
    Mat Y;                  //当前样本的label

    int max_itr_all;        //最外层的迭代次数
    int cur_itr_batch;      //当前batch的迭代次数
    int max_itr_batch;      //每个batch的最多迭代次数
    double cur_loss;        //当前loss
    double min_loss;        //loss阈值
    int batch;              //batch大小
    double step=0.01;       //学习率

private:
    void init(Mat&, int);               //矩阵初始化
    void init_weights();                //初始化w,b
    void set_layers(vector<int>);       //设置每层神经元的个数
    void set_X(vector<double>);         //设置当前需要训练的样本
    void set_Y(vector<double>);         //设置当前需要训练的样本的label

    void show_output();                 //显示某一次迭代的结果
    void forward_flow();                //前向传播
    void backword_flow();               //反向传播
    void update_weights();              //根据梯度，更新参数
    void train_itr();                   //优化一个样本，迭代次数控制迭代结束
    void train_loss();                  //优化一个样本，loss阈值控制迭代结束

    int argmax(vector<double>);         //vector数组中最大元素的下标
    void create_sigmoid_table();        //创建一个sigmoid表，加快计算速度
    Mat sigmoid(Mat);                   //对矩阵每个元素进行sigmoid计算
    double sigmoid(double);             //对单独的值进行sigmoid计算

private:
    vector<int> layers_info;            //每一层的信息
    Mat i_input, i_output;              //i表示输入层，分别是输入层的输入和输出
    Mat h_input, h_output;              //h表示隐含层，分别是隐含层的输入和输出
    Mat o_input, o_output;              //o表示输出层，分别是输出层的输入和输出
    vector<double> sigmoid_table;       //sigmoid表

};

#endif
