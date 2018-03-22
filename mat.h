#ifndef MAT_H
#define MAT_H
#include <vector>
#include <iostream>
#include <assert.h>
#include <iomanip>

using namespace std;

class Mat {
public:
    Mat():rows(0), cols(0), m(vector<vector<double>>{}){}
    Mat(vector<double> _m):rows(1), cols(_m.size()) {
        m = vector<vector<double>>(rows, vector<double>(cols, 0));
        m[0] = _m;
    }
    Mat(vector<vector<double>> _m):m(_m) {
        rows = _m.size();
        cols = _m[0].size();
    }
    Mat(int _rows, int _cols, double val=0.):rows(_rows),cols(_cols){
        m = vector<vector<double>>(rows, vector<double>(cols, val));
    }

    void print();                       //打印矩阵

    Mat operator *(Mat);                //矩阵行列式计算
    Mat operator *(double);             //数乘
    Mat operator +(Mat);                //矩阵各对应元素相加（矩阵行列数相同）
    Mat operator +(double);             //矩阵各元素增加一个常量
    Mat operator -(Mat);                //矩阵各对应元素相减（矩阵行列数相同）
    Mat operator -(double);             //矩阵各元素减去一个常量
    Mat operator /(double);             //矩阵各元素除以一个常量
    vector<double>& operator [](int);   //用Mat[x][y]方式获取一个元素的值

    Mat mul(Mat);                       //矩阵对应元素相乘，返回一个新矩阵
    Mat square();                       //矩阵各元素取平方
    Mat inverse();                      //矩阵转置
    double sum();                       //矩阵所有元素求和

public:
    int rows = -1;                      //矩阵行数
    int cols = -1;                      //矩阵列数
    vector<vector<double>> m;           //用二维vector数组存储矩阵元素
};

#endif
