#include "mat.h"

void Mat::print() {
    assert(rows>=0 && cols>=0);
    cout << "[";
    for (int r=0;r<rows; ++r) {
        if (r>0) cout << " ";//从第二行开始对齐
        if (cols>0) cout << "[";
        for (int c=0;c<cols;++c) {
            cout << setprecision(10) << m[r][c];
            if (c < cols-1) cout << ",";
        }
        if (cols>0) cout << "]";
        if (r != rows-1) cout << endl;
    }
    cout << "]" << endl << endl;
}


Mat Mat::operator *(Mat b) {
    assert(cols == b.rows);
    Mat res(rows, b.cols, 0);
    for (int i=0;i<rows;++i) {
        for (int j=0;j<b.cols;++j) {
            double sum = 0.;
            for (int _e=0;_e<cols;++_e) sum+=m[i][_e]*b.m[_e][j];
            res.m[i][j] = sum;
        }
    }
    return res;
}

Mat Mat::operator *(double x) {
    Mat res(rows, cols, 0);
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            res.m[i][j] = m[i][j]*x;
        }
    }
    return res;
}

Mat Mat::operator +(Mat b) {
    assert(rows==b.rows && cols==b.cols);
    Mat res(rows, cols, 0);
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            res.m[i][j] = m[i][j]+b.m[i][j];
        }
    }
    return res;
}

Mat Mat::operator +(double x) {
    Mat res(rows, cols, 0);
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            res.m[i][j] = m[i][j]+x;
        }
    }
    return res;
}

Mat Mat::operator -(Mat b) {
    assert(rows==b.rows && cols==b.cols);
    Mat res(rows, cols, 0);
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            res.m[i][j] = m[i][j]-b.m[i][j];
        }
    }
    return res;
}

Mat Mat::operator -(double x) {
    Mat res(rows, cols, 0);
    for (int i=0; i<rows; ++i) {
        for ( int j=0; j<cols; ++j) {
            res.m[i][j] = m[i][j]-x;
        }
    }
    return res;
}


Mat Mat::operator /(double x) {
    Mat res(rows, cols, 0);
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            res.m[i][j]= m[i][j]/x;
        }
    }
    return res;
}

vector<double>& Mat::operator [](int x) {
    assert(x>=0 && x<rows);
    return m[x];
}

Mat Mat::inverse() {
    Mat res(cols, rows, 0);
    for (int i=0;i<rows;++i) {
        for (int j=0;j<cols;++j) {
            res.m[j][i] = m[i][j];
        }
    }
    return res;
}

Mat Mat::mul(Mat b) {
    assert(rows==b.rows && cols==b.cols);
    Mat res(rows, cols, 0);
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            res.m[i][j] = m[i][j]*b.m[i][j];
        }
    }
    return res;
}

Mat Mat::square() {
    Mat res(rows, cols, 0);
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            res.m[i][j] = m[i][j]*m[i][j];
        }
    }
    return res;
}

double Mat::sum() {
    double sum = 0;
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<cols; ++j) {
            sum += m[i][j];
        }
    }
    return sum;
}
