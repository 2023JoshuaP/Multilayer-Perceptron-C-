#include "Activation.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

Matrix Sigmoid::forward(const Matrix& x) const {
    return x.applyFunction([](double val) {
        return 1.0 / (1.0 + exp(-val));
    });
}

Matrix Sigmoid::derivative(const Matrix& a) const {
    return a.applyFunction([](double val) {
        return val * (1.0 - val);
    });
}

Matrix ReLU::forward(const Matrix& x) const {
    return x.applyFunction([](double val) {
        return max(0.0, val);
    });
}

Matrix ReLU::derivative(const Matrix& a) const {
    return a.applyFunction([](double val) {
        return val > 0.0 ? 1.0 : 0.0;
    });
}

Matrix Softmax::forward(const Matrix& z) {
    Matrix res(z.rows, z.cols);
    
    for (int i = 0; i < z.rows; i++) {
        double row_max = z.at(i, 0);
        for (int j = 1; j < z.cols; j++) {
            row_max = max(row_max, z.at(i, j));
        }

        double sum_exp = 0.0;
        for (int j = 0; j < z.cols; j++) {
            res.at(i, j) = exp(z.at(i, j) - row_max);
            sum_exp += res.at(i, j);
        }

        for (int j = 0; j < z.cols; j++) {
            res.at(i, j) /= sum_exp;
        }
    }

    return res;
}