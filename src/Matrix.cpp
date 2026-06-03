#include "Matrix.h"
#include <stdexcept>
#include <numeric>

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match for addition.");
    }

    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        res.data[i] = data[i] + other.data[i];
    }

    return res;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match for subtraction.");
    }

    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        res.data[i] = data[i] - other.data[i];
    }

    return res;
}

Matrix Matrix::operator*(double s) const {
    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        res.data[i] = data[i] * s;
    }
    return res;
}

Matrix Matrix::operator/(double s) const {
    if (s == 0) {
        throw invalid_argument("Division by zero.");
    }

    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        res.data[i] = data[i] / s;
    }
    return res;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match for Hadamard product.");
    }

    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        res.data[i] = data[i] * other.data[i];
    }

    return res;
}

Matrix Matrix::dot(const Matrix& other) const {
    if (cols != other.rows) {
        throw invalid_argument("Inner matrix dimensions must match for dot product.");
    }

    Matrix res(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols; k++) {
                sum += at(i, k) * other.at(k, j);
            }
            res.at(i, j) = sum;
        }
    }

    return res;
}

Matrix Matrix::transpose() const {
    Matrix res(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.at(j, i) = at(i, j);
        }
    }
    return res;
}

Matrix Matrix::applyFunction(const function<double(double)>& func) const {
    Matrix res(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        res.data[i] = func(data[i]);
    }
    return res;
}

Matrix Matrix::col_mean() const {
    Matrix res(1, cols, 0.0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res.at(0, j) += at(i, j);
        }
    }

    for (int j = 0; j < cols; j++) {
        res.at(0, j) /= rows;
    }

    return res;
}

Matrix Matrix::slice(int start_row, int end_row) const {
    int n = end_row - start_row;
    Matrix res(n, cols);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < cols; j++) {
            res.at(i, j) = at(start_row + i, j);
        }
    }
    return res;
}

double Matrix::sum() const {
    double sum = 0.0;
    for (double val : data) {
        sum += val;
    }
    return sum;
}

double Matrix::mean() const {
    return sum() / (rows * cols);
}

Matrix zeros(int rows, int cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix random(int rows, int cols, double scale, mt19937& rng) {
    normal_distribution<double> dist(0.0, scale);
    Matrix res(rows, cols);
    for (double &val : res.data) {
        val = dist(rng);
    }
    return res;
}