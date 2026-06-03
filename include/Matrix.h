#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <functional>

using namespace std;

struct Matrix {
    int rows, cols;
    vector<double> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, double val = 0.0) : rows(r), cols(c), data(r * c, val) {}

    double &at(int r, int c) {
        return data[r * cols + c];
    }

    double at(int r, int c) const {
        return data[r * cols + c];
    }

    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(double s) const;
    Matrix operator/(double s) const;

    Matrix hadamard(const Matrix &other) const;
    Matrix dot(const Matrix &other) const;
    Matrix transpose() const;
    Matrix applyFunction(const function<double(double)> &func) const;
    Matrix col_mean() const;

    Matrix slice(int start_row, int end_row) const;

    double sum() const;
    double mean() const;
};

Matrix zeros(int rows, int cols);
Matrix random(int rows, int cols, double scale, mt19937 &rng);