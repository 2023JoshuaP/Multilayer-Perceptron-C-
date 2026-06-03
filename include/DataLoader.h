#pragma once

#include "Matrix.h"
#include <vector>
#include <string>
#include <utility>

using namespace std;

struct DataLoader {
    static const vector<string> CLASS_NAMES;
    static pair<Matrix, Matrix> load_data(const string& dir);
    static double accuracy(const Matrix& predictions, const Matrix& trues);
};