#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include "include/Matrix.h"
#include "include/Activation.h"
#include "include/MultiLayerPerceptron.h"
#include "include/DataLoader.h"

using namespace std;

int main(int argc, char* argv[]) {
    string train_data_grayscale = "simpsons-mnist/dataset/grayscale/train";
    string test_data_grayscale = "simpsons-mnist/dataset/grayscale/test";
    string train_data_rgb = "simpsons-mnist/dataset/rgb/train";
    string test_data_rgb = "simpsons-mnist/dataset/rgb/test";

    if (argc == 3) {
        train_data_grayscale = argv[1];
        test_data_grayscale = argv[2];
    }
    else if (argc == 5) {
        train_data_grayscale = argv[1];
        test_data_grayscale = argv[2];
        train_data_rgb = argv[3];
        test_data_rgb = argv[4];
    }

    cout << "Loading grayscale training data..." << endl;
    auto [X_train_gray, y_train_gray] = DataLoader::load_data(train_data_grayscale);
    cout << "Loading grayscale test data..." << endl;
    auto [X_test_gray, y_test_gray] = DataLoader::load_data(test_data_grayscale);

    cout << "X_train_gray: (" << X_train_gray.rows << ", " << X_train_gray.cols << ")" << endl;
    cout << "X_test_gray: (" << X_test_gray.rows << ", " << X_test_gray.cols << ")" << endl;

    auto activation = make_shared<ReLU>();

    MultiLayerPerceptron mlp({784, 256, 128, 64, 10}, activation, 0.005, 0.9, 1e-5, 42);

    constexpr int EPOCHS = 300;
    constexpr int BATCH_SIZE = 64;
    auto history = mlp.train(X_train_gray, y_train_gray, EPOCHS, BATCH_SIZE, &X_test_gray, &y_test_gray, true, 80);

    cout << "Evaluating on grayscale test set..." << endl;
    Matrix train_predictions = mlp.predict(X_train_gray);
    double train_acc = DataLoader::accuracy(train_predictions, y_train_gray);
    cout << "Final Training Accuracy: " << fixed << setprecision(2) << train_acc << "%" << endl;

    Matrix test_predictions = mlp.predict(X_test_gray);
    double test_acc = DataLoader::accuracy(test_predictions, y_test_gray);
    cout << "Final Test Accuracy: " << fixed << setprecision(2) << test_acc << "%" << endl;

    cout << "Per-class Test Accuracy:" << endl;
    int num_classes = static_cast<int>(DataLoader::CLASS_NAMES.size());
    vector<int> class_correct(num_classes, 0), class_total(num_classes, 0);

    for (int i = 0; i < test_predictions.rows; i++) {
        int pred_cls = 0;
        double best = test_predictions.at(i, 0);
        for (int j = 1; j < test_predictions.cols; j++) {
            if (test_predictions.at(i, j) > best) {
                best = test_predictions.at(i, j);
                pred_cls = j;
            }
        }

        int true_cls = 0;
        for (int j = 0; j < y_test_gray.cols; j++) {
            if (y_test_gray.at(i, j) > y_test_gray.at(i, true_cls)) {
                true_cls = j;
            }
        }

        class_total[true_cls]++;
        if (pred_cls == true_cls) {
            class_correct[true_cls]++;
        }
    }

    for (int c = 0; c < num_classes; c++) {
        double acc = class_total[c] > 0 ? 100.0 * class_correct[c] / class_total[c] : 0.0;
        cout << " [" << c << "] " << left << setw(28) << DataLoader::CLASS_NAMES[c] << ": " << fixed << setprecision(1) << acc << "%" << " (" << class_correct[c] << "/" << class_total[c] << ")" << endl;
    }

    return 0;
}