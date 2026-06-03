#include "MultiLayerPerceptron.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <numeric>

MultiLayerPerceptron::MultiLayerPerceptron(const vector<int> &layer_sizes, shared_ptr<ActivationFunction> activation, double learning_rate, double momentum, double weight_decay, int seed) : layer_sizes_(layer_sizes), activation_(activation), learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay), num_layers_(static_cast<int>(layer_sizes.size())), rng_(seed) {
    mt19937 init_rng(seed);
    for (int i = 0; i < num_layers_ - 1; i++) {
        int fan_in = layer_sizes_[i];
        int fan_out = layer_sizes_[i + 1];
        double scale = sqrt(2.0 / fan_in);
        weights_.push_back(random(fan_in, fan_out, scale, init_rng));
        biases_.push_back(zeros(1, fan_out));
        vel_weights_.push_back(zeros(fan_in, fan_out));
        vel_biases_.push_back(zeros(1, fan_out));
    }
}

vector<Matrix> MultiLayerPerceptron::forward(const Matrix &input) const {
    vector<Matrix> activations;
    activations.reserve(num_layers_);
    activations.push_back(input);

    int n_weight_layers = num_layers_ - 1;
    Matrix a = input;
    for (int i = 0; i < n_weight_layers; i++) {
        Matrix z = a.dot(weights_[i]);
        for (int r = 0; r < z.rows; r++) {
            for (int c = 0; c < z.cols; c++) {
                z.at(r, c) = z.at(r, c) + biases_[i].at(0, c);
            }
        }

        if (i == n_weight_layers - 1) {
            a = Softmax::forward(z);
        }
        else {
            a = activation_->forward(z);
        }

        activations.push_back(a);
    }

    return activations;
}

void MultiLayerPerceptron::backward(const vector<Matrix> &activations, const Matrix &y_true) {
    int n = y_true.rows;
    int n_lay = num_layers_ - 1;

    vector<Matrix> dW(n_lay);
    vector<Matrix> db(n_lay);

    Matrix delta = activations.back() - y_true;

    for (int i = n_lay - 1; i >= 0; i--) {
        dW[i] = activations[i].transpose().dot(delta) / n;
        db[i] = delta.col_mean();

        if (i > 0) {
            Matrix da = delta.dot(weights_[i].transpose());
            Matrix derivative = activation_->derivative(activations[i]);
            delta = da.hadamard(derivative);
        }
    }

    // SGD with momentum + L2 regularization: v = β*v + grad + λ*w, w = w - lr*v
    for (int i = 0; i < n_lay; i++) {
        for (int k = 0; k < weights_[i].rows * weights_[i].cols; k++) {
            double grad_with_decay = dW[i].data[k] + weight_decay_ * weights_[i].data[k];
            vel_weights_[i].data[k] = momentum_ * vel_weights_[i].data[k] + grad_with_decay;
            weights_[i].data[k] -= learning_rate_ * vel_weights_[i].data[k];
        }
        for (int k = 0; k < biases_[i].cols; k++) {
            vel_biases_[i].data[k] = momentum_ * vel_biases_[i].data[k] + db[i].data[k];
            biases_[i].data[k] -= learning_rate_ * vel_biases_[i].data[k];
        }
    }
}

double MultiLayerPerceptron::cross_entropy_loss(const Matrix &y_pred, const Matrix &y_true) {
    constexpr double epsilon = 1e-12;
    double loss = 0.0;
    for (int i = 0; i < y_true.rows; i++) {
        for (int j = 0; j < y_true.cols; j++) {
            double p = max(min(y_pred.at(i, j), 1.0 - epsilon), epsilon);
            loss -= y_true.at(i, j) * log(p);
        }
    }
    return loss / y_true.rows;
}

double MultiLayerPerceptron::mse_loss(const Matrix &y_pred, const Matrix &y_true) {
    Matrix diff = y_pred - y_true;
    double s = 0.0;
    for (double value : diff.data) {
        s += value * value;
    }
    return s / diff.data.size();
}

void MultiLayerPerceptron::shuffle_data(Matrix &X, Matrix &y) {
    int n = X.rows;
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng_);

    Matrix X_shuffled(n, X.cols);
    Matrix y_shuffled(n, y.cols);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < X.cols; j++) {
            X_shuffled.at(i, j) = X.at(indices[i], j);
        }
        for (int j = 0; j < y.cols; j++) {
            y_shuffled.at(i, j) = y.at(indices[i], j);
        }
    }

    X = X_shuffled;
    y = y_shuffled;
}

TrainHistory MultiLayerPerceptron::train(const Matrix &X, const Matrix &y, int epochs, int batch_size, const Matrix *X_val, const Matrix *y_val, bool verbose, int patience) {
    TrainHistory history;
    bool has_val = (X_val != nullptr && y_val != nullptr);

    // Make mutable copies for shuffling
    Matrix X_train = X;
    Matrix y_train = y;

    int n = X_train.rows;
    int num_batches = (n + batch_size - 1) / batch_size;

    double best_val_loss = 1e18;
    int epochs_no_improve = 0;
    // Save best weights for restoration
    vector<Matrix> best_weights = weights_;
    vector<Matrix> best_biases = biases_;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        // Shuffle data at the start of each epoch
        shuffle_data(X_train, y_train);

        double epoch_loss = 0.0;

        // Mini-batch training
        for (int b = 0; b < num_batches; b++) {
            int start = b * batch_size;
            int end = min(start + batch_size, n);

            Matrix X_batch = X_train.slice(start, end);
            Matrix y_batch = y_train.slice(start, end);

            auto activations = forward(X_batch);
            double batch_loss = cross_entropy_loss(activations.back(), y_batch);
            epoch_loss += batch_loss * (end - start);

            backward(activations, y_batch);
        }

        epoch_loss /= n;
        history.train_losses.push_back(epoch_loss);

        // Validation
        double val_loss = 0.0;
        if (has_val) {
            Matrix val_pred = predict(*X_val);
            val_loss = cross_entropy_loss(val_pred, *y_val);
            history.val_losses.push_back(val_loss);

            // Early stopping with best weight saving
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                epochs_no_improve = 0;
                best_weights = weights_;
                best_biases = biases_;
            } else {
                epochs_no_improve++;
            }

            if (epochs_no_improve >= patience) {
                if (verbose) {
                    cout << "Early stopping at epoch " << epoch << " (best val loss: " << fixed << setprecision(4) << best_val_loss << ")" << endl;
                }
                // Restore best weights
                weights_ = best_weights;
                biases_ = best_biases;
                break;
            }
        }

        if (verbose && (epoch % max(1, epochs / 20) == 0 || epoch == 1)) {
            cout << "Epoch " << epoch << "/" << epochs << " - Train Loss: " << fixed << setprecision(4) << epoch_loss;
            if (has_val) {
                cout << " - Val Loss: " << fixed << setprecision(4) << val_loss;
            }
            cout << endl;
        }
    }

    return history;
}

Matrix MultiLayerPerceptron::predict(const Matrix &X) const {
    return forward(X).back();
}