#include "DataLoader.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

namespace fs = std::filesystem;

const vector<string> DataLoader::CLASS_NAMES = {
    "bart_simpson",
    "charles_montgomery_burns",
    "homer_simpson",
    "krusty_the_clown",
    "lisa_simpson",
    "marge_simpson",
    "milhouse_van_houten",
    "moe_szyslak",
    "ned_flanders",
    "principal_skinner"
};

pair<Matrix, Matrix> DataLoader::load_data(const string &dir) {
    constexpr int img_size = 28;
    constexpr int flat_size = img_size * img_size;
    constexpr int num_classes = 10;

    vector<vector<double>> rows_x;
    vector<int> labels;

    for (int i = 0; i < num_classes; i++) {
        fs::path class_dir = fs::path(dir) / CLASS_NAMES[i];
        if (!fs::exists(class_dir) || !fs::is_directory(class_dir)) {
            cerr << "Directory not found: " << class_dir << endl;
            continue;
        }

        int count = 0;
        vector<fs::path> entries;
        for (auto &entry : fs::directory_iterator(class_dir)) {
            if (entry.is_regular_file()) {
                entries.push_back(entry.path());
            }
        }
        sort(entries.begin(), entries.end());

        for (auto &path : entries) {
            string ext = path.extension().string();
            if (ext != ".jpg" && ext != ".png" && ext != ".jpeg" && ext != ".bmp") {
                cerr << "Unsupported file format: " << path << endl;
                continue;
            }

            cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                cerr << "Failed to load image: " << path << endl;
                continue;
            }

            if (image.rows != img_size || image.cols != img_size) {
                cv::resize(image, image, cv::Size(img_size, img_size), 0, 0, cv::INTER_AREA);
            }

            vector<double> flat_image(flat_size);
            for (int i = 0; i < img_size; i++) {
                for (int j = 0; j < img_size; j++) {
                    flat_image[i * img_size + j] = static_cast<double>(image.at<uint8_t>(i, j)) / 255.0;
                }
            }

            rows_x.push_back(flat_image);
            labels.push_back(i);
            count++;
        }
        cout << "Loaded " << count << " images for class: " << CLASS_NAMES[i] << endl;
    }

    int N = static_cast<int>(rows_x.size());
    if (N == 0) {
        throw runtime_error("No images loaded. Please check the directory and file formats.");
    }

    Matrix X(N, flat_size);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < flat_size; j++) {
            X.at(i, j) = rows_x[i][j];
        }
    }

    Matrix y(N, num_classes, 0.0);
    for (int i = 0; i < N; i++) {
        y.at(i, labels[i]) = 1.0;
    }

    cout << "Total images loaded: " << N << endl;
    return {X, y};
}

double DataLoader::accuracy(const Matrix &predictions, const Matrix &trues) {
    int correct = 0;
    int total = predictions.rows;

    for (int i = 0; i < total; i++) {
        int pred_label = 0;
        double best = predictions.at(i, 0);
        for (int j = 1; j < predictions.cols; j++) {
            if (predictions.at(i, j) > best) {
                best = predictions.at(i, j);
                pred_label = j;
            }
        }

        int true_label = 0;
        for (int j = 1; j < predictions.cols; j++) {
            if (trues.at(i, j) > trues.at(i, true_label)) {
                true_label = j;
            }
        }
        if (pred_label == true_label) {
            correct++;
        }
    }

    return static_cast<double>(correct) / predictions.rows * 100.0;
}