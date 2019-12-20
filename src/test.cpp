#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

const int WIDTH = 64;

struct TrainSet
{
    String file_name;
    Mat vec;
};

static void prepare(String root, vector<String> &files)
{
    glob(root + "*.tiff", files, false);
}

static void loadModel(String filename, Mat &m)
{
    ifstream fin(filename);

    if (!fin.is_open())
    {
        cout << "Fail to open the file...!" << '\n';
        return;
    }
    vector<double> v;
    double x;
    while (fin >> x)
    {
        v.push_back(x);
    }
    Mat tmpMat = Mat(v);
    tmpMat = tmpMat.reshape(0, tmpMat.rows / 4096);
    m = tmpMat.clone();
    fin.close();
}

static double calDistance(const Mat &a, const Mat &b)
{
    double dst = 0;
    for (int i = 0; i < a.rows; i++)
    {
        const double *p = a.ptr<double>(i);
        const double *q = b.ptr<double>(i);
        for (int j = 0; j < a.cols; j++)
        {
            dst += pow(p[j] - q[j], 2);
        }
    }
    return sqrt(dst);
}

int main(int argc, char const *argv[])
{
    if (argc != 4)
    {
        cout << "wrong arguments..." << '\n';
        return -1;
    }
    String image_name = argv[1];
    String model_name = argv[2];
    String trainset_name = argv[3];
    image_name.insert(0, "./JAFFE/test/");
    trainset_name.insert(0, "./");
    trainset_name.append("/");
    model_name.insert(0, "./model/");
    Mat eigenVector;
    vector<String> files;
    prepare(trainset_name, files);
    loadModel(model_name, eigenVector);
    vector<TrainSet> train_set;
    Mat mean = Mat::zeros(Size(WIDTH, WIDTH), CV_64FC1);
    mean = mean.reshape(0, 1);
    for (auto file : files)
    {
        Mat tmp = imread(file);
        resize(tmp, tmp, Size(WIDTH, WIDTH));
        cvtColor(tmp, tmp, COLOR_BGR2GRAY);
        tmp.convertTo(tmp, CV_64FC1);
        TrainSet t;
        t.file_name = file;
        t.vec = tmp.reshape(0, 1).clone();
        mean = mean + t.vec;
        train_set.push_back(t);
    }
    mean = mean / files.size();
    for (auto &t : train_set)
    {
        t.vec = eigenVector * (t.vec - mean).t();
    }
    Mat tmp = imread(image_name);
    resize(tmp, tmp, Size(WIDTH, WIDTH));
    cvtColor(tmp, tmp, COLOR_BGR2GRAY);
    tmp.convertTo(tmp, CV_64FC1);
    Mat test_image = tmp.reshape(0, 1).clone() - mean;
    Mat test_image_sub = eigenVector * test_image.t();
    double min_dist = -1;
    int min_index = -1;
    int count = 0;
    for (auto t : train_set)
    {
        double dst = calDistance(t.vec, test_image_sub);
        if (min_dist < 0)
        {
            min_dist = dst;
            min_index = count;
        }
        else
        {
            if (dst < min_dist)
            {
                min_dist = dst;
                min_index = count;
            }
        }
        cout << "Euler distance with image " << t.file_name << " : " << dst << '\n';
        count++;
    }
    cout << "Test done...!" << '\n';
    cout << "Origin image: " << image_name << '\n';
    cout << "Find similar image: " << train_set[min_index].file_name << '\n';
    cout << "The corresponding distance is " << min_dist << '\n';
    Mat syn = test_image_sub.t() * eigenVector + mean;
    syn.convertTo(syn, CV_8UC1);
    syn = syn.reshape(0, WIDTH);
    resize(syn, syn, Size(256, 256));
    Mat origin = imread(image_name);
    cvtColor(origin, origin, COLOR_BGR2GRAY);
    imshow("synthesis", syn);
    imshow("synthesis+origin", syn + origin);
    imshow("origin", imread(image_name));
    imshow("similar", imread(train_set[min_index].file_name));
    waitKey(0);
    return 0;
}