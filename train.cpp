#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

const int WIDTH = 64;

Mat result = Mat::zeros(Size(WIDTH * 11, WIDTH), CV_8UC1);

static void prepare(String root, vector<String> &files)
{
    glob(root + "*.tiff", files, false);
}

static void writeModel(String filename, Mat &m)
{
    ofstream fout(filename);

    if (!fout)
    {
        cout << "Fail to open the file...!" << '\n';
        return;
    }
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            fout << m.at<double>(i, j) << '\t';
        }
        fout << '\n';
    }
    fout.close();
}

static void displayEigenFace(int index, Mat &m)
{
    double min, max;
    for (int i = 0; i < m.cols; i++)
    {
        if (i == 0)
        {
            min = max = m.at<double>(0, i);
        }
        else
        {
            if (m.at<double>(0, i) < min)
                min = m.at<double>(0, i);
            if (m.at<double>(0, i) > max)
                max = m.at<double>(0, i);
        }
    }
    double *p = m.ptr<double>(0);
    for (int i = 0; i < m.cols; i++)
    {
        p[i] = (p[i] - min) / (max - min) * 255;
    }
    m = m.reshape(0, WIDTH);
    m.convertTo(m, CV_8UC1);
    for (int i = 0; i < m.rows; i++)
    {
        uchar *p = m.ptr<uchar>(i);
        uchar *q = result.ptr<uchar>(i);
        for (int j = 0; j < m.cols; j++)
        {
            q[WIDTH * index + j] = p[j];
        }
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 4)
    {
        cout << "wrong arguments..." << '\n';
        return -1;
    }
    double energy = atof(argv[1]);
    String filename = argv[2];
    filename.insert(0, "./model/");
    String root = argv[3];
    root.insert(0, "./");
    root.append("/");
    vector<String> files;
    prepare(root, files);
    vector<Mat> faces;
    for (auto file : files)
    {
        Mat tmp = imread(file);
        resize(tmp, tmp, Size(WIDTH, WIDTH));
        cvtColor(tmp, tmp, COLOR_BGR2GRAY);
        faces.push_back(tmp.reshape(0, 1));
    }
    Mat covar;
    Mat mean;
    calcCovarMatrix(faces.data(), faces.size(), covar, mean, COVAR_NORMAL);
    Mat eigenValue, eigenVector;
    eigen(covar, eigenValue, eigenVector);
    double sumEigenValue = 0.0;
    for (int i = 0; i < eigenValue.rows; i++)
    {
        sumEigenValue += eigenValue.at<double>(i, 0);
    }
    sumEigenValue *= energy;
    int numOfEigen = 0;
    for (int i = 0; i < eigenValue.rows; i++)
    {
        sumEigenValue -= eigenValue.at<double>(i, 0);
        if (sumEigenValue < 0)
            break;
        numOfEigen++;
    }
    int count = 0;
    while (count < 10)
    {
        Mat tmp = eigenVector.row(count).clone();
        displayEigenFace(count + 1, tmp);
        count++;
    }
    Mat m = eigenVector.rowRange(0, numOfEigen - 1);
    writeModel(filename, m);
    Mat tmp;
    mean.convertTo(tmp, CV_8UC1);
    tmp = tmp.reshape(0, 64);
    for (int i = 0; i < tmp.rows; i++)
    {
        uchar *p = tmp.ptr<uchar>(i);
        uchar *q = result.ptr<uchar>(i);
        for (int j = 0; j < tmp.cols; j++)
        {
            q[j] = p[j];
        }
    }
    imshow("result", result);
    waitKey(0);
    return 0;
}