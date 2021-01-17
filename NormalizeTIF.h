#include <opencv2/opencv.hpp>

cv::Mat normTIF(const cv ::Mat &mat)
{
    double min, max;
    cv::Mat ret;
    cv::minMaxLoc(mat, &min, &max, NULL, NULL);
    ret = mat * (256.0f / max);
    ret = ret - min * (256.0f / max);
    ret.convertTo(ret, CV_8U);
    return ret;
}
