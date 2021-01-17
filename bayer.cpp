#include <iostream>

#include <opencv2/opencv.hpp>

#include <map>
#include <vector>

#include "NormalizeTIF.h"
#include "bayer.h"

using bayerType = std::map<std::string, cv::Mat>;
bayerType getMasks(const cv::Mat &seed, const cv::Size &size) // std::vector<std::string> channels_name = {"R","G","B"})
{
    cv::Mat mask = cv::Mat(size.height, size.width, CV_8UC1, cv::Scalar(0));
    cv::repeat(seed, size.height / seed.rows, size.width / seed.cols).copyTo(mask(cv::Rect(0, 0, size.width - size.width % seed.cols, size.height - size.height % size.height)));
    if (size.height % seed.rows != 0)
    {
        cv::repeat(seed(cv::Rect(0, 0, seed.cols, size.height % seed.rows)), 1, size.width / seed.cols).copyTo(mask(cv::Rect(0, size.height - size.height % seed.rows, size.width - size.width % seed.cols, size.height % seed.rows)));
    }
    if (size.width % seed.cols != 0)
    {
        cv::repeat(seed(cv::Rect(0, 0, size.width % seed.cols, seed.rows)), size.height / seed.rows, 1).copyTo(mask(cv::Rect(size.width - size.width % seed.cols, 0, size.width % seed.cols, size.height - size.height % seed.rows)));
    }
    if (size.height % seed.rows != 0 && size.width % seed.cols != 0)
    {
        seed(cv::Rect(0, 0, size.width % seed.cols, size.height % seed.cols)).copyTo(mask(cv::Rect(size.width - size.width % seed.cols, size.height - size.height % seed.rows, size.width % seed.cols, size.height % seed.rows)));
    }

    bayerType table;
    table["R"] = cv::Mat(1, 256, CV_8UC1, cv::Scalar(0));
    table["R"].data['R'] = 255;
    table["G"] = cv::Mat(1, 256, CV_8UC1, cv::Scalar(0));
    table["G"].data['G'] = 255;
    table["B"] = cv::Mat(1, 256, CV_8UC1, cv::Scalar(0));
    table["B"].data['B'] = 255;

    bayerType mask_channels;
    cv::LUT(mask, table["R"], mask_channels["R"]);
    cv::LUT(mask, table["G"], mask_channels["G"]);
    cv::LUT(mask, table["B"], mask_channels["B"]);

    return mask_channels;
}
bayerType getBayerMatfromColor(const cv::Mat &mat, bayerType &masks)
{
    bayerType mat_channels;
    std::vector<cv::Mat> channels;
    if (mat.channels() == 1)
    {
        mat_channels["R"] = mat & masks["R"];
        mat_channels["G"] = mat & masks["G"];
        mat_channels["B"] = mat & masks["B"];
    }
    else
    {
        cv::split(mat, channels);
        mat_channels["R"] = channels[2] & masks["R"];
        mat_channels["G"] = channels[1] & masks["G"];
        mat_channels["B"] = channels[0] & masks["B"];
    }

    return mat_channels;
};

int main(int argc, const char *const argv[])
{
    cv::CommandLineParser parser(argc, argv, "{help||}{path|../lancaster.jpg|}");
    cv::Mat mat = cv::imread(parser.get<std::string>("path"), cv::IMREAD_REDUCED_COLOR_2);
    cv::Mat seed = (cv::Mat_<char>(2, 2) << 'R', 'G',
                    'G', 'B');
    auto masks = getMasks(seed, mat.size());
    auto bayer = getBayerMatfromColor(mat, masks);
    // cv::Mat bilinearMat = bayer::RB::linear(bayer::G::linear(bayer), bayer);
    // cv::Mat colorRatMat = bayer::RB::colorRatios(bayer::G::linear(bayer), bayer);
    // cv::Mat colorRatInlogMat = bayer::RB::colorRatiosInLog(bayer::G::linear(bayer), bayer);
    // cv::Mat GradientlineraMat = bayer::RB::linear(bayer::G::GradientFirstOrder(bayer), bayer);
    // cv::Mat GradientCRMat = bayer::RB::colorRatios(bayer::G::GradientFirstOrder(bayer), bayer);
    // cv::imshow("bilinear", bilinearMat);
    // cv::imshow("colorRatios", colorRatMat);
    // cv::imshow("colorRatiosInLog", colorRatInlogMat);
    // cv::imshow("firstOrderGradient", GradientCRMat);
    // cv::imshow("Glinear", GradientlineraMat);
    // cv::imshow("mat", mat);
    bayer::G::RonKimmel(bayer);
    // std::cout << cv::sum(cv::abs(GradientlineraMat - mat)) / mat.size().area() << std::endl;
    // cv::CommandLineParser parser(argc, argv, "{h5elp||}{path|../白光-40×-LG-01_0-0-0_Normal.tif|}{pathRed|../645nm滤光片-1×-HG.raw_0-0-0_Normal.tif|}");
    // cv::Mat mat = normTIF(cv::imread(parser.get<std::string>("path"), cv::IMREAD_UNCHANGED));
    // cv::Mat maskR = normTIF(cv::imread(parser.get<std::string>("pathRed"), cv::IMREAD_UNCHANGED));

    // cv::Mat translationMat = (cv::Mat_<float>(2, 3) << 1, 0, -6,
    //                                                     0, 1, 12);
    // std::cout << "sdadas";
    // cv::warpAffine(maskR, maskR, translationMat, maskR.size(), 1, 0, cv::Scalar(0, 0, 0));

    // cv::namedWindow("mat", cv::WINDOW_FREERATIO);
    // cv::namedWindow("maskR", cv::WINDOW_FREERATIO);

    // cv::imshow("mat", mat);
    // cv::imshow("maskR", maskR);
    while (cv::waitKey(0) != 'q')
        ;
}