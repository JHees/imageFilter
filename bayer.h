#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

namespace bayer
{
    using Type = std::map<std::string, cv::Mat>;
    using channelsT = std::vector<cv::Mat>;
    const cv::Mat gradientX33 = (cv::Mat_<char>(3, 3) << 0, 0, 0, 1, 1, -1, 0, 0, 0);
    const cv::Mat gradientY33 = (cv::Mat_<char>(3, 3) << 0, 1, 0, 0, 1, 0, 0, -1, 0);

    namespace G
    {
        const cv::Mat averKernel33 = (cv::Mat_<float>(3, 3) << 0, 0.25f, 0,
                                      0.25f, 1, 0.25f,
                                      0, 0.25f, 0);
        inline cv::Mat linear(Type& bayer)
        {
            cv::Mat ret;
            cv::filter2D(bayer["G"], ret, CV_8UC1, G::averKernel33, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            return ret;
        }
        inline cv::Mat Gradient(Type& bayer, uchar epsilon, cv::Mat gradientX, cv::Mat gradientY)
        {
            cv::Mat xGra, yGra; // gradient
            Type bayer_buf;
            // bayer["G"].convertTo(xGra, CV_8SC1);
            // bayer["G"].convertTo(yGra, CV_8SC1);
            bayer["G"].convertTo(bayer_buf["G"], CV_16SC1);

            cv::filter2D(bayer_buf["G"], xGra, CV_16SC1, gradientX, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            cv::filter2D(bayer_buf["G"], yGra, CV_16SC1, gradientY, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            xGra = cv::abs(xGra);
            yGra = cv::abs(yGra);

            cv::Mat buf;
            buf = xGra - yGra;
            for (size_t x = 0; x < buf.cols; ++x)
            {
                for (size_t y = 0; y < buf.rows; ++y)
                {
                    short& value = buf.at<short>(cv::Point(x, y));
                    if (value > epsilon)
                    {
                        value = 1 << 0;
                    }
                    else if (value < -epsilon)
                    {
                        value = 1 << 1;
                    }
                    else
                    {
                        value = 1 << 2;
                    }
                }
            }

            std::map<std::string, cv::Mat> masks;
            masks["X"] = (buf & 1 << 0) * 255;
            masks["X"].convertTo(masks["X"], CV_8UC1);
            masks["Y"] = (buf & 1 << 1) * 255;
            masks["Y"].convertTo(masks["Y"], CV_8UC1);
            masks["XY"] = (buf & 1 << 2) * 255;
            masks["XY"].convertTo(masks["XY"], CV_8UC1);

            cv::Mat kernelX = (cv::Mat_<float>(3, 3) << 0, 0.5f, 0, 0, 1, 0, 0, 0.5f, 0);
            cv::Mat kernelY = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0.5f, 1, 0.5f, 0, 0, 0);
            std::map<std::string, cv::Mat> gradient;
            cv::filter2D(bayer["G"], gradient["X"], CV_8UC1, kernelX, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            cv::filter2D(bayer["G"], gradient["Y"], CV_8UC1, kernelY, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            cv::filter2D(bayer["G"], gradient["XY"], CV_8UC1, G::averKernel33, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);

            cv::Mat ret = (gradient["X"] & masks["X"]) + (gradient["Y"] & masks["Y"]) + (gradient["XY"] & masks["XY"]);
            return ret;
        }
        inline cv::Mat GradientFirstOrder(Type& bayer, uchar epsilon = 0)
        {
            return Gradient(bayer, epsilon, gradientX33, gradientY33);
        }
        inline cv::Mat RonKimmel(Type& bayer)
        {
            Type bayer_buf;
            bayer["R"].convertTo(bayer_buf["R"], CV_32FC1);
            bayer["G"].convertTo(bayer_buf["G"], CV_32FC1);
            bayer["B"].convertTo(bayer_buf["B"], CV_32FC1);
            bayer_buf["RB"] = bayer_buf["R"] + bayer_buf["B"];
            bayer_buf["RGB"] = bayer_buf["RB"] + bayer_buf["G"];


            const cv::Mat gradientX33 = (cv::Mat_<float>(3, 3) << 0, 0, 0, 1, 0, -1, 0, 0, 0);
            const cv::Mat gradientY33 = (cv::Mat_<float>(3, 3) << 0, 1, 0, 0, 0, 0, 0, -1, 0);
            std::map<std::string, cv::Mat> diff;
            cv::filter2D(bayer_buf["RGB"], diff["x"], CV_32FC1, gradientX33 * 0.5f, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            cv::filter2D(bayer_buf["RGB"], diff["y"], CV_32FC1, gradientY33 * 0.5f, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            diff["x"] = cv::abs(diff["x"]);
            diff["y"] = cv::abs(diff["y"]);

            cv::Mat buf;

            cv::Mat kernelxd = (cv::Mat_<float>(3, 3) << 0, 0, 1, 0, -1, 0, 0, 0, 0);
            cv::filter2D(bayer_buf["G"], diff["xd"], CV_32FC1, kernelxd, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            diff["xd"] = cv::abs(diff["xd"]);
            cv::copyMakeBorder(diff["xd"], buf, 1, 1, 1, 1, cv::BORDER_REFLECT_101);
            for (size_t x = 1; x < buf.cols - 1; ++x)
                for (size_t y = 1; y < buf.rows - 1; ++y)
                {
                    diff["xd"].at<float>(cv::Point(x - 1, y - 1)) = fmax(buf.at<float>(cv::Point(x, y)), buf.at<float>(cv::Point(x + 1, y - 1))) / sqrt(2);
                }
            diff["xd"] = diff["xd"] / std::sqrt(2);
            // buf = cv::Mat();
            cv::Mat kernelxd2 = (cv::Mat_<float>(3, 3) << 0, 0, 1, 0, 0, 0, -1, 0, 0);
            cv::filter2D(bayer_buf["RB"], buf, CV_32FC1, kernelxd2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            diff["xd"] += (buf / sqrt(8));


            cv::Mat kernelyd = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, 0);
            cv::filter2D(bayer_buf["G"], diff["yd"], CV_32FC1, kernelyd, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            diff["yd"] = cv::abs(diff["yd"]);
            cv::copyMakeBorder(diff["yd"], buf, 1, 1, 1, 1, cv::BORDER_REFLECT_101);
            for (size_t x = 1; x < buf.cols - 1; ++x)
                for (size_t y = 1; y < buf.rows - 1; ++y)
                {
                    diff["yd"].at<float>(cv::Point(x - 1, y - 1)) = fmax(buf.at<float>(cv::Point(x, y)), buf.at<float>(cv::Point(x - 1, y - 1))) / sqrt(2);
                }

            cv::Mat kernelyd2 = (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 0, 0, 0, 0, -1);
            cv::filter2D(bayer_buf["RB"], buf, CV_32FC1, kernelyd2, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            diff["yd"] += buf / sqrt(8);



            diff["xd"] = cv::abs(diff["xd"]);
            diff["yd"] = cv::abs(diff["yd"]);
            cv::copyMakeBorder(diff["x"], diff["x"], 1, 1, 1, 1, cv::BORDER_REFLECT_101);
            cv::copyMakeBorder(diff["xd"], diff["xd"], 1, 1, 1, 1, cv::BORDER_REFLECT_101);
            cv::copyMakeBorder(diff["y"], diff["y"], 1, 1, 1, 1, cv::BORDER_REFLECT_101);
            cv::copyMakeBorder(diff["yd"], diff["yd"], 1, 1, 1, 1, cv::BORDER_REFLECT_101);
            cv::copyMakeBorder(bayer_buf["G"], bayer_buf["G"], 1, 1, 1, 1, cv::BORDER_REFLECT_101);

            std::vector<cv::Mat> channels(3);
            channels[0] = bayer_buf["B"].clone();
            channels[1] = bayer_buf["B"].clone();
            channels[2] = bayer_buf["B"].clone();

            // bayer_buf["R/G"] = bayer_buf["R"] / bayer_buf["G"];
            // bayer_buf["B/G"] = bayer_buf["B"] / bayer_buf["G"];
            for (size_t x = 1; x < bayer_buf["G"].cols - 1; ++x)
                for (size_t y = 1; y < bayer_buf["G"].rows - 1; ++y)
                {
                    double e[9];
                    e[0] = 1.0f / sqrt(1 + pow(diff["yd"].at<float>(cv::Point(x - 1, y - 1)), 2) + pow(diff["yd"].at<float>(cv::Point(x, y)), 2));
                    e[1] = 1.0f / sqrt(1 + pow(diff["y"].at<float>(cv::Point(x, y - 1)), 2) + pow(diff["y"].at<float>(cv::Point(x, y)), 2));
                    e[2] = 1.0f / sqrt(1 + pow(diff["xd"].at<float>(cv::Point(x + 1, y - 1)), 2) + pow(diff["xd"].at<float>(cv::Point(x, y)), 2));
                    e[3] = 1.0f / sqrt(1 + pow(diff["x"].at<float>(cv::Point(x - 1, y)), 2) + pow(diff["x"].at<float>(cv::Point(x, y)), 2));
                    e[4] = 1;
                    e[5] = 1.0f / sqrt(1 + pow(diff["x"].at<float>(cv::Point(x + 1, y)), 2) + pow(diff["x"].at<float>(cv::Point(x, y)), 2));
                    e[6] = 1.0f / sqrt(1 + pow(diff["xd"].at<float>(cv::Point(x - 1, y + 1)), 2) + pow(diff["xd"].at<float>(cv::Point(x, y)), 2));
                    e[7] = 1.0f / sqrt(1 + pow(diff["y"].at<float>(cv::Point(x, y + 1)), 2) + pow(diff["y"].at<float>(cv::Point(x, y)), 2));
                    e[8] = 1.0f / sqrt(1 + pow(diff["yd"].at<float>(cv::Point(x + 1, y + 1)), 2) + pow(diff["yd"].at<float>(cv::Point(x, y)), 2));

                    channels[1].at<float>(cv::Point(x - 1, y - 1)) = bayer_buf["G"].at<float>(cv::Point(x, y))
                        + (e[1] * bayer_buf["G"].at<float>(cv::Point(x, y - 1))
                        + e[3] * bayer_buf["G"].at<float>(cv::Point(x - 1, y))
                        + e[5] * bayer_buf["G"].at<float>(cv::Point(x + 1, y))
                        + e[7] * bayer_buf["G"].at<float>(cv::Point(x, y + 1))
                            ) / (e[1] + e[3] + e[5] + e[7]);


                    channels[0].at<float>(cv::Point(x - 1, y - 1)) = bayer_buf["B"].at<float>(cv::Point(x, y)) / channels[1].at<float>(cv::Point(x - 1, y - 1)) +
                        (e[0] * bayer_buf["B"].at<float>(cv::Point(x - 1, y - 1)) / channels[1].at<float>(cv::Point(x - 1, y - 1)) 
                        // + e[1] * bayer_buf["B"].at<float>(cv::Point(x, y - 1)) / channels[1].at<float>(cv::Point(x - 1, y - 1)) 
                        + e[2] * bayer_buf["B"].at<float>(cv::Point(x + 1, y - 1)) / channels[1].at<float>(cv::Point(x - 1, y - 1)) 
                        + e[6] * bayer_buf["B"].at<float>(cv::Point(x - 1, y + 1)) / channels[1].at<float>(cv::Point(x - 1, y - 1)) 
                        + e[8] * bayer_buf["B"].at<float>(cv::Point(x + 1, y + 1)) / channels[1].at<float>(cv::Point(x - 1, y - 1)) 
                        ) / (e[0] + e[2] + e[6] + e[8]);


                }

            channels[0] = channels[0].mul(channels[1]);
            channels[1].convertTo(channels[1], CV_8UC1);
            channels[0].convertTo(channels[0], CV_8UC1);
            cv::imshow("G", channels[1]);
            cv::imshow("B", channels[0]);
            cv::waitKey(0);
        }

    } // namespace G
    namespace RB
    {
        const cv::Mat averKernel33 = (cv::Mat_<float>(3, 3) << 0.25f, 0.5f, 0.25f,
                                      0.5f, 1.0f, 0.5f,
                                      0.25f, 0.5f, 0.25f);
        inline cv::Mat linear(cv::Mat G, Type& bayer)
        {
            channelsT channels(3, cv::Mat());
            cv::filter2D(bayer["R"], channels[2], CV_8UC1, RB::averKernel33, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            cv::filter2D(bayer["B"], channels[0], CV_8UC1, RB::averKernel33, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            if (G.type() != CV_8UC1)
                G.convertTo(channels[1], CV_8UC1);
            G.copyTo(channels[1]);
            cv::Mat ret;
            cv::merge(channels, ret);
            return ret;
        }
        inline cv::Mat colorRatios(cv::Mat G, Type& bayer)
        {
            Type bayer_buf;
            bayer["R"].convertTo(bayer_buf["R"], CV_32FC1);
            G.convertTo(bayer_buf["G"], CV_32FC1);
            bayer["B"].convertTo(bayer_buf["B"], CV_32FC1);
            for (;;)
            {
                double min;
                cv::Point minLoc;
                cv::minMaxLoc(bayer_buf["G"], &min, NULL, &minLoc, NULL);
                if (min == 0)
                    bayer_buf["G"].at<float>(minLoc) = 1;
                else
                {
                    break;
                }
            }

            std::vector<cv::Mat> channels(3, cv::Mat());
            bayer_buf["B"] = bayer_buf["B"] / bayer_buf["G"];
            cv::filter2D(bayer_buf["B"], bayer_buf["B"], CV_32FC1, RB::averKernel33, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            cv::multiply(bayer_buf["B"], bayer_buf["G"], channels[0]);
            channels[0].convertTo(channels[0], CV_8UC1);

            channels[1] = bayer_buf["G"].clone();
            channels[1].convertTo(channels[1], CV_8UC1);

            bayer_buf["R"] = bayer_buf["R"] / bayer_buf["G"];
            cv::filter2D(bayer_buf["R"], bayer_buf["R"], CV_32FC1, RB::averKernel33, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            cv::multiply(bayer_buf["R"], bayer_buf["G"], channels[2]);
            channels[2].convertTo(channels[2], CV_8UC1);
            cv::Mat ret;
            cv::merge(channels, ret);
            return ret;
        }
        inline cv::Mat colorRatiosInLog(cv::Mat G, Type& bayer)
        {
            Type bayer_buf;
            bayer["R"].convertTo(bayer_buf["R"], CV_32FC1);
            G.convertTo(bayer_buf["G"], CV_32FC1);
            bayer["B"].convertTo(bayer_buf["B"], CV_32FC1);
            for (;;)
            {
                double min;
                cv::Point minLoc;
                cv::minMaxLoc(bayer_buf["G"], &min, NULL, &minLoc, NULL);
                if (min == 0)
                    bayer_buf["G"].at<float>(minLoc) = 1;
                else
                {
                    break;
                }
            }

            std::vector<cv::Mat> channels(3, cv::Mat());
            channels[1] = bayer_buf["G"].clone();
            channels[1].convertTo(channels[1], CV_8UC1);

            bayer_buf["B"] = bayer_buf["B"] / bayer_buf["G"];
            for (size_t x; x < bayer_buf["B"].cols; ++x)
            {
                for (size_t y; y < bayer_buf["B"].rows; ++y)
                {
                    bayer_buf["B"].at<float>(cv::Point(x, y)) = log(bayer_buf["B"].at<float>(cv::Point(x, y)));
                }
            }
            cv::filter2D(bayer_buf["B"], bayer_buf["B"], CV_32FC1, RB::averKernel33, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            for (size_t x; x < bayer_buf["B"].cols; ++x)
            {
                for (size_t y; y < bayer_buf["B"].rows; ++y)
                {
                    bayer_buf["B"].at<float>(cv::Point(x, y)) = exp(bayer_buf["B"].at<float>(cv::Point(x, y)));
                }
            }
            cv::multiply(bayer_buf["B"], bayer_buf["G"], channels[0]);
            channels[0].convertTo(channels[0], CV_8UC1);

            bayer_buf["R"] = bayer_buf["R"] / bayer_buf["G"];
            for (size_t x; x < bayer_buf["G"].cols; ++x)
            {
                for (size_t y; y < bayer_buf["G"].rows; ++y)
                {
                    bayer_buf["G"].at<float>(cv::Point(x, y)) = log(bayer_buf["G"].at<float>(cv::Point(x, y)));
                }
            }
            cv::filter2D(bayer_buf["R"], bayer_buf["R"], CV_32FC1, RB::averKernel33, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
            for (size_t x; x < bayer_buf["G"].cols; ++x)
            {
                for (size_t y; y < bayer_buf["G"].rows; ++y)
                {
                    bayer_buf["G"].at<float>(cv::Point(x, y)) = exp(bayer_buf["G"].at<float>(cv::Point(x, y)));
                }
            }
            cv::multiply(bayer_buf["R"], bayer_buf["G"], channels[2]);
            channels[2].convertTo(channels[2], CV_8UC1);

            cv::Mat ret;
            cv::merge(channels, ret);
            return ret;
        }
    } // namespace RB

} // namespace bayer