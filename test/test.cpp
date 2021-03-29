// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image
#include "opencv2/core/core.hpp"
#include <opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>
#include <Leap.h>
#include <imgproc.hpp>

using namespace cv;
using namespace std;
using namespace Leap;

Mat leftP, rightP;
vector<vector<Point> > po;

static void help()
{
    cout <<
        "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
        "memory storage to find squares in a list of images\n"
        "Returns sequence of squares detected on the image.\n"
        "the sequence is stored in the specified memory storage\n"
        "Call:\n"
        "./squares\n"
        "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}

int thresh = 40, N = 5;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
    squares.clear();

    //s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    //pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    //pyrUp(pyr, timg, image.size());


    // blur will enhance edge detection
    Mat timg = image.clone();

    medianBlur(image, timg, 9);
    Mat gray0(timg.size(), CV_8UC1), gray;

    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++)
    {
        int ch[] = { c, 0 };
        //basicly our data is actually gray
        //so the mixchanel function is not usable
        //mixChannels(&timg, 1, &gray0, 1, ch, 1);
        gray0 = image.clone();

        // try several threshold levels
        for (int l = 0; l < N; l++)
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0)
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 5, thresh, 5);
                //imshow("gray", gray);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1, -1));
            }
            else
            {
                // apply threshold if l!=0:
                // tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l + 1) * 255 / N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for (size_t i = 0; i < contours.size(); i++)
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++)
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }

}

// the function draws all the squares in the image
void drawSquares(Mat& image, const vector<vector<Point> >& squares)
{
    for (size_t i = 0; i < squares.size(); i++)
    {
        const Point* p = &squares[i][0];

        int n = (int)squares[i].size();
        //dont detect the border
        if (p->x > 3 && p->y > 3)
            polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
    }

    imshow(wndname, image);
}

class sampleListener : public Listener {
public:
    virtual void onInit(const Controller&);
    virtual void onConnect(const Controller&);
    virtual void onDisconnect(const Controller&);
    virtual void onExit(const Controller&);
    virtual void onFrame(const Controller&);
};
void sampleListener::onInit(const Controller& controller)
{
    cout << "Initialized" << endl;
}
void sampleListener::onConnect(const Controller& controller)
{
    cout << "Connected" << endl;
}
void sampleListener::onDisconnect(const Controller& controller)
{
    cout << "Disconnected" << endl;
}
void sampleListener::onExit(const Controller& controller)
{
    cout << "Exited" << endl;
}
void sampleListener::onFrame(const Controller& controller)
{
    const Frame frame = controller.frame();
    ImageList images = frame.images();
    Mat leftMat;
    Mat rightMat;
    if (images.count() == 2)
    {
        leftMat = Mat(images[0].height(), images[0].width(), CV_8UC1, (void*)images[0].data());
        rightMat = Mat(images[1].height(), images[1].width(), CV_8UC1, (void*)images[1].data());
        rightP = rightMat.clone();
        leftP = leftMat.clone();
        
        HandList hands = frame.hands();
        Hand hand1 = hands[0];
        Hand hand2 = hands[1];
        
        if (hand1.isValid() == true || hand2.isValid() == true) {
            cout << "found hand" << endl;
        }
        else {
            cout << "not yet" << endl;
        }

        imshow("wen",rightP);
        findSquares(rightP, po);
        drawSquares(rightP, po);
        waitKey(1);
    }
}



int main(int /*argc*/, char** /*argv*/)
{
    sampleListener listener;
    Controller leap;
    //static const char* names[] = { "C:\\Users\\Administrator\\Desktop\\test\\test\\1.jpeg",0 };
    vector<vector<Point> > squares;
    leap.addListener(listener);
    leap.setPolicy(Leap::Controller::POLICY_BACKGROUND_FRAMES);
    leap.setPolicy(Leap::Controller::POLICY_IMAGES);
    cin.get();
    leap.removeListener(listener);
    /*
    for (int i = 0; names[i] != 0; i++)
    {
        Mat image = imread(names[i], 1);
        if (image.empty())
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        findSquares(image, squares);
        drawSquares(image, squares);
        //imwrite( "out", image );
        int c = waitKey();
        if ((char)c == 27)
            break;
    }
    */
    return 0;
}