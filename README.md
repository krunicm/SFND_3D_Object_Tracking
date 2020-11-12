# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Project Specification

### FP.1 Match 3D Objects

Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

Implementation:

```
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    for (auto prevBB : prevFrame.boundingBoxes)
    {
        std::vector<cv::DMatch> enclose;

        for (cv::DMatch match : matches)
        {
            int prevIdx = match.queryIdx;
            if (prevBB.roi.contains(prevFrame.keypoints.at(prevIdx).pt))
                enclose.push_back(match);
        }

        std::multimap<int, int> record;

        for (auto candidate : enclose)
        {
            int currIDx = candidate.trainIdx;
            for (auto currBB : currFrame.boundingBoxes)
            {
                if (currBB.roi.contains(currFrame.keypoints.at(currIDx).pt))
                    record.insert(std::pair<int, int>(currBB.boxID, currIDx));
            }
        }

        int maxOccurrences = 0;
        int index = 0;

        for (auto rec : record)
        {
            if (record.count(rec.first) > maxOccurrences)
            {
                maxOccurrences = record.count(rec.first);
                index = rec.first;
            }
        }

        bbBestMatches.insert(std::pair<int, int>(prevBB.boxID, index));
    }
}
```

### FP.2 Compute Lidar-based TTC

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

```
void kNearestPointsMedianValue(vector<double> minXPVector, int& kNearest, double& medianValue)
{
    typedef vector<double>::size_type vec_sz;

    sort(minXPVector.begin(), minXPVector.end());
    vector<double> kNearestVector;
    kNearestVector.insert(kNearestVector.begin(), minXPVector.begin(), next(minXPVector.begin(), kNearest));

    vec_sz size = kNearestVector.size();
    vec_sz mid = size/2;

    medianValue = kNearest % 2 == 0 ? (kNearestVector[mid] + kNearestVector[mid-1]) / 2 : kNearestVector[mid];
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 0.1;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane
    double scope = (laneWidth-0.2) / 2;
    std::vector<double> prevPoints;
    std::vector<double> currPoints;

    auto checkFunc = [&scope](const LidarPoint &lp){return abs(lp.y) >= scope;};

    lidarPointsPrev.erase(std::remove_if(lidarPointsPrev.begin(), lidarPointsPrev.end(), checkFunc), lidarPointsPrev.end());
    lidarPointsCurr.erase(std::remove_if(lidarPointsCurr.begin(), lidarPointsCurr.end(), checkFunc), lidarPointsCurr.end());

    for (auto prevPoint : lidarPointsPrev)
        prevPoints.push_back(prevPoint.x);

    for (auto currPoint : lidarPointsCurr)
        currPoints.push_back(currPoint.x);

    int kNearestPoints = 100;
    double kMinXCurr = 0;
    double kMinXPrev = 0;

    kNearestPointsMedianValue(currPoints, kNearestPoints, kMinXCurr);
    kNearestPointsMedianValue(prevPoints, kNearestPoints, kMinXPrev);

    // compute TTC from both measurements
    TTC = kMinXCurr * dT / (kMinXPrev - kMinXCurr);

    cout << "Median min current:  " << kMinXCurr << endl;
    cout << "Median min previous: " << kMinXPrev << endl;
    cout << "Lidar TTC: " << TTC << "s" << endl;
}

```

### FP.3 Associate Keypoint Correspondences with Bounding Boxes

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

```
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<double> euclidian;

    for (cv::DMatch match : kptMatches)
    {
        int currIdx = match.trainIdx;
        if (boundingBox.roi.contains(kptsCurr[currIdx].pt))
        {
            int prevIdx = match.queryIdx;
            euclidian.push_back(cv::norm(kptsCurr[currIdx].pt - kptsPrev[prevIdx].pt));
        }
            
    }

    double euclidianMean = std::accumulate(euclidian.begin(), euclidian.end(), 0)/euclidian.size();

    for (cv::DMatch match : kptMatches)
    {
        int currIdx = match.trainIdx;
        if (boundingBox.roi.contains(kptsCurr[currIdx].pt))
        {
            int prevIdx = match.queryIdx;
            int distance = cv::norm(kptsCurr[currIdx].pt - kptsPrev[prevIdx].pt);

            if (distance <= euclidianMean)
            {
                boundingBox.keypoints.push_back(kptsCurr[currIdx]);
                boundingBox.kptMatches.push_back(match);
            }
        }
    }

    cout << "Mean value: " << euclidianMean << endl;
    cout << "Before filtering there are: " << euclidian.size() << endl;
    cout << "After filtering, there are: " << boundingBox.keypoints.size() << endl;
}

```

### FP.4 Compute Camera-based TTC

Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

```
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK
}
```

### FP.5 Performance Evaluation 1

Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

1. example

0000000007.png \
Median min current:  7.555 \
Median min previous: 7.577 \
Lidar TTC: 34.3404s 

<img src="images/lidar_ttc_07.png" width="779" height="414" />

Explanation:
Current and previous distance are to close which cause TTC to rise, since in calculus divider is "kMinXPrev - kMinXCurr".

2. example

0000000010.png \
Median min current:  7.4315 \
Median min previous: 7.434 \
Lidar TTC: 297.282s 

<img src="images/lidar_ttc_10.png" width="779" height="414" />

Explanation:
The closer two values are (kMinXPrev - kMinXCurr), will make TTC to strive to infinit.

3. example

0000000012.png \
Median min current:  7.272 \
Median min previous: 7.205 \
Lidar TTC: -10.8537s 

Explanation:
In this case current distance is bigger then the previous which imply that vehicles will never colide therefore negativ value for TTC.

### FP.6 Performance Evaluation 2

Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.


Combinations Detector/Descriptor 



### 1. Combo **SHITOMASI/BRISK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 15.0726 | 0.88871 | 
| 12.311 | 13.2428 | 0.931783 | 
| 38.3338 | 14.3921 | 23.9417 | 
| 13.3334 | 12.8477 | 0.485645 | 
| 8.91732 | 12.8285 | 3.91114 | 
| 11.8589 | 11.3009 | 0.558061 | 
| 14.0648 | 12.1551 | 1.90968 | 
| 19.8869 | 13.6508 | 6.23615 | 
| 12.7085 | 10.8869 | 1.82162 | 
| 16.0247 | 35.585 | 19.5602 | 
| 11.2763 | 12.8 | 1.52367 | 
| 9.88103 | 11.5836 | 1.70256 | 
| 8.8171 | 11.1144 | 2.29725 | 
| 9.60472 | 12.5097 | 2.90501 | 
| 7.93987 | 10.225 | 2.2851 | 
| 8.95961 | 11.414 | 2.45437 | 
| 11.4513 | 10.6233 | 0.827947 | 
| 10.397 | 10.2699 | 0.127066 | 

Median difference between Lidar and Camera TTC: 1.86565s \
Processing time: 10.4263s 

KPI = 100 / ( Processing time + TTC difference ) = 8.13538


### 2. Combo **SHITOMASI/BRIEF**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 14.6756 | 0.491736 | 
| 12.311 | 13.9717 | 1.66068 | 
| 38.3338 | 9.73978 | 28.594 | 
| 13.3334 | 14.982 | 1.64864 | 
| 8.91732 | 12.7628 | 3.84548 | 
| 11.8589 | 13.2703 | 1.41138 | 
| 14.0648 | 15.2664 | 1.20158 | 
| 19.8869 | 12.0847 | 7.80218 | 
| 12.7085 | 11.8703 | 0.838168 | 
| 16.0247 | 12.6285 | 3.39628 | 
| 11.2763 | 11.8507 | 0.574338 | 
| 9.88103 | 11.7642 | 1.88319 | 
| 8.8171 | 11.7197 | 2.90258 | 
| 9.60472 | 11.3557 | 1.75098 | 
| 7.93987 | 12.1983 | 4.25839 | 
| 8.95961 | 8.23961 | 0.719999 | 
| 11.4513 | 11.1382 | 0.313066 | 
| 10.397 | 8.43097 | 1.96601 | 

Median difference between Lidar and Camera TTC: 1.7863s \
Processing time: 9.4687s 

KPI = 100 / ( Processing time + TTC difference ) = 8.88494


### 3. Combo **SHITOMASI/ORB**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 13.8743 | 0.30955 | 
| 12.311 | 11.0038 | 1.30722 | 
| 38.3338 | 12.8155 | 25.5183 | 
| 13.3334 | 12.4814 | 0.851976 | 
| 8.91732 | 12.5187 | 3.60141 | 
| 11.8589 | 13.2198 | 1.36083 | 
| 14.0648 | 13.0506 | 1.01421 | 
| 19.8869 | 11.8822 | 8.00473 | 
| 12.7085 | 11.4134 | 1.29511 | 
| 16.0247 | 13.9512 | 2.07359 | 
| 11.2763 | 11.5174 | 0.241031 | 
| 9.88103 | 11.5712 | 1.69015 | 
| 8.8171 | 11.6341 | 2.81699 | 
| 9.60472 | 10.8747 | 1.26994 | 
| 7.93987 | 10.266 | 2.32618 | 
| 8.95961 | 8.17853 | 0.781077 | 
| 11.4513 | 9.68906 | 1.76219 | 
| 10.397 | 8.11259 | 2.28439 | 

Median difference between Lidar and Camera TTC: 1.72677s \
Processing time: 10.9314s 

KPI = 100 / ( Processing time + TTC difference ) = 7.90004


### 4. Combo **SHITOMASI/FREAK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 16.3505 | 2.16659 | 
| 12.311 | 13.3128 | 1.00179 | 
| 38.3338 | 11.4627 | 26.8711 | 
| 13.3334 | 13.0422 | 0.291176 | 
| 8.91732 | 12.9056 | 3.98828 | 
| 11.8589 | 14.6025 | 2.74358 | 
| 14.0648 | 11.1458 | 2.91901 | 
| 19.8869 | 12.0493 | 7.83765 | 
| 12.7085 | 11.2348 | 1.47372 | 
| 16.0247 | 13.2947 | 2.73006 | 
| 11.2763 | 11.4787 | 0.20241 | 
| 9.88103 | 11.6118 | 1.73074 | 
| 8.8171 | 11.9459 | 3.12876 | 
| 9.60472 | 12.1906 | 2.58588 | 
| 7.93987 | 11.6886 | 3.74869 | 
| 8.95961 | 10.7084 | 1.74876 | 
| 11.4513 | 10.9452 | 0.506092 | 
| 10.397 | 10.6677 | 0.270681 | 

Median difference between Lidar and Camera TTC: 1.75659s \
Processing time: 11.8594s 

KPI = 100 / ( Processing time + TTC difference ) = 7.34431


### 5. Combo **SHITOMASI/AKAZE**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 6. Combo **SHITOMASI/SIFT**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

Assertion failed

### 7. Combo **HARRIS/BRISK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 10.9082 | 3.27565 | 
| 12.311 | 52.7952 | 40.4842 | 
| 38.3338 | -11.4731 | 49.8069 | 
| 13.3334 | 11.3951 | 1.93823 | 
| 8.91732 | 44.9166 | 35.9993 | 
| 11.8589 | 12.9945 | 1.13555 | 
| 14.0648 | 13.497 | 0.567802 | 
| 19.8869 | 17.6204 | 2.26653 | 
| 12.7085 | 0.136149 | 12.5723 | 
| 16.0247 | -153.93 | 169.954 | 
| 11.2763 | 11.7414 | 0.465097 | 
| 9.88103 | 11.6948 | 1.81378 | 
| 8.8171 | 284.161 | 275.344 | 
| 9.60472 | 7.72144 | 1.88327 | 
| 7.93987 | -12.639 | 20.5789 | 
| 8.95961 | 7.09479 | 1.86482 | 
| 11.4513 | 12.5848 | 1.13359 | 
| 10.397 | 0.687055 | 9.70992 | 

Median difference between Lidar and Camera TTC: 1.88323s \
Processing time: 14.4662s 

KPI = 100 / ( Processing time + TTC difference ) = 6.11642


### 8. Combo **HARRIS/BRIEF**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | -inf | inf | 
| 12.311 | 63.8475 | 51.5365 | 
| 38.3338 | -11.4731 | 49.8069 | 
| 13.3334 | 11.5792 | 1.75416 | 
| 8.91732 | 35.3833 | 26.4659 | 
| 11.8589 | 15.2483 | 3.38933 | 
| 14.0648 | 14.2744 | 0.209623 | 
| 19.8869 | 17.6204 | 2.26653 | 
| 12.7085 | 3.93864 | 8.76983 | 
| 16.0247 | 20.5862 | 4.56147 | 
| 11.2763 | 11.1552 | 0.121169 | 
| 9.88103 | -inf | inf | 
| 8.8171 | 13.4327 | 4.61563 | 
| 9.60472 | 5.66097 | 3.94375 | 
| 7.93987 | -13.6263 | 21.5662 | 
| 8.95961 | 6.7641 | 2.19551 | 
| 11.4513 | 12.5848 | 1.13359 | 
| 10.397 | 12.8381 | 2.44116 | 

Median difference between Lidar and Camera TTC: 2.12009s \
Processing time: 11.3883s 

KPI = 100 / ( Processing time + TTC difference ) = 7.40283


### 9. Combo **HARRIS/ORB**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | -inf | inf | 
| 12.311 | 63.8475 | 51.5365 | 
| 38.3338 | -11.4731 | 49.8069 | 
| 13.3334 | 11.5792 | 1.75416 | 
| 8.91732 | 13.6432 | 4.72588 | 
| 11.8589 | nan | nan | 
| 14.0648 | 13.497 | 0.567802 | 
| 19.8869 | 17.6204 | 2.26653 | 
| 12.7085 | 3.93864 | 8.76983 | 
| 16.0247 | -inf | inf | 
| 11.2763 | 11.6702 | 0.393899 | 
| 9.88103 | 11.6948 | 1.81378 | 
| 8.8171 | 13.4327 | 4.61563 | 
| 9.60472 | 6.06984 | 3.53488 | 
| 7.93987 | -inf | inf | 
| 8.95961 | 6.71705 | 2.24255 | 
| 11.4513 | 12.5848 | 1.13359 | 
| 10.397 | -inf | inf | 

Median difference between Lidar and Camera TTC: 2.28474s \
Processing time: 11.544s 

KPI = 100 / ( Processing time + TTC difference ) = 7.23132


### 10. Combo **HARRIS/FREAK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 9.74953 | 4.43434 | 
| 12.311 | 18.2178 | 5.90676 | 
| 38.3338 | -11.4731 | 49.8069 | 
| 13.3334 | 12.1284 | 1.20499 | 
| 8.91732 | -192.17 | 201.088 | 
| 11.8589 | 15.2483 | 3.38933 | 
| 14.0648 | 13.342 | 0.72281 | 
| 19.8869 | 12.9162 | 6.9707 | 
| 12.7085 | 0.136149 | 12.5723 | 
| 16.0247 | 10.2931 | 5.73163 | 
| 11.2763 | 11.0967 | 0.179608 | 
| 9.88103 | -inf | inf | 
| 8.8171 | 13.4095 | 4.5924 | 
| 9.60472 | 12.288 | 2.68327 | 
| 7.93987 | -12.4873 | 20.4271 | 
| 8.95961 | 6.65726 | 2.30235 | 
| 11.4513 | 11.7964 | 0.345128 | 
| 10.397 | 25.6763 | 15.2793 | 

Median difference between Lidar and Camera TTC: 2.29118s \
Processing time: 11.7394s 

KPI = 100 / ( Processing time + TTC difference ) = 7.12731


### 11. Combo **HARRIS/AKAZE**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 12. Combo **HARRIS/SIFT**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

Assertion failed

### 13. Combo **FAST/BRISK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 12.3 | 1.88387 | 
| 12.311 | 12.309 | 0.00202877 | 
| 38.3338 | 14.3428 | 23.991 | 
| 13.3334 | 12.7004 | 0.632983 | 
| 8.91732 | 47.3039 | 38.3866 | 
| 11.8589 | 13.2481 | 1.38912 | 
| 14.0648 | 29.3182 | 15.2535 | 
| 19.8869 | 11.2085 | 8.67839 | 
| 12.7085 | 13.375 | 0.666515 | 
| 16.0247 | 13.0801 | 2.94466 | 
| 11.2763 | 14.1886 | 2.91223 | 
| 9.88103 | 11.4832 | 1.60219 | 
| 8.8171 | 12.3705 | 3.55338 | 
| 9.60472 | 12.0431 | 2.43837 | 
| 7.93987 | 11.7034 | 3.76354 | 
| 8.95961 | 11.9434 | 2.98375 | 
| 11.4513 | 8.23767 | 3.21359 | 
| 10.397 | 11.7502 | 1.35325 | 

Median difference between Lidar and Camera TTC: 2.31427s \
Processing time: 15.0954s 

KPI = 100 / ( Processing time + TTC difference ) = 5.74393


### 14. Combo **FAST/BRIEF**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 10.7101 | 3.47381 | 
| 12.311 | 11.0063 | 1.30465 | 
| 38.3338 | 14.1559 | 24.1779 | 
| 13.3334 | 14.3886 | 1.05522 | 
| 8.91732 | 20.0387 | 11.1214 | 
| 11.8589 | 13.293 | 1.43406 | 
| 14.0648 | 12.2182 | 1.84662 | 
| 19.8869 | 12.7596 | 7.12733 | 
| 12.7085 | 12.6 | 0.108472 | 
| 16.0247 | 13.4681 | 2.55668 | 
| 11.2763 | 13.7533 | 2.47699 | 
| 9.88103 | 10.974 | 1.09294 | 
| 8.8171 | 12.3343 | 3.51721 | 
| 9.60472 | 11.2431 | 1.63843 | 
| 7.93987 | 11.8747 | 3.93479 | 
| 8.95961 | 11.8398 | 2.88023 | 
| 11.4513 | 7.92013 | 3.53113 | 
| 10.397 | 11.554 | 1.15701 | 

Median difference between Lidar and Camera TTC: 2.38228s \
Processing time: 11.8299s 

KPI = 100 / ( Processing time + TTC difference ) = 7.03621


### 15. Combo **FAST/ORB**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 11.0544 | 3.12946 | 
| 12.311 | 10.6351 | 1.67589 | 
| 38.3338 | 13.4008 | 24.933 | 
| 13.3334 | 12.7877 | 0.545722 | 
| 8.91732 | 18.0923 | 9.17498 | 
| 11.8589 | 12.9892 | 1.13024 | 
| 14.0648 | 12.4642 | 1.6006 | 
| 19.8869 | 11.3522 | 8.53472 | 
| 12.7085 | 12.1119 | 0.596549 | 
| 16.0247 | 13.4637 | 2.561 | 
| 11.2763 | 13.1 | 1.82367 | 
| 9.88103 | 10.4239 | 0.542891 | 
| 8.8171 | 12.0462 | 3.22908 | 
| 9.60472 | 11.0103 | 1.40558 | 
| 7.93987 | 11.4079 | 3.46799 | 
| 8.95961 | 11.6 | 2.64039 | 
| 11.4513 | 7.57979 | 3.87147 | 
| 10.397 | 10.5903 | 0.193281 | 

Median difference between Lidar and Camera TTC: 2.38228s \
Processing time: 12.3014s 

KPI = 100 / ( Processing time + TTC difference ) = 6.8103


### 16. Combo **FAST/FREAK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 12.3 | 1.88387 | 
| 12.311 | 14.2376 | 1.92659 | 
| 38.3338 | 14.4553 | 23.8785 | 
| 13.3334 | 14.0766 | 0.743225 | 
| 8.91732 | 14.4739 | 5.55653 | 
| 11.8589 | 12.435 | 0.576023 | 
| 14.0648 | 13.4674 | 0.597358 | 
| 19.8869 | 11.819 | 8.0679 | 
| 12.7085 | 12.8498 | 0.141335 | 
| 16.0247 | 13.4169 | 2.60785 | 
| 11.2763 | 13.1788 | 1.90244 | 
| 9.88103 | 11.5256 | 1.64452 | 
| 8.8171 | 11.7063 | 2.88919 | 
| 9.60472 | 11.4 | 1.79529 | 
| 7.93987 | 11.3869 | 3.44704 | 
| 8.95961 | 12.305 | 3.34542 | 
| 11.4513 | 8.61022 | 2.84104 | 
| 10.397 | 11.5066 | 1.10958 | 

Median difference between Lidar and Camera TTC: 2.31427s \
Processing time: 12.8872s 

KPI = 100 / ( Processing time + TTC difference ) = 6.57833


### 17. Combo **FAST/AKAZE**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 18. Combo **FAST/SIFT**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

Assertion failed

### 19. Combo **BRISK/BRISK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 13.9354 | 0.248439 | 
| 12.311 | 14.6313 | 2.32025 | 
| 38.3338 | 11.9318 | 26.402 | 
| 13.3334 | 15.5311 | 2.19773 | 
| 8.91732 | 26.8377 | 17.9204 | 
| 11.8589 | 14.2475 | 2.38852 | 
| 14.0648 | 18.1356 | 4.0708 | 
| 19.8869 | 17.1048 | 2.7821 | 
| 12.7085 | 15.1063 | 2.39788 | 
| 16.0247 | 14.2575 | 1.76727 | 
| 11.2763 | 13.0458 | 1.76949 | 
| 9.88103 | 11.5799 | 1.69884 | 
| 8.8171 | 12.6288 | 3.81173 | 
| 9.60472 | 11.3714 | 1.76664 | 
| 7.93987 | 12.1932 | 4.2533 | 
| 8.95961 | 10.3542 | 1.39463 | 
| 11.4513 | 9.3492 | 2.10205 | 
| 10.397 | 10.5758 | 0.178837 | 

Median difference between Lidar and Camera TTC: 2.3113s \
Processing time: 20.9528s 

KPI = 100 / ( Processing time + TTC difference ) = 4.29848


### 20. Combo **BRISK/BRIEF**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 13.7134 | 0.470428 | 
| 12.311 | 18.4095 | 6.09852 | 
| 38.3338 | 12.4078 | 25.926 | 
| 13.3334 | 19.8631 | 6.52971 | 
| 8.91732 | 27.574 | 18.6567 | 
| 11.8589 | 19.1942 | 7.33523 | 
| 14.0648 | 17.9009 | 3.83612 | 
| 19.8869 | 18.3884 | 1.49855 | 
| 12.7085 | 14.9617 | 2.25319 | 
| 16.0247 | 12.0464 | 3.97837 | 
| 11.2763 | 11.5343 | 0.257967 | 
| 9.88103 | 14.3096 | 4.4286 | 
| 8.8171 | 12.8926 | 4.07547 | 
| 9.60472 | 10.3194 | 0.714673 | 
| 7.93987 | 11.1036 | 3.16376 | 
| 8.95961 | 13.047 | 4.08741 | 
| 11.4513 | 11.1959 | 0.255328 | 
| 10.397 | 9.60206 | 0.794908 | 

Median difference between Lidar and Camera TTC: 2.35735s \
Processing time: 17.0978s 

KPI = 100 / ( Processing time + TTC difference ) = 5.14004


### 21. Combo **BRISK/ORB**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 13.8466 | 0.337259 | 
| 12.311 | 14.8464 | 2.53539 | 
| 38.3338 | 13.4152 | 24.9186 | 
| 13.3334 | 19.9039 | 6.57056 | 
| 8.91732 | 20.2689 | 11.3516 | 
| 11.8589 | 15.573 | 3.71407 | 
| 14.0648 | 19.0353 | 4.97049 | 
| 19.8869 | 15.4934 | 4.39354 | 
| 12.7085 | 12.5426 | 0.16583 | 
| 16.0247 | 12.5749 | 3.44989 | 
| 11.2763 | 12.6888 | 1.41244 | 
| 9.88103 | 11.1215 | 1.24046 | 
| 8.8171 | 11.4522 | 2.63515 | 
| 9.60472 | 10.7668 | 1.16211 | 
| 7.93987 | 13.0559 | 5.11598 | 
| 8.95961 | 10.3409 | 1.38126 | 
| 11.4513 | 9.62679 | 1.82447 | 
| 10.397 | 10.3842 | 0.0128146 | 

Median difference between Lidar and Camera TTC: 2.3932s \
Processing time: 17.1109s 

KPI = 100 / ( Processing time + TTC difference ) = 5.12712


### 22. Combo **BRISK/FREAK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 14.0521 | 0.131779 | 
| 12.311 | 22.3822 | 10.0712 | 
| 38.3338 | 15.4475 | 22.8863 | 
| 13.3334 | 13.5966 | 0.263185 | 
| 8.91732 | 24.9967 | 16.0794 | 
| 11.8589 | 13.8709 | 2.01196 | 
| 14.0648 | 14.3446 | 0.279829 | 
| 19.8869 | 17.0377 | 2.84924 | 
| 12.7085 | 18.1204 | 5.41193 | 
| 16.0247 | 13.5758 | 2.44898 | 
| 11.2763 | 12.1363 | 0.860001 | 
| 9.88103 | 12.5849 | 2.70386 | 
| 8.8171 | 12.3611 | 3.544 | 
| 9.60472 | 11.5847 | 1.98 | 
| 7.93987 | 12.3974 | 4.45757 | 
| 8.95961 | 9.98764 | 1.02803 | 
| 11.4513 | 8.8349 | 2.61636 | 
| 10.397 | 10.6095 | 0.212509 | 

Median difference between Lidar and Camera TTC: 2.41812s \
Processing time: 17.361s 

KPI = 100 / ( Processing time + TTC difference ) = 5.05584


### 23. Combo **BRISK/AKAZE**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 24. Combo **BRISK/SIFT**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

Assertion failed

### 25. Combo **ORB/BRISK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 17.4836 | 3.29971 | 
| 12.311 | 15.5366 | 3.2256 | 
| 38.3338 | 19.2234 | 19.1104 | 
| 13.3334 | 21.3012 | 7.96784 | 
| 8.91732 | 73.2489 | 64.3315 | 
| 11.8589 | 12.6176 | 0.758707 | 
| 14.0648 | 17.36 | 3.2952 | 
| 19.8869 | 11.1 | 8.78688 | 
| 12.7085 | 11.9786 | 0.729902 | 
| 16.0247 | 13.3055 | 2.71928 | 
| 11.2763 | 8.49484 | 2.78149 | 
| 9.88103 | 5416.2 | 5406.32 | 
| 8.8171 | 8.33753 | 0.479563 | 
| 9.60472 | 9.26013 | 0.34459 | 
| 7.93987 | 9.65484 | 1.71498 | 
| 8.95961 | 9.37082 | 0.411212 | 
| 11.4513 | 14.3156 | 2.86439 | 
| 10.397 | 20.1038 | 9.70687 | 

Median difference between Lidar and Camera TTC: 2.46568s \
Processing time: 15.5438s 

KPI = 100 / ( Processing time + TTC difference ) = 5.55264


### 26. Combo **ORB/BRIEF**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 16.4808 | 2.29694 | 
| 12.311 | 16.969 | 4.65797 | 
| 38.3338 | 30.925 | 7.40877 | 
| 13.3334 | 25.2172 | 11.8838 | 
| 8.91732 | 25.429 | 16.5117 | 
| 11.8589 | 18.2578 | 6.39886 | 
| 14.0648 | 40.0236 | 25.9588 | 
| 19.8869 | 31.8478 | 11.9609 | 
| 12.7085 | 188.579 | 175.871 | 
| 16.0247 | 13.1476 | 2.87718 | 
| 11.2763 | 10.4394 | 0.836933 | 
| 9.88103 | 16.6093 | 6.72823 | 
| 8.8171 | 10.7998 | 1.98268 | 
| 9.60472 | 8.40597 | 1.19874 | 
| 7.93987 | 12.491 | 4.55116 | 
| 8.95961 | 13.4955 | 4.53589 | 
| 11.4513 | 17.2956 | 5.84435 | 
| 10.397 | 13.275 | 2.87799 | 

Median difference between Lidar and Camera TTC: 2.59687s \
Processing time: 11.8627s 

KPI = 100 / ( Processing time + TTC difference ) = 6.91582


### 27. Combo **ORB/ORB**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 24.7809 | 10.5971 | 
| 12.311 | 10.126 | 2.18498 | 
| 38.3338 | 18.5308 | 19.803 | 
| 13.3334 | 26.6829 | 13.3495 | 
| 8.91732 | 25.7718 | 16.8545 | 
| 11.8589 | 10.7829 | 1.07602 | 
| 14.0648 | 41.874 | 27.8092 | 
| 19.8869 | 10.9114 | 8.97547 | 
| 12.7085 | 18.651 | 5.94252 | 
| 16.0247 | 16.6469 | 0.622197 | 
| 11.2763 | 8.48098 | 2.79535 | 
| 9.88103 | 30.9371 | 21.0561 | 
| 8.8171 | 9.26406 | 0.446962 | 
| 9.60472 | 9.5359 | 0.068823 | 
| 7.93987 | 12.9444 | 5.00453 | 
| 8.95961 | 9.65407 | 0.694459 | 
| 11.4513 | 25.3838 | 13.9325 | 
| 10.397 | 21.8102 | 11.4132 | 

Median difference between Lidar and Camera TTC: 2.63777s \
Processing time: 12.006s 

KPI = 100 / ( Processing time + TTC difference ) = 6.82884


### 28. Combo **ORB/FREAK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 11.9371 | 2.24675 | 
| 12.311 | 25.2016 | 12.8906 | 
| 38.3338 | 17.7441 | 20.5897 | 
| 13.3334 | 11.2797 | 2.0537 | 
| 8.91732 | 52.139 | 43.2216 | 
| 11.8589 | 35.7942 | 23.9353 | 
| 14.0648 | 127.591 | 113.526 | 
| 19.8869 | 9.38588 | 10.501 | 
| 12.7085 | 13.3292 | 0.620681 | 
| 16.0247 | -545.044 | 561.068 | 
| 11.2763 | 8.40781 | 2.86852 | 
| 9.88103 | -inf | inf | 
| 8.8171 | 7.03048 | 1.78661 | 
| 9.60472 | 55.7876 | 46.1829 | 
| 7.93987 | 8.72363 | 0.783759 | 
| 8.95961 | 11.1021 | 2.14246 | 
| 11.4513 | -inf | inf | 
| 10.397 | 24.1855 | 13.7885 | 

Median difference between Lidar and Camera TTC: 2.71157s \
Processing time: 11.8024s 

KPI = 100 / ( Processing time + TTC difference ) = 6.88993


### 29. Combo **ORB/AKAZE**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 30. Combo **ORB/SIFT**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

Assertion failed

### 31. Combo **AKAZE/BRISK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 32. Combo **AKAZE/BRIEF**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 33. Combo **AKAZE/ORB**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 34. Combo **AKAZE/FREAK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 35. Combo **AKAZE/AKAZE**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 12.4563 | 1.7276 | 
| 12.311 | 14.7901 | 2.47909 | 
| 38.3338 | 12.949 | 25.3848 | 
| 13.3334 | 14.6163 | 1.28292 | 
| 8.91732 | 16.4067 | 7.48939 | 
| 11.8589 | 13.2509 | 1.39193 | 
| 14.0648 | 15.2119 | 1.14712 | 
| 19.8869 | 14.5567 | 5.33019 | 
| 12.7085 | 14.1308 | 1.4223 | 
| 16.0247 | 11.5876 | 4.4371 | 
| 11.2763 | 12.1869 | 0.910528 | 
| 9.88103 | 11.274 | 1.39295 | 
| 8.8171 | 10.6701 | 1.85298 | 
| 9.60472 | 10.4339 | 0.829229 | 
| 7.93987 | 10.5929 | 2.653 | 
| 8.95961 | 10.0937 | 1.13413 | 
| 11.4513 | 9.35968 | 2.09158 | 
| 10.397 | 9.00572 | 1.39125 | 

Median difference between Lidar and Camera TTC: 2.6121s \
Processing time: 13.0777s 

KPI = 100 / ( Processing time + TTC difference ) = 6.37356


### 36. Combo **AKAZE/SIFT**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 37. Combo **SIFT/BRISK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 13.3321 | 0.851757 | 
| 12.311 | 12.8716 | 0.560598 | 
| 38.3338 | 12.7195 | 25.6143 | 
| 13.3334 | 19.1649 | 5.83151 | 
| 8.91732 | 17.9358 | 9.01846 | 
| 11.8589 | 10.4924 | 1.36658 | 
| 14.0648 | 14.619 | 0.554217 | 
| 19.8869 | 12.3717 | 7.51525 | 
| 12.7085 | 13.3507 | 0.642276 | 
| 16.0247 | 10.0383 | 5.98643 | 
| 11.2763 | 11.2625 | 0.0138236 | 
| 9.88103 | 10.5916 | 0.710547 | 
| 8.8171 | 9.53066 | 0.71356 | 
| 9.60472 | 10.644 | 1.03928 | 
| 7.93987 | 9.48179 | 1.54192 | 
| 8.95961 | 9.64438 | 0.684774 | 
| 11.4513 | 8.86971 | 2.58155 | 
| 10.397 | 9.31625 | 1.08073 | 

Median difference between Lidar and Camera TTC: 2.55884s \
Processing time: 16.4775s 

KPI = 100 / ( Processing time + TTC difference ) = 5.25311


### 38. Combo **SIFT/BRIEF**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 12.0691 | 2.11479 | 
| 12.311 | 12.6771 | 0.366059 | 
| 38.3338 | 13.9155 | 24.4183 | 
| 13.3334 | 21.5634 | 8.22999 | 
| 8.91732 | 14.1642 | 5.24693 | 
| 11.8589 | 11.0631 | 0.795873 | 
| 14.0648 | 13.1317 | 0.933122 | 
| 19.8869 | 16.1708 | 3.71607 | 
| 12.7085 | 13.2586 | 0.550105 | 
| 16.0247 | 10.331 | 5.69376 | 
| 11.2763 | 12.0448 | 0.768458 | 
| 9.88103 | 9.75529 | 0.125742 | 
| 8.8171 | 9.62883 | 0.81173 | 
| 9.60472 | 9.3534 | 0.251315 | 
| 7.93987 | 8.92271 | 0.982838 | 
| 8.95961 | 8.86647 | 0.093135 | 
| 11.4513 | 8.61572 | 2.83553 | 
| 10.397 | 9.41223 | 0.984744 | 

Median difference between Lidar and Camera TTC: 2.47804s \
Processing time: 13.2759s 

KPI = 100 / ( Processing time + TTC difference ) = 6.34762


### 39. Combo **SIFT/ORB**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

OutOfMemoryError

### 40. Combo **SIFT/FREAK**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 14.1839 | 11.5787 | 2.60521 | 
| 12.311 | 13.0746 | 0.763601 | 
| 38.3338 | 12.9327 | 25.4011 | 
| 13.3334 | 19.6967 | 6.36336 | 
| 8.91732 | 13.5618 | 4.64445 | 
| 11.8589 | 11.0118 | 0.847163 | 
| 14.0648 | 14.3184 | 0.253574 | 
| 19.8869 | 12.6281 | 7.25879 | 
| 12.7085 | 15.2314 | 2.52292 | 
| 16.0247 | 9.8725 | 6.15224 | 
| 11.2763 | 11.2887 | 0.0124095 | 
| 9.88103 | 10.5844 | 0.703413 | 
| 8.8171 | 9.14642 | 0.32932 | 
| 9.60472 | 9.44517 | 0.159544 | 
| 7.93987 | 9.07095 | 1.13108 | 
| 8.95961 | 9.65444 | 0.694831 | 
| 11.4513 | 8.63926 | 2.812 | 
| 10.397 | 9.02737 | 1.3696 | 

Median difference between Lidar and Camera TTC: 2.46568s \
Processing time: 13.7884s 

KPI = 100 / ( Processing time + TTC difference ) = 6.15231


### 41. Combo **SIFT/AKAZE**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

### 42. Combo **SIFT/SIFT**

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| N/A | N/A | N/A |

Assertion failed


-----------------------------------------------------------------------------

| | Detector | Descriptor | TTC Difference (s) | Performance (s) |     KPI     |
| ----------- | ----------- | ----------- | ----------- |----------- |----------- |
| 1. | SHITOMASI | BRIEF | 1.7863 | 9.4687 | 8.88494 | 
| 2. | SHITOMASI | BRISK | 1.86565 | 10.4263 | 8.13538 | 
| 3. | SHITOMASI | ORB | 1.72677 | 10.9314 | 7.90004 | 
| 4. | HARRIS | BRIEF | 2.12009 | 11.3883 | 7.40283 | 
| 5. | SHITOMASI | FREAK | 1.75659 | 11.8594 | 7.34431 | 
| 6. | HARRIS | ORB | 2.28474 | 11.544 | 7.23132 | 
| 7. | HARRIS | FREAK | 2.29118 | 11.7394 | 7.12731 | 
| 8. | FAST | BRIEF | 2.38228 | 11.8299 | 7.03621 | 
| 9. | ORB | BRIEF | 2.59687 | 11.8627 | 6.91582 | 
| 10. | ORB | FREAK | 2.71157 | 11.8024 | 6.88993 | 
| 11. | ORB | ORB | 2.63777 | 12.006 | 6.82884 | 
| 12. | FAST | ORB | 2.38228 | 12.3014 | 6.8103 | 
| 13. | FAST | FREAK | 2.31427 | 12.8872 | 6.57833 | 
| 14. | AKAZE | AKAZE | 2.6121 | 13.0777 | 6.37356 | 
| 15. | SIFT | BRIEF | 2.47804 | 13.2759 | 6.34762 | 
| 16. | SIFT | FREAK | 2.46568 | 13.7884 | 6.15231 | 
| 17. | HARRIS | BRISK | 1.88323 | 14.4662 | 6.11642 | 
| 18. | FAST | BRISK | 2.31427 | 15.0954 | 5.74393 | 
| 19. | ORB | BRISK | 2.46568 | 15.5438 | 5.55264 | 
| 20. | SIFT | BRISK | 2.55884 | 16.4775 | 5.25311 | 
| 21. | BRISK | BRIEF | 2.35735 | 17.0978 | 5.14004 | 
| 22. | BRISK | ORB | 2.3932 | 17.1109 | 5.12712 | 
| 23. | BRISK | FREAK | 2.41812 | 17.361 | 5.05584 | 
| 24. | BRISK | BRISK | 2.3113 | 20.9528 | 4.29848 | 