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
    std::vector<double> minXPrevList;
    std::vector<double> minXCurrList;

    auto checkFunc = [&scope](const LidarPoint &lp){return abs(lp.y) >= scope;};

    lidarPointsPrev.erase(std::remove_if(lidarPointsPrev.begin(), lidarPointsPrev.end(), checkFunc), lidarPointsPrev.end());
    lidarPointsCurr.erase(std::remove_if(lidarPointsCurr.begin(), lidarPointsCurr.end(), checkFunc), lidarPointsCurr.end());

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        minXPrev = minXPrev > it->x ? it->x : minXPrev;
        minXPrevList.push_back(minXPrev);
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        minXCurr = minXCurr > it->x ? it->x : minXCurr;
        minXCurrList.push_back(minXCurr);
    }

    int kNearestPoints = 100;
    double kMinXCurr = 0;
    double kMinXPrev = 0;

    // kNearestPointsMeanValue(minXPrevList, kNearestPoints, kMinXCurr);
    // kNearestPointsMeanValue(minXCurrList, kNearestPoints, kMinXPrev);   

    kNearestPointsMedianValue(minXCurrList, kNearestPoints, kMinXCurr);
    kNearestPointsMedianValue(minXPrevList, kNearestPoints, kMinXPrev);

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

1. Combo (Detector/Descriptor)

SHITOMASI/BRISK

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 12.9722 | 15.0726 | 2.10042 | 
| 12.264 | 13.2428 | 0.97875 | 
| 13.9161 | 14.3921 | 0.47594 | 
| 14.8865 | 12.8477 | 2.03879 | 
| 7.41552 | 12.8285 | 5.41294 | 
| 12.4213 | 11.3009 | 1.12046 | 
| 34.3404 | 12.1551 | 22.1853 | 
| 9.34376 | 13.6508 | 4.307 | 
| 18.1318 | 10.8869 | 7.2449 | 
| 297.282 | 35.585 | 261.697 | 
| 3.18101 | 12.8 | 9.61899 | 
| -10.8537 | 11.5836 | 22.4373 | 
| 9.22307 | 11.1144 | 1.89128 | 
| 10.9678 | 12.5097 | 1.54197 | 
| 8.18954 | 10.225 | 2.03543 | 
| 3.16065 | 11.414 | 8.25333 | 
| -8.58076 | 10.6233 | 19.2041 | 
| 9.04898 | 10.2699 | 1.22093 | 

Median difference between Lidar and Camera TTC: 3.20371s \
Processing time: 10.7093s

KPI = 100 / ( Processing time + TTC difference ) = 7.1875 

2. Combo (Detector/Descriptor)

FAST/BRIEF

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 12.9722 | 10.7101 | 2.2621 | 
| 12.264 | 11.0063 | 1.25769 | 
| 13.9161 | 14.1559 | 0.239767 | 
| 14.8865 | 14.3886 | 0.497926 | 
| 7.41552 | 20.0387 | 12.6232 | 
| 12.4213 | 13.293 | 0.871662 | 
| 34.3404 | 12.2182 | 22.1223 | 
| 9.34376 | 12.7596 | 3.41582 | 
| 18.1318 | 12.6 | 5.53176 | 
| 297.282 | 13.4681 | 283.813 | 
| 3.18101 | 13.7533 | 10.5723 | 
| -10.8537 | 10.974 | 21.8277 | 
| 9.22307 | 12.3343 | 3.11124 | 
| 10.9678 | 11.2431 | 0.27538 | 
| 8.18954 | 11.8747 | 3.68512 | 
| 3.16065 | 11.8398 | 8.67919 | 
| -8.58076 | 7.92013 | 16.5009 | 
| 9.04898 | 11.554 | 2.50501 | 

Median difference between Lidar and Camera TTC: 3.55047s \
Processing time: 7.61744s

KPI = 100 / ( Processing time + TTC difference ) = 8.95422

3. Combo (Detector/Descriptor)

FAST/ORB

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 12.9722 | 11.0544 | 1.91775 | 
| 12.264 | 10.6351 | 1.62892 | 
| 13.9161 | 13.4008 | 0.515357 | 
| 14.8865 | 12.7877 | 2.09887 | 
| 7.41552 | 18.0923 | 10.6768 | 
| 12.4213 | 12.9892 | 0.567842 | 
| 34.3404 | 12.4642 | 21.8762 | 
| 9.34376 | 11.3522 | 2.00844 | 
| 18.1318 | 12.1119 | 6.01983 | 
| 297.282 | 13.4637 | 283.818 | 
| 3.18101 | 13.1 | 9.91899 | 
| -10.8537 | 10.4239 | 21.2777 | 
| 9.22307 | 12.0462 | 2.8231 | 
| 10.9678 | 11.0103 | 0.0425369 | 
| 8.18954 | 11.4079 | 3.21832 | 
| 3.16065 | 11.6 | 8.43935 | 
| -8.58076 | 7.57979 | 16.1605 | 
| 9.04898 | 10.5903 | 1.54128 | 

Median difference between Lidar and Camera TTC: 3.02071s \
Processing time: 7.21001s

KPI = 100 / ( Processing time + TTC difference ) = 9.77448

4. Combo (Detector/Descriptor)


SHITOMASI/BRIEF

| TTC Lidar (s) | TTC Camera (s) | Difference (s) |
| -----------   | ----------- | ----------- |
| 12.9722 | 14.6756 | 1.70345 | 
| 12.264 | 13.9717 | 1.70765 | 
| 13.9161 | 9.73978 | 4.17635 | 
| 14.8865 | 14.982 | 0.0954879 | 
| 7.41552 | 12.7628 | 5.34728 | 
| 12.4213 | 13.2703 | 0.848987 | 
| 34.3404 | 15.2664 | 19.0741 | 
| 9.34376 | 12.0847 | 2.74097 | 
| 18.1318 | 11.8703 | 6.26145 | 
| 297.282 | 12.6285 | 284.653 | 
| 3.18101 | 11.8507 | 8.66966 | 
| -10.8537 | 11.7642 | 22.618 | 
| 9.22307 | 11.7197 | 2.49661 | 
| 10.9678 | 11.3557 | 0.387933 | 
| 8.18954 | 12.1983 | 4.00872 | 
| 3.16065 | 8.23961 | 5.07896 | 
| -8.58076 | 11.1382 | 19.7189 | 
| 9.04898 | 8.43097 | 0.61801 | 

Median difference between Lidar and Camera TTC: 4.09254s \
Processing time: 7.11801s

KPI = 100 / ( Processing time + TTC difference ) = 8.92017

Conclusion: \
Combination of Detector/Descriptor - FAST/ORB, provides the best performanse, if we calculate KPI as following: \
KPI = 100 / ( Processing time + TTC difference ) = 9.77448

