# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/carnotcar.png
[image2]: ./writeup_images/hog_data.png
[image3]: ./writeup_images/window_area.png
[image4]: ./writeup_images/detections.png
[image5]: ./writeup_images/heatmap.png
[image6]: ./writeup_images/labels.png
[image7]: ./writeup_images/success1.jpg
[image8]: ./writeup_images/success2.jpg
[image9]: ./writeup_images/example.jpg
[image10]: ./writeup_images/exampleoutput.jpg

[attempt1]: https://youtu.be/el8ZmU-kPO4
[finalvideo]: https://youtu.be/kFleL6VFWgU

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function LoadImageData() in the file Project5.ipynb.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried a number of parameters and with several rounds of trial and error I came up with the following numbers:

Orientations = 9 <br>
Pixels per cell = 8 <br>
Cells per block = 2 <br>

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before I get to how I used the classifier, let's talk about augmentation of data.
#### Collecting more data and augmenting it
While building this model and without using augmentation, I realized a few things.<br>
1. The quality of the images from most of the datasets wasn't very good.
2. They appeared to mostly be taken from low quality cameras and seemed like they had been upscaled
3. The sizes couldn't deal with larger size of search windows
4. The images didn't have too many samples with shadows, consequently my classifier kept failing around shadowed roads
5. The images were very dull in terms of colour

In order to fix these issues I performed the following
1. Collected some more samples of non cars from the video
2. Added some augmented data by:
    1. Flipping the images
    2. Generating random shadows on non vehicle images. An example is shown in the file "Noise Generation.ipynb"

#### Training step

With the data augmented, I was ready to go and trained my Linear SVM again. This time however it took a lot longer to train considering there were a lot more samples.
I collected the following information from each of the images before scaling it with a StandardScaler()
1. Spatial information
2. Color histogram in 32 bins
3. HOG information
Before training my model, I made sure to shuffle it and then split it into training and test data. I trained the SVM in the function LoadImageData().

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used multiple sliding windows of different sizes and scales depending on where the car was on the road. If there were yellow lines demarcating the right or left sides I would ignore searching on those respective sides as they were not needed.<br>
I looked in front of the car in fixed windows. You can see my search areas illustrated in the following image.

![alt text][image3]

After searching in these areas, I obtained successful searches that looked something like this<br>
#### Original Image
![alt text][image9]
#### Processed Image
![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here's some examples of my pipeline successfully detecting vehicles

![alt text][image7]
![alt text][image8]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][finalvideo]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

After this I also filtered out label rectangles that were too small to avoid some edge cases of really small images

### For the example above this is the corresponding heatmap:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from above:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image10]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There were several problems I faced during this project. Some of them are as follows

1. Initially I had an issue where I was detecting vehicles all over the lower half of the image. I was traffic on the other side as well as traffic in my direction
2. I also initially had many false positives. I used augmentation to avoid this problem

Here is an example [link][attempt1] to my earlier attempt.<br>

As you can see, compared with my final result there are several improvements in terms of false positives. Not only that, it also ran much faster.<br>

Some Improvements I would like to make are as follows
1. Use better training data
2. Try different models ( Bayesian, Tree and Neural Network classifiers )
3. Use state information to stabilize the detection and avoid false positives
4. For performance, I could sample the image at a lower resolution