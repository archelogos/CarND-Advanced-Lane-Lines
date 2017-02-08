# Advanced Lane Finding

In the Project 1, some techniques were applied to identify the lane lines of the road in a given driving video.
At that moment, we based our model on the Canny Edge algorithm. In this project we go one step
further applying perspective and gradient transformation as well as other techniques to the video frames, in order to
detect the lane lines in a more approachable way.

The goal of this project is to develop a pipeline to process a video stream from a camera mounted on the front of a car, and output an annotated video which identifies:

* The positions of the lane lines
* The location of the vehicle relative to the center of the lane
* The radius of curvature of the road

To do that, some steps have been defined:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/c1.jpeg "Corners"
[image1]: ./others/undistort_output.png "Undistorted"
[image2]: ./output_images/c2.jpeg "Test Images"
[image3]: ./output_images/c3.jpeg "Thresholded"
[image4]: ./output_images/c4.jpeg "Warped"
[image5]: ./output_images/c5.jpeg "Lines"
[image6]: ./output_images/c6.jpeg "Output"
[image7]: ./output_images/c7.jpeg "Video Poster"
[video1]: ./result.mp4 "Video"

All the code of this project can be found in the Jupyter Notebook named: [lane_lines_detection.ipynb](./lane_lines_detection.ipynb)

In the first part of the project it was defined a step-by-step pipeline. This pipeline was tested in the test images and once all the steps were tried and we were sure about its stability and performance, it was built the pipeline method that computes the video frames.

## Camera Calibration

First of all it was necessary to compute the calibration matrix of the camera and apply the distortion coefficients to the test
images. Making use of the ```cv2.findChessBoardCorners()``` method it was possible to detect the 9x6 corners of the sample images.

Then, these images were modified, adding the corners detected in them.

![alt text][image0]

From 20 test images, in 17 of them were detected the corners, that were used to calibrate the camera from that moment to the rest of the project. These values were obtained thanks to the ```cv2.calibrateCamera()``` method that provides you the Matrix and Distortion coefficients given a set of chessboard images.

With the computed coefficients, and in order to avoid doing this step all the time, these values were saved in a pickle file, doing much easier it use in the future steps of the pipeline.

This part can be found here: [Camera Calibration](./lane_lines_detection.ipynb#Corners-detection)

In the following iamge can be seen the differences between the original and the distorted image, using a calibration one to point
out the distortion effect.

![alt text][image1]

## Pipeline (single images)

### Distortion correction

With the coefficients properly calculated, the next logical step was to start the pipeline process with the distortion correction.

The pipeline can be divided in 2 bigs phases. The first one related to the image processing (distortion correction, binary-image transformation and perspective warping), and the second one oriented to the lines finding.

As it's already mentioned, it was needed to distorted the raw test images. In this case is more complicated to see the
differences between both images, but in the corners the distortion is stonger.

![alt text][image2]

In this case it was necessary to use the ```cv2.undistort()``` method: [Undistort](./lane_lines_detection.ipynb#Undistort)

In the majority of these steps, all transformations and processing was coded with lambda functions in order to do the code simpler.


### 2. Color Transform and gradients

Once the image has been correctly distorted, the next step was to transform it to a binary image. To do that, we set some color thresholds where the lane lines are likely well-detected. It was also set a gradient transformation in the coordinate 'x' to detect vertical lines much easier.

Following the recommendations from the lessons and doing some research at the same time, the final binary was made computing the following transformations:

- Gradient in **'x'** with a 40-100 threshold and kernel of 7 (to get smoothier edges)
- The **S channel (HLS)** with a threshold of 170-255
- The **H channel (HSV)** with a threshold of 170-255

[Applying Gradients to build the binary](./lane_lines_detection.ipynb#Applying-Gradients)

```python
def threshold_image(image):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=7, thresh=(40, 100))
    #grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100))
    #mag_thresh(image, sobel_kernel=5, mag_thresh=(50, 200))
    #dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    s_channel = s_select(image, thresh=(170, 255))
    h_channel = h_select(image, thresh=(170, 255))
    combined = np.zeros_like(gradx)
    ### Sobel X + S Threshold + H Threshold
    combined[(gradx == 1) | ((s_channel == 1) & (h_channel == 1))] = 1
    thresholded_image = combined
    return thresholded_image
```

![alt text][image3]

### 3. Warp the images

With the binary images generated, the next step was to apply a perspective transformation to the image (called 'bird-eye-view')
to extract the relevant information of the image. Because the goal we want to achieve is detect the lane lines, it's necessary to specify
the area where the transformation should be applied.

Taking some time to study the test images and their arrangement, the area selected finally was:

```
src = np.float32([[490, 482],[810, 482], [1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0], [1250, 720],[40, 720]])
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 490, 482      | 0, 0        |
| 810, 482      | 1280, 0      |
| 1250, 720     | 1250, 720      |
| 40, 460      | 40, 720        |

Thanks to the method ```cv2.getPerspectiveTransform()``` it was possible to determine the Tranformation Matrix, and then making use of the function ```cv2.warpPerspective()``` we were able to get the relevant points of the image that define the lane line.

[Apply a perspective transformation](./lane_lines_detection.ipynb#3.-Apply-a-perspective-transform-("birds-eye-view"))

![alt text][image4]

With this last step, all image transformations were applied and checked that they worked as it was expected. At the beginning of the process we had a picture from a standard camara placed in the front of a car. At this point, we had some clear points that
define the lane lines on the road in a particular and useful format.

With this point it was possible to determine the lane lines applying some mathmematical functions.  

### 4. Fitting the polynomials

[Fitting the polynomials](./lane_lines_detection.ipynb#Lane-detection-+-radius-measurement-+-center-finding)

Now that we had the binary images warped, it was the moment to fit a polynomial to each lane line, which was done by:

- Finding the first lane line on the first frame, computing a histogram and identifying the peaks in it.
- Detecting all non zero pixels around the histogram peaks using the numpy function ```numpy.nonzero()```
- Fitting a second order polynomial to each lane using the numpy function ```numpy.polyfit()```

The code can be a bit confused in the first sight, but the key point of the understanding is to keep in mind that the image, ideally,
only has values non zero where the lane lines are placed.

For that reason the calculation can be described as: getting all nonzeros from the image, applying the second order polynomial to both
lines.

![alt text][image5]

### 5. Radius of curvature of the lane and the position of the vehicle with respect to center

Once we had the lines values it was possible to determine the position of the vehicle and the radius of curvature.

For the position of the vehicle, it was assumed the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset we wanted to calculate.

First it was necessary to compute the best fitted x on each line (mean of all x values), and then it was calculated by taking the absolute value of the vehicle position minus the point in the middle along the horizontal axis.

```python
center = abs((1280/2) - ((left.bestx+right.bestx)/2))
center = abs(center*3.7/700)
```

After that, the distance from center was converted from pixels to meters by multiplying the number of pixels by 3.7/700.

Another requirement was to estimate how much the road is curving (the radius of curvature). The radius of curvature was also converted from pixels to meters (it can be described as the curve of the road follows a circle)

```python
ym_per_pix = 30./720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meteres per pixel in x dimension
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*np.max(lefty) + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])
```

First, both lines were fitted to a second order polynomial in meters and it was applied the radius of curvature formula to get
the radius of each line.

To determine mean radius of curvature of that point, it was just calculated the mean of both values.

```python
curverad = int((left.radius_of_curvature + right.radius_of_curvature)/2)
```

### 6. Pipeline Output

At the end of the pipeline, the result was an eye-bird view of each image, previously converted to a binary one. Where lane lines where
detected and interpolated to define a whole line that can be used in the resulting pipeline to draw an area where the car should be driving.

At the same time, the radius of curvature and the relative position of the car to the center were properly calculated, giving
some extra information to the agent in a possible future scenario where these inforation can be the inputs for a making decision
algorithm.

![alt text][image5]


---

## Pipeline (video)

[Video Pipeline](./lane_lines_detection.ipynb#VIDEO-PIPELINE)

After establishing a pipeline to process still images, the final step was to expand the pipeline to process videos frame-by-frame. In this
part of the project we wanted to simulate a real-time processing situation. At the same time we were driving the car, the camera placed in the front of it, is taking pictures of the road. Frame-by-frame we are processing the images and determine not only the area where the car should be driving, but also the position offset from the center, and the radius of curvature as well.

The pipeline process was defined in a single method ```pipeline()``` which receives a standard image from the video and after processes it, it returns
a new and transformed image with a area drawn in it.

These are the steps followed in the process

- Undistort the input
- Convert the image to a binary
- Apply a perspective transformation to get the bird-eye-view
- Find the first lane lines (based on the blind mode)
- Fit a second polynomial to each lane line
- Estimate the center and the radius of curvature
- Modify the image with the area between the two lines and other information
- Keep some information of that iteration to use it in a n-frame window
- Return the output

The make the process easier, it was defined a ```Line``` class with some properties and a ```update()``` method. Two instances of this class
were created (left and right) and the method update is called on each iteration.

```python
# Define a class to receive the characteristics of each line detection
class Line():

    def __init__(self, num_frames=1):
        self.num_frames = num_frames
        self.frame_count = 0
        self.frame_points = []
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update(self, x, y):

        self.frame_count += 1

        # Update points
        self.allx = x
        self.ally = y

        # Append x values
        self.recent_xfitted.extend(self.allx)
        self.frame_points.append(len(self.allx))

        # Don't take into account more than x frames  
        if len(self.frame_points) > self.num_frames:
            points = self.frame_points.pop(0)
            self.recent_xfitted = self.recent_xfitted[points:]

        # Get the mean
        self.bestx = np.mean(self.recent_xfitted)

        # Fit a second order polynomial to each
        self.current_fit= np.polyfit(self.ally, self.allx, 2)

        # Best fit
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.num_frames - 1) + self.current_fit) / self.num_frames
```

The video pipeline first checks whether or not the lane was detected in the previous frame. If it was, then it only checks for lane pixels in close proximity to the polynomial calculated in the previous frame.

At the same time, the pipeline fits the polynomial based on the n (5) previous frames. It also computes the best x of each lane (mean) to estimate the center.

![alt text][image6]

If the pipeline loses the lane lines in a frame it automatically actives the blind mode and scan the entire binary image for nonzero pixels to represent the lanes.


![alt text][image7]

Here's a [link to my video result][video1]

---

## Discussion
