In this project we'll implement the circle hough transform filter compared to the cv2 built in function.

I've used the following imge as a test subject:\
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Circle%20Hough%20transform/img1.JPG)

Then I've used the canny filter:\
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Circle%20Hough%20transform/img2.JPG)

The following proces is to produced the accumulation matrix, as we can see there's about 4 points with most votes(places that has the most interactions and neighbors that suggest a circle might be at the location)\

Accumulation Matrix:\
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Circle%20Hough%20transform/img10.JPG)

After performing a threshold(this is a noise cleaning method) on the accumulation matrix I got the following:\
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Circle%20Hough%20transform/img9.JPG)

And the final result:\
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Circle%20Hough%20transform/img3.JPG)\
as we can see the filter I've implemented did not catch all circle but gave a fair proof of concept to the algorithm.\

The following result is using the build in cv2 function:\
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Circle%20Hough%20transform/img4.JPG)\

part 2:

Here we'll find the correct radius in order to attribute the correct name to each coin:\

original image:\
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Circle%20Hough%20transform/img5.JPG)\
Result:\
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Circle%20Hough%20transform/img6.JPG)\
