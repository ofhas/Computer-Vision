# Computer-Vision
# words detector

in this project we'll implement a simple words detector using masks and filters.
first I've used threshold filter over the original image:
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Words%20detection/word1.JPG)
Then I've dilated it:
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Words%20detection/word2.JPG)

Then I've used the find contours and draw contours functions and then drew rectangulares around each countour that was created:

result:
![alt text](https://github.com/ofhas/Computer-Vision/blob/master/Words%20detection/word3.JPG)
