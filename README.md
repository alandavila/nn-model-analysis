# Object Recognition Performance Dashboard

App to determine the performance of a pre-trained CNN model for user selected images both visually and quantitatively.
Currently we use the [MobileNet v1_0.25_224](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)
model for object classification

Our app will display the ten most frequent classes found in the input images as orange nodes of size proportional to the class frequency.
Each node will have blue children nodes with links to the image and size proportional to the probability of belonging to the given class as assigned by the model.

A frequency plot for each class and violin distribution of the class' members probability are displayed.

Getting the performance of a pre-trained model is the first step of implementing [transfer learning](https://www.tensorflow.org/tutorials/image_retraining)
where the last layer of the model is truncated and the network is connected to a new layer with weights
that are trained using the new set of data


### Web site: https://orpd.herokuapp.com/

# Usage

We use the MobileNet CNN pre-trained model to a label a user selected set of images. We use the model to predict each image and select the class label
with the highest probability for plotting and analysis.  


1. Load a set of images by clicking on "Choose Files" and click on "Make Plots"

2. Explore the classes retrieved by the CNN model by hovering over the orange parent nodes.

3. Explore the images clustered in a particular class by clicking on the blue child nodes to see the image classified. Hover over the child node to see the probability assigned for that image-class pair

4. The Class Frequency Histogram displays the number of images assigned to each of the top ten frequency classes

5. The Class Probability Stats displays the distribution of probability for the images assigned to each of the top ten frequency classes



![dashboard](static/resources/dashboard.PNG)


# App requirements.

## Back End

### Python modules (requirements.txt)

  * Flask==1.0.2
  * gunicorn==19.8.1

### JavaScript libraries
  * [tensorflow.js](https://js.tensorflow.org/)

## Front End

### Bootstrap v4.0.0
  * https://getbootstrap.com/
    * Copyright 2011-2017 Twitter, Inc.
    * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
    * (Glyphiconas as available from bootstrap. http://glyphicons.com/)

### D3
  * https://d3js.org/

### jQuery
   * https://jquery.com/

### Plotly
  * https://plot.ly/
