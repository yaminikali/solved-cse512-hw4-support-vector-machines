Download Link: https://assignmentchef.com/product/solved-cse512-hw4-support-vector-machines
<br>
Linear case

Consider training a linear SVM on linearly separable dataset consisting of <em>n </em>points. Let <em>m </em>be the number of support vectors obtained by training on the entire set. Show that the LOOCV error is bounded above by.

<em>Hint: Consider two cases: (1) removing a support vector data point and (2) removing a non-support vector data point.</em>

1.2     General case

Now consider the same problem as above. But instead of using a linear SVM, we will use a general kernel. Assuming that the data is linearly separable in the high dimensional feature space corresponding to the kernel, does the bound in previous section still hold? Explain why or why not.

Question 2 – Implementation of SVMs

In this problem, you will implement SVMs using quadratic programming. Quadratic programs refer to optimization problems in which the objective function is quadratic and the constraints are linear. Quadratic programs are well studied in optimization literature, and there are efficient solvers. Many Machine Learning algorithms are reduced to solving quadratic programs. In this question, you will use the quadratic program solver of Matlab to optimize the dual objective of a kernel SVM.

The dual objective of kernel SVM can be written as:

<em>n                                n         n</em>

maximize                                              (1)

<em>i</em>=1 <em>j</em>=1

s.t.(2)

(3)

<ol>

 <li>(10 points) Write the SVM dual objective as a quadratic program. Look at the quadprog function of Matlab, and write down what <strong>H</strong><em>,</em><strong>f</strong><em>,</em><strong>A</strong><em>,</em><strong>b</strong><em>,</em>Aeq<em>,</em>beq, lb, ub are.</li>

 <li>Use quadratic programming to optimize the dual SVM objective. In Matlab, you can use the function quadprog.</li>

 <li>Write a program to compute <strong>w </strong>and <em>b </em>of the primal from <em>α </em>of the dual. You only need to do this for linear kernel.</li>

 <li>(10 points) Set <em>C </em>= 0<em>.</em>1, train an SVM with linear kernel using trD, trLb in q21data.mat (in Matlab, load the data using load q21data.mat). Test the obtained SVM on valD, valLb, and report the accuracy, the objective value of SVM, the number of support vectors, and the confusion matrix.</li>

 <li>(5 points) Repeat the above question with <em>C </em>= 10.</li>

 <li><em>(10 points + 10 Bonus)</em></li>

</ol>

For this question, you will use multiple binary kernel SVMs to do Crowd Image Classification(Task is same as the one in HW3). This data has 4 classes. Train your multiclass SVM classifier for these 4 classes and compete in an in-class Kaggle competition: <a href="https://www.kaggle.com/c/cse512hw4">https://www.kaggle.com/ </a><a href="https://www.kaggle.com/c/cse512hw4">c/cse512hw4</a><a href="https://www.kaggle.com/c/cse512hw4">.</a>

Training, Test and Validation data can be downloaded from the Kaggle page <a href="https://www.kaggle.com/c/cse512hw4">https://www.kaggle</a>. <a href="https://www.kaggle.com/c/cse512hw4">com/c/cse512hw4</a><a href="https://www.kaggle.com/c/cse512hw4">.</a> Use Training date for training your SVM classifier. You can use the Validation data for hyperparameter tuning. Submit the predictions on the test data to the Kaggle page.

We have already computed feature vectors for you. Each feature vector has 512 features. For reference, we also provide the jpeg images from which the feature vectors were extracted, but you are not required to use them. For multi-class classification, you can use one-versus-one or one-versus-rest approaches. You’re not allowed to use any other classifiers for this submission. Report the best accuracy and the approach, the kernel, the parameters you used to achieve that.

We will maintain a leader board, and the top three entries at the end of the competition (assignment due date) will receive 10 bonus points. Any submission that rises to top three after the assignment deadline is not eligible for bonus points. The ranking will be based on the Categorization accuracy (percentage of correct label).

To prevent exploiting test data, you are allowed to make a maximum of 3 submissions per 24 hours. Your submission will be evaluated immediately and the leader board will be updated.

For this question, you don’t need to have the highest accuracy to earn full points. However, you might loose all or some points if your performance is much lower than a certain threshold. The threshold will be determined by us, based on what we believe to be the minimum value that a correct implementation should achieves.

Question 3 – SVM for object detection

In this question, you will train an SVM and use it for detecting human upper bodies in your favorite TV series The Big Bang Theory. You must use your SVM implementation from Question 2.

To detect human upper bodies in images, we need a classifier that can distinguish between upper-body image patches from non-upper-body patches. To train such a classifier, we can use SVMs. The training data is typically a set of images with bounding boxes of the upper bodies. Positive training examples are image patches extracted at the annotated locations. A negative training example can be any image patch that does not significantly overlap with the annotated upper bodies. Thus there are potentially many more negative training examples than positive training examples. Due to memory limitation, it will not be possible to use all negative training examples at the same time. In this question, you will implement hard-negative mining to find hardest negative examples and iteratively train an SVM.

3.1     Data

Training images are provided in the subdirectory trainIms. The annotated locations of the upper bodies are given in trainAnno.mat. This file contains a cell structure ubAnno; ubAnno{i} is the annotated locations of the upper bodies in the <em>i<sup>th </sup></em>image. ubAnno{i} is 4×<em>k </em>matrix, where each column corresponds to an upper body. The rows encode the left, top, right, bottom coordinates of the upper bodies (the origin of the image coordinate is at the top left corner).

Images for validation and test are given in valIms, testIms respectively. The annotation file for test images is not released.

3.2      External library

Raw image intensity values are not robust features for classification. In this question, we will use Histogram of Oriented Gradient (HOG) as image features. HOG uses the gradient information instead of intensities, and this is more robust to changes in color and illumination conditions. See [1] for more information about HOG, but it is not required for this assignment.

To use HOG, you will need to install an VLFEAT: <a href="http://www.vlfeat.org">http://www.vlfeat.org</a><a href="http://www.vlfeat.org">.</a> This is an excellent cross-platform library for computer vision and machine learning. However, in this homework, you are only allowed to use the HOG calculation and visualization function vlhog. In fact, you should not call vlhog directly. Use the supplied helper functions instead; they will call vlhog.

3.3     Helper functions

To help you, a number of utility functions and classes are provided. The most important functions are in HW4Utils.m.

<ol>

 <li>Run HW4Utils.demo1 to see how to read and display upper body annotation</li>

 <li>Run HW4Utils.demo2 to display image patches and HOG feature images. Compare HOG features for positive and negative examples, can you see why HOG would be useful for detect upper bodies?</li>

 <li>Use HW4Utils.getPosAndRandomNeg() to get initial training and validation data. Positive instances are HOG features extracted at the locations of upper bodies. Negative instances are HOG features at random locations of the images. The data used in Question 3 is actually generated using this function.</li>

 <li>Use HW4Utils.detect to run the sliding window detector. This returns a list of locations and SVM scores. This function can be used for detecting upper bodies in an image. It can also be used to find hardest negative examples in an image.</li>

 <li>Use HW4Utils.cmpFeat to compute HOG feature vector for an image patch.</li>

 <li>Use HW4Utils.genRsltFile to generate result file.</li>

 <li>Use HW4Utils.cmpAP to compute the Average Precision for the result file.</li>

 <li>Use HW4Utils.rectOverlap to compute the overlap between two rectangular regions. The overlap is defined as the area of the intersection over the area of the union. A returned detection region is considered correct (true positive) if there is an annotated upper body such that the overlap between the two boxes is more than 0.5.</li>

 <li>Some useful Matlab functions to work with images are: imread, imwrite, imshow, rgb2gray, imresize.</li>

</ol>

3.4     What to implement?

<ol>

 <li>(15 points) Use the training data in HW4Utils.getPosAndRandomNeg() to train an SVM classifier. Use this classifier to generate a result file (use HW4Utils.genRsltFile) for validation data. Use HW4Utils.cmpAP to compute the AP and plot the precision recall curve. Submit your AP and precision recall curve (on validation data).</li>

</ol>

Algorithm 1 Hard negative mining algorithm

<em>PosD </em>← all annotated upper bodies

<em>NegD </em>← random image patches (<strong>w</strong><em>,b</em>) ← trainSVM(<em>PosD,NegD</em>) for <em>iter </em>= 1<em>,</em>2<em>,</em>··· do

<ul>

 <li>← All non support vectors in <em>NegD</em>.</li>

 <li>← Hardest negative examples <em>. </em>Run UB detection and find negative patches that</li>

</ul>

<em>. </em>violate the SVM margin constraint the most

<em>NegD </em>← (<em>NegD </em> <strong>A</strong>) ∪ <strong>B</strong>.

(<strong>w</strong><em>,b</em>) ← trainSVM(<em>PosD,NegD</em>) end for

<ol start="2">

 <li>Implement hard negative mining algorithm given in Algorithm 2. Positive training data and random negative training data can be generated using HW4Utils.getPosAndRandomNeg(). At each iteration, you should remove negative examples that do not correspond to support vectors from the negative set. Use the function HW4Utils.detect on train images to identify hardest negative examples and include them in the negative training set. Use HW4Utils.cmpFeat to compute HOG feature vectors.</li>

</ol>

Hints: (1) a negative example should not have significant overlap with any annotated upper body. You can experiment with different threshold but 0.3 is a good starting point. (2) make sure you normalize the feature vectors for new negative examples. (3) you should compute the objective value at each iteration; the objective values should not decrease.

<ol start="3">

 <li>(15 points) Run the negative mining for 10 iterations. Assume your computer is not so powerful and so you cannot add more than 1000 new negative training examples at each iteration. Record the objective values (on train data) and the APs (on validation data) through the iterations. Plot the objective values. Plot the APs.</li>

 <li><em>(15 points) </em>For this question, you will need to generate a result file for test data using the function HW4Utils.genRsltFile. You will need to submit this file on <a href="https://docs.google.com/forms/d/e/1FAIpQLSeqg5Gb3bS1omsEn6zdpHYn7AeD68c7IheA8qKZ0VcuuSabLQ/viewform?usp=sf_link">https://docs.google. </a><a href="https://docs.google.com/forms/d/e/1FAIpQLSeqg5Gb3bS1omsEn6zdpHYn7AeD68c7IheA8qKZ0VcuuSabLQ/viewform?usp=sf_link">com/forms/d/e/1FAIpQLSeqg5Gb3bS1omsEn6zdpHYn7AeD68c7IheA8qKZ0VcuuSab</a>LQ/ <a href="https://docs.google.com/forms/d/e/1FAIpQLSeqg5Gb3bS1omsEn6zdpHYn7AeD68c7IheA8qKZ0VcuuSabLQ/viewform?usp=sf_link">viewform?usp=sf_link</a> to receive the AP on test data. Report the AP in your answer file. Important Note: You MUST use your Stony Brook ID as the name of your submission file, i.e., yourSBUID.mat (e.g., 012345679.mat). Your submission will not be evaluated if you don’t use your SBU ID. For this question, you don’t need to have the highest AP to earn full marks. However, you might loose all or some points if your performance is much lower than a certain threshold. The threshold will be determined by us, based on what we believe to be the minimum value that a correct implementation should achieve.</li>

 <li><em>(10 bonus points) </em>Your submitted result file for test data will be automatically entered in a competition for fame. We will maintain a leader board (<a href="https://docs.google.com/spreadsheets/d/1Deg3NNjZrwwZOVfZAs1zZTApcdKO_P8Xg6H7ZJcCu9s/edit?usp=sharing">https://docs.google.com/spreadsheets/d/ </a><a href="https://docs.google.com/spreadsheets/d/1Deg3NNjZrwwZOVfZAs1zZTApcdKO_P8Xg6H7ZJcCu9s/edit?usp=sharing">1Deg3NNjZrwwZOVfZAs1zZTApcdKO_P8Xg6H7ZJcCu9s/edit?usp=sharing</a><a href="https://docs.google.com/spreadsheets/d/1Deg3NNjZrwwZOVfZAs1zZTApcdKO_P8Xg6H7ZJcCu9s/edit?usp=sharing">)</a> and the top three entries at the end of the competition (due date) will receive 10 bonus points. The ranking is based on AP.</li>

</ol>

You can submit the result as frequent as you want. However, the evaluation server will only evaluate all submissions two times a day, at 12:00pm and 10:00pm. The system only keeps the recent submission file, and your new submission will override the previous ones. Therefore, you have two chances a day to evaluate your method. The leader board will be updated in 30 minutes after every evaluation cycle.

You are allowed to use any feature types for this part of the homework. For example, you can use different parameter settings for HOG feature computation. You can even combine multiple HOG features. You can also append HOG features with geometric features (e.g., think about the locations of the upper body). You are allowed to perform different types of feature normalization (e.g, <em>L</em><sub>1</sub>, <em>L</em><sub>2</sub>). You can use both training and validation data to train your classifier. You are allowed to use SVMs, Ridge Regression, Lasso Regression, or any technique that we have covered. You can run hard negative mining algorithm for as many iterations as you want, and the number of negative examples added at each iteration is not limited by 1000. You are not allowed to use Deep Learning features