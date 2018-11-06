
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

ONEVONE = "one-vs-one"

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    Cvalue=1000)

def confusion(y_test, Y_vote, num_classes):
    """     
            Calculates confusion matrix

            This method calculates the confusion matrix based on a vector of
            ground-truth labels (y-test) and a matrix containing the predicted
            class (c) of each sample (x). Will return a matrix of predicted classes 
            against the actual class of each sample, to show how each sample was 
            classified by the SVM.

            :param y_test: vector of actual labels for each sample.
            :param Y_vote: 2D voting matrix (rows= samples, cols= class votes)
            :returns: confusion matrix
    """

    y_hat = np.argmax(Y_vote, axis=1)
    conf = np.zeros((num_classes, num_classes)).astype(np.int32)

    for c_true in xrange(num_classes):
            # looking at all samples of a given class, c_true
            # how many were classified as c_true? how many as others?
        for c_pred in xrange(num_classes):
            y_this = np.where((y_test == c_true) * (y_hat == c_pred))
            conf[c_pred, c_true] = np.count_nonzero(y_this)

    return conf

def precision(mode, y_test, Y_vote, num_classes):
    """     
            Calculates precision

            This method calculates precision defined as tp/(tp + fp)
            where, tp is the number of true positives 
            and fp is the number pf false positives

            :param y_test: vector of actual classes for each sample
            :param Y_vote: 2D voting matrix (rows=samples, cols=class votes)
            :returns: precision e[0,1]
    """
        # predicted classes
    y_hat = np.argmax(Y_vote, axis=1)

    if mode == "one-vs-one":
            # need confusion matrix
        conf = confusion(y_test, Y_vote, num_classes)

            # consider each class separately
        prec = np.zeros(num_classes)
        for c in xrange(num_classes):
                # true positives: label is c, classifier predicted c
            tp = conf[c, c]

                # false positives: label is c, classifier predicted not c
            fp = np.sum(conf[:, c]) - conf[c, c]

            if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)
    elif mode == "one-vs-all":
            # consider each class separately
        prec = np.zeros(num_classes)
        for c in xrange(num_classes):
                # true positives: label is c, classifier predicted c
            tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false positives: label is c, classifier predicted not c
            fp = np.count_nonzero((y_test == c) * (y_hat != c))

            if tp + fp != 0:
                prec[c] = tp * 1. / (tp + fp)
    return prec

def recall(mode, y_test, Y_vote, num_classes):
    """     
            Calculates Recall
            
            This method calculates recall defined as tp/(tp + fn),
            where tp is the number of true positives 
            and fn is the number of false negatives 

            :param y_test: vector of actual cllasses for each sample
            :param Y_vote: 2D voting matrix (rows=samples, cols=class votes)
            :returns: recall e[0,1]
    """
    
    # predicted classes
    y_hat = np.argmax(Y_vote, axis=1)

    if mode == "one-vs-one":
            # need confusion matrix
        conf = confusion(y_test, Y_vote, num_classes)

            # consider each class separately
        recall = np.zeros(num_classes)
        for c in xrange(num_classes):
                # true positives: label is c, classifier predicted c
            tp = conf[c, c]

                # false negatives: label is not c, classifier predicted c
            fn = np.sum(conf[c, :]) - conf[c, c]
            if tp + fn != 0:
                recall[c] = tp * 1. / (tp + fn)

    elif mode == "one-vs-all":
            # consider each class separately
        recall = np.zeros(num_classes)
        for c in xrange(num_classes):
                # true positives: label is c, classifier predicted c
            tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false negatives: label is not c, classifier predicted c
            fn = np.count_nonzero((y_test != c) * (y_hat == c))

            if tp + fn != 0:
                recall[c] = tp * 1. / (tp + fn)

    return recall

def accuracy(labels_test, Y_vote):
        """ 
            Calculates accuracy

            This method calculates the accuracy defined as (tp+tn)/(tp+tn+fp+fn)
            where, tp is the number of true positives,
            tn is the number of true negatives,
            fp is the number of false positives,
            fn is the number of false negatives

            :param labels_test: vector of actual cllasses for each sample
            :param Y_vote: 2D voting matrix (rows=samples, cols=class votes)
            :returns: accuracy e[0,1]
        """
        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        # all cases where predicted class was correct
        mask = y_hat == labels_test

        antimask = y_hat != labels_test

        for i in range(0, len(antimask)):
            if not (y_hat[i] == labels_test[i]):
                print("index:" + str(i) + " ,predicted: " + str(y_hat[i]) + " ,actual: " + str(labels_test[i]))

        return np.float32(np.count_nonzero(mask)) / len(labels_test)

def extract_feature_for_image(x, feature):
    """ 
            Performs feature extraction gven an image.

            :param x Image to extract from.
            :param feature Feature to extact from x
            :returns the extracted feature vector
    """

    # Transform image colours for feature extraction.
    if feature == 'gray' or feature == 'surf':
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    elif feature == 'hsv':
        x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)

    # Make the image smaller (due to memory constraints) 
    img_size = (128, 128)
    x = cv2.resize(x, img_size)


    # ---- Extract feature ----

    # ---- SURF ----
    if feature == 'surf':

        surf = cv2.SURF(1600)
        surf.upright = True
        surf.extended = True
        feature_count = 36

        # create dense grid of keypoints
        dense = cv2.FeatureDetector_create("Dense")
        dense.setInt("initXyStep", 5)
        dense.setInt("initFeatureScale", 5)
        keypoints = dense.detect(np.zeros(img_size).astype(np.uint8))

        # compute keypoints and descriptors
        keypoints_and_descriptors = [surf.compute(x, keypoints)]

        # the second element is descriptor: choose first feature_count
        # elements
        x = [d[1][:feature_count, :] for d in keypoints_and_descriptors]
    elif feature == 'hog':

        # histogram of gradients
        block_size = (img_size[0] / 2, img_size[1] / 2)
        block_stride = (img_size[0] / 4, img_size[1] / 4)
        cell_size = block_stride
        num_bins = 9
        hog = cv2.HOGDescriptor(img_size, block_size, block_stride,
                                cell_size, num_bins)
        x = hog.compute(x)

    elif feature == 'multihist':

        b = cv2.calcHist(x, [0], None, [256], [0, 256])
        g = cv2.calcHist(x, [1], None, [256], [0, 256])
        r = cv2.calcHist(x, [2], None, [256], [0, 256])
        multi = []
        for i in range(0, 256):
            multi.append(b[i])
            multi.append(g[i])
            multi.append(r[i])
        x = np.array(multi)
        
    elif feature is not None:
        # normalize all intensities to be between 0 and 1
        x = np.array(x).astype(np.float32) / 255

        # subtract mean
        x = [xx - np.mean(x) for xx in x]

    x = [xx.flatten() for xx in x]

    #print(x)
    return x

def extract_multiple_feature(X, features):
    """ 
        Extract multiple features from a list of images.
        (build the data set)

        :param X A vector of images
        :param features A list of features to extract from each image.abs
        :returns The data set.
            
    """
    
    AllX = []
    count = 0
    for x in X:
        count = count + 1
        image_features = []
        for f in features:
            feature_matrix = extract_feature_for_image(x,f)
            if len(feature_matrix[0] == 1):
                for i in feature_matrix:
                    image_features.append(i[0])
            if  len(feature_matrix[0] > 1):
                for i in feature_matrix:
                    for j in i:
                        image_features.append(j)
        AllX.append(image_features)

    return AllX


#classes2 = np.array(['bridge', 'coast', 'mountain', 'rainforest'])
classes2 = np.array(['bridge', 'coast', 'mountain', 'rainforest'])
graphFeaturesAll = np.array([1, 2, 3, 4, 5, 6])
#featuresAll = np.array([['gray'], ['rgb'], ['hsv'], ['surf'], ['hog'], ['multihist']])
featuresAll = np.array([['gray'], ['rgb'], ['hsv'], ['surf'], ['hog'], ['multihist']])
features = np.array(['gray'])

OneVOneA = []
OneVOneP = []
OneVOneR = []

OneVAllA = []
OneVAllP = []
OneVAllR = []

for features in featuresAll:
    print(features)

# read all training samples and corresponding class labels
    X = []  # data
    labels = []  # corresponding labels
    image_paths = []
    for c in xrange(len(classes2)):

        pathstring = "C:\Python27\cv\Exercise 2\Training set\Training set\\"
        jpg_string = '\*.jpg'
        url = pathstring + classes2[c] + jpg_string
        print(url)

        images = glob.glob(pathstring + classes2[c] + jpg_string)
        image_paths.extend(images)
        print(len(images))

        # loop over all images in current annotations file
        for image_url in images:
            # first column is filename
            im = cv2.imread(image_url)

            #X.append(extract_multiple_feature(im, features))
            X.append(im)
            labels.append(c)

    # perform feature extraction
    X = extract_multiple_feature(X, features)
    #print(len(X))
    #print(len(labels))

    # read all training samples and corresponding class labels
    X_test = []  # data
    labels_test = []  # corresponding labels_test
    for c in xrange(len(classes2)):

        pathstring = "C:\Python27\cv\Exercise 2\Testing set\Testing set\\"
        jpg_string = '\*.jpg'
        url = pathstring + classes2[c] + jpg_string
        print(url)

        images = glob.glob(pathstring + classes2[c] + jpg_string)
        print(len(images))

        # loop over all images in current annotations file
        for image_url in images:
        # first column is filename
            im = cv2.imread(image_url)

            #X_test.append(extract_multiple_feature(im, features))
            X_test.append(im)
            labels_test.append(c)

    # perform feature extraction
    #X_test = _extract_feature(X_test, feature)
    X_test = extract_multiple_feature(X_test, features)

    X = np.squeeze(np.array(X)).astype(np.float32)
    X_test = np.squeeze(np.array(X_test)).astype(np.float32)
    labels = np.array(labels)
    labels_test = np.array(labels_test)

    classifiers = []
    for _ in xrange(len(classes2)):
        classifiers.append(cv2.SVM())

    for c in xrange(len(classes2)):
                    # train c-th SVM on class c vs. all other classes
                    # set class label to 1 where class==c, else 0
        y_train_bin = np.where(labels == c, 1, 0).flatten()

                    # train SVM
        classifiers[c].train(X, y_train_bin, params=svm_params)

    Y_vote = np.zeros((len(labels_test), len(classes2)))
    for c in xrange(len(classes2)):
                    # set class label to 1 where class==c, else 0
                    # predict class labels
                    # y_test_bin = np.where(y_test==c,1,0).reshape(-1,1)

                    # predict labels
        y_hat = classifiers[c].predict_all(X_test)

                    # we vote for c where y_hat is 1
        if np.any(y_hat):
            Y_vote[np.where(y_hat == 1)[0], c] += 1

                # with this voting scheme it's possible to end up with samples
                # that have no label at all, in this case, pick a class at
                # random
    no_label = np.where(np.sum(Y_vote, axis=1) == 0)[0]
    Y_vote[no_label, np.random.randint(len(classes2), size=len(no_label))] = 1

    print("1 vs All")

    accuracy1va = accuracy(labels_test, Y_vote)
    print("Accuracy: ", accuracy1va)
    OneVAllA.append(accuracy1va)

    precision1va = precision(ONEVONE, labels_test, Y_vote, len(classes2))
    print("Precision: ", precision1va)
    print("Mean precision: ", np.mean(precision1va))
    OneVAllP.append(np.mean(precision1va))

    recall1va = recall(ONEVONE, labels_test, Y_vote, len(classes2))
    print("Recall: ", recall1va)
    print("Mean Recall: ", np.mean(recall1va))
    OneVAllR.append(np.mean(recall1va))

    confusion1va = confusion(labels_test, Y_vote, len(classes2))
    print(confusion1va)


    print('1 v 1')

    classifiers = []
    num_classes = len(classes2)
    # k classes: need k*(k-1)/2 classifiers
    for _ in xrange(num_classes*(num_classes - 1) / 2):
        classifiers.append(cv2.SVM())

    svm_id = 0
    for c1 in xrange(len(classes2)):
        for c2 in xrange(c1 + 1, len(classes2)):
        # indices where class labels are either `c1` or `c2`
            data_id = np.where((labels == c1) + (labels == c2))[0]

            # set class label to 1 where class is `c1`, else 0
            y_train_bin = np.where(labels[data_id] == c1, 1, 0).flatten()

            classifiers[svm_id].train(X[data_id, :], y_train_bin, params=svm_params)
            svm_id += 1

    Y_vote = np.zeros((len(labels_test), len(classes2)))

    svm_id = 0
    for c1 in xrange(len(classes2)):
        for c2 in xrange(c1 + 1, len(classes2)):
            data_id = np.where((labels_test == c1) + (labels_test == c2))[0]
            X_test_id = X_test[data_id, :]
            labels_test_id = labels_test[data_id]

                        # set class label to 1 where class==c1, else 0
                        # labels_test_bin = np.where(labels_test_id==c1,1,0).reshape(-1,1)

                        # predict labels
            y_hat = classifiers[svm_id].predict_all(X_test_id)

            for i in xrange(len(y_hat)):
                if y_hat[i] == 1:
                    Y_vote[data_id[i], c1] += 1
                elif y_hat[i] == 0:
                    Y_vote[data_id[i], c2] += 1
                else:
                    print "y_hat[", i, "] = ", y_hat[i]

                        # we vote for c1 where y_hat is 1, and for c2 where y_hat
                        # is 0 np.where serves as the inner index into the data_id
                        # array, which in turn serves as index into the results
                        # array
                        # Y_vote[data_id[np.where(y_hat == 1)[0]], c1] += 1
                        # Y_vote[data_id[np.where(y_hat == 0)[0]], c2] += 1
            svm_id += 1

    accuracy1v1 = accuracy(labels_test, Y_vote)
    print("Accuracy: ", accuracy1v1)
    OneVOneA.append(accuracy1v1)

    precision1v1 = precision(ONEVONE, labels_test, Y_vote, len(classes2))
    print("Precision: ", precision1v1)
    print("Mean Precision: ", np.mean(precision1v1))
    OneVOneP.append(np.mean(precision1v1))

    recall1v1 = recall(ONEVONE, labels_test, Y_vote, len(classes2))
    print("Recall: ", recall1v1)
    print("Mean Recall: ", np.mean(recall1v1))
    OneVOneR.append(np.mean(recall1v1))

    confusion1v1 = confusion(labels_test, Y_vote, len(classes2))
    print("Confusion Matrix: ", confusion1v1)

w = 0.2
ax = plt.subplot(111)
ax.set_title("1vAll")
ax.set_ylim([0,1])
my_xticks = featuresAll
plt.xticks(graphFeaturesAll, my_xticks)
ax.bar(graphFeaturesAll-w, OneVAllA,width=0.2,color='b',align='center')
ax.bar(graphFeaturesAll, OneVAllP,width=0.2,color='g',align='center')
ax.bar(graphFeaturesAll+w, OneVAllR,width=0.2,color='r',align='center')
plt.show()

w = 0.2
ax = plt.subplot(111)
ax.set_title("1v1")
ax.set_ylim([0,1])
my_xticks = featuresAll
plt.xticks(graphFeaturesAll, my_xticks)
ax.bar(graphFeaturesAll-w, OneVOneA,width=0.2,color='b',align='center')
ax.bar(graphFeaturesAll, OneVOneP,width=0.2,color='g',align='center')
ax.bar(graphFeaturesAll+w, OneVOneR,width=0.2,color='r',align='center')
plt.show()

