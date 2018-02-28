import math

import time
from nltk.corpus import stopwords
import sys

TRAINING_DATA_WORD_NUMBER = 61180
TEST_DATA_DOCUMENT_NUMBER = 7505


def calculatePriorProbablity():
    # Step 1 : Calculate prior probabilities

    with open("train.label", "r") as f:
        content = f.readlines()
    label_dict = dict()
    category_probability_dict = dict()
    count = 0
    # Counting total number of labels and add individual counts in a dictionary
    for x in content:
        y = int(x)
        count += 1
        if y not in label_dict:
            label_dict[y] = 1
        else:
            label_dict[y] = label_dict.get(y) + 1
    # print(label_dict)
    # print(count)
    # Calculating each of the label's prior probability by dividing their count by total
    for k in label_dict:
        category_probability_dict[k] = float(label_dict.get(k)) / float(count)

    # print("label dict ", label_dict)
    # print("probability dict ", category_probability_dict)
    return label_dict, category_probability_dict


def calculateLikelihoodOfWords(label_dict, word_frequency_dict, stopWordsList):
    # Calculate Likelihood.Likelihood is the conditional
    # probability of a word occurring in a document given that the document belongs to a particular category.

    word_likelihood_dict = {}
    probabilty_likelihood_dict = {}
    total_word_count_per_category_dict = {}
    # looping over the words
    for i in range(1, TRAINING_DATA_WORD_NUMBER + 1):
        cumulative_doc_count = 0
        # looping over the category
        for j in range(1, len(label_dict) + 1):
            total_frequency_of_word = 0
            x = cumulative_doc_count + label_dict[j]
            # looping over the documents of each category
            for k in range(cumulative_doc_count + 1, x + 1):
                temp_tuple = (k, i)
                if temp_tuple in word_frequency_dict:
                    total_frequency_of_word += word_frequency_dict.get(temp_tuple)
                    q = int(0 if total_word_count_per_category_dict.get(
                        j) is None else total_word_count_per_category_dict.get(j))
                    total_word_count_per_category_dict[j] = q + word_frequency_dict.get(temp_tuple)
            cumulative_doc_count = x
            word_appearance_tuple = (i, j)
            word_likelihood_dict[word_appearance_tuple] = total_frequency_of_word

    # print(word_likelihood_dict)
    # print(total_word_count_per_category_dict)
    # print(cumulative_doc_count)
    # TODO : Finally, calculating the probabilty of this word appearing in this category
    idfCountList = []
    for i in range (TRAINING_DATA_WORD_NUMBER+1):
        idfCountList.append(0)

    for i in range(1, TRAINING_DATA_WORD_NUMBER + 1):
        if i in stopWordsList:
            pass
        else:
            for j in range(1, len(label_dict) + 1):
                temp_tuple = (i, j)
                m = word_likelihood_dict.get(temp_tuple)
                n = total_word_count_per_category_dict.get(j)
                o = TRAINING_DATA_WORD_NUMBER

                if (m>0):
                    idfCountList[i] +=1
                # TODO : Need to test with different alpha
                probabilty_likelihood_dict[(i, j)] = float((.2) + m) / float(
                    n + ((1) * o))
    # print(probabilty_likelihood_dict)
    return probabilty_likelihood_dict, total_word_count_per_category_dict, idfCountList
    print("### Training Completed ###")


def loadWordFrequencies():
    word_frequency_dict = {}
    with open("train.data", "r") as f:
        content = f.readlines()
    for c in content:
        numbers = c.split(" ")
        doc_id = int(numbers[0])
        word_id = int(numbers[1])
        word_freq = int(numbers[2])

        key_tuple = (doc_id, word_id)
        word_frequency_dict[key_tuple] = word_freq

    test_tuple = (1, 99999)
    # print(word_frequency_dict.get(test_tuple))
    return word_frequency_dict


def testAccuracy(label_dict, probabilty_likelihood_dict, total_word_count_per_category_dict, category_probability_dict,
                 stopWordsList, idfCountList):
    ## Put the real label at the beginning
    test_data_real_category_list = []
    test_data_predicted_category_list = []
    test_data_predicted_category_dict = {}
    test_document_dict = {}

    with open("test.label", "r") as f:
        content = f.readlines()
    for c in content:
        x = int(c)
        test_data_real_category_list.append(x)

    with open("test.data", "r") as f:
        content = f.readlines()
    for c in content:
        numbers = c.split(" ")
        doc_id = int(numbers[0])
        word_id = int(numbers[1])
        word_freq = int(numbers[2])

        if word_id in stopWordsList:
            pass
        else:
            ### Listing the documents in test data ###
            if test_document_dict.get(doc_id) is None:
                test_document_dict[doc_id] = 1
            else:
                pass
            # print(test_document_dict)
            ### Calculating the log likelihood for each documents falling in different category ###
            for j in range(1, len(label_dict) + 1):
                ## Calculating likelihood of a document in a category
                temp_cat_tuple = (doc_id, j)
                temp_word_tuple = (word_id, j)

                for k in range(1, (word_freq + 1)):
                    q = int(0 if test_data_predicted_category_dict.get(
                        temp_cat_tuple) is None else test_data_predicted_category_dict.get(temp_cat_tuple))
                    try:
                        # This is actually tf-idf
                        p = probabilty_likelihood_dict.get(temp_word_tuple)
                        if p is None:
                            p = (1) / (total_word_count_per_category_dict[j] + (
                                        (1) * TRAINING_DATA_WORD_NUMBER))
                        # r = float(math.log(p, 2)) + float(math.log((20/idfCountList[word_id]),2))
                        r = float(math.log((p * (20/idfCountList[word_id])),2))
                    except:
                        r = 0.0

                    test_data_predicted_category_dict[temp_cat_tuple] = q + r

    prediction_dict = {}
    doc_count = 0
    for k in test_document_dict:
        max = -9999999
        for j in range(1, len(label_dict) + 1):
            temp_tuple = (k, j)
            test_data_predicted_category_dict[temp_tuple] += math.log(category_probability_dict[j], 2)
            if test_data_predicted_category_dict.get(temp_tuple) > max:
                max = test_data_predicted_category_dict.get(temp_tuple)
                prediction_dict[k] = j
        doc_count += 1
    print("real ", test_data_real_category_list)
    print("tested ", prediction_dict)

    ## Creating Confusion matrix

    # Creates a list containing 20 lists, each of 20 items, all set to 0
    w, h = 21, 21;
    confusion_matrix = [[0 for x in range(w)] for y in range(h)]

    ## Calculating Accuracy
    correct_count = 0
    for k in prediction_dict:
        if test_data_real_category_list[k - 1] == prediction_dict[k]:
            correct_count += 1
            confusion_matrix[prediction_dict[k] - 1][prediction_dict[k] - 1] += 1
        else:
            confusion_matrix[test_data_real_category_list[k - 1] - 1][prediction_dict[k] - 1] += 1

    print(correct_count)
    print("Accuracy is : ", (correct_count / TEST_DATA_DOCUMENT_NUMBER) * 100)
    print("Confusion Matrix : ", confusion_matrix)
    # print(test_data_predicted_category_dict)

    ## ****** Printing Confusion Matrix ***********


def detectStopWords():
    stopWordsList = []
    stop_words = set(stopwords.words('english'))
    with open("vocabulary.txt", "r") as f:
        content = f.readlines()
    count = 1
    for c in content:
        c = c.strip()
        if c in stop_words:
            stopWordsList.append(count)
            # print(c, "index : ", count)
            # print()
        count += 1
    return stopWordsList


def start():
    start_time = time.time()
    print("hola")
    stopWordsList = detectStopWords()
    label_dict, category_probability_dict = calculatePriorProbablity()
    word_frequency_dict = loadWordFrequencies()
    # calculateWordCountsForEachCategory(label_dict)
    probabilty_likelihood_dict, total_word_count_per_category_dict, idfCountList = calculateLikelihoodOfWords(label_dict,
                                                                                                word_frequency_dict,
                                                                                                stopWordsList)
    print("Training Completion Time --- %s seconds ---" % (time.time() - start_time))
    testAccuracy(label_dict, probabilty_likelihood_dict, total_word_count_per_category_dict, category_probability_dict,
                 stopWordsList, idfCountList)
    print("Total Execution Time --- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    start()
