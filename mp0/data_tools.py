"""Processing data tools for mp0.
"""
import re
import numpy as np


def title_cleanup(data):
    """Remove all characters except a-z, A-Z and spaces from the title,
       then convert all characters to lower case.

    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    """
    # pass
    for key,val in data.items():
        title = val[0]
        newtitle = []
        for char in title:
            if char == ' ' or char.isalpha():
                newtitle.append(char)
        data[key][0] = ''.join(newtitle).lower()


def most_frequent_words(data):
    """Find the more frequeny words (including all ties), returned in a list.

    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        max_words(list): List of strings containing the most frequent words.
    """
    max_words = []
    freq_dict = {}
    for key,val in data.items():
        title = val[0].split()
        for word in title:
            if word not in freq_dict:
                freq_dict[word] = 0
            else:
                freq_dict[word] += 1
    sorted_freq = sorted(freq_dict.items(), key=lambda t: t[1],reverse=True)
    max_freq = sorted_freq[0][1]
    while sorted_freq[0][1] == max_freq:
        max_words.append(sorted_freq[0][0])
        sorted_freq.pop(0)
    return max_words


def most_positive_titles(data):
    """Computes the most positive titles.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        titles(list): List of strings containing the most positive titles,
                      include all ties.
    """
    titles = []
    most_pos = -1
    for key,val in data.items():
        score = val[1]
        if score > most_pos:
            most_pos = score
            titles.clear()
            titles.append(val[0])
        elif score == most_pos:
            titles.append(val[0])
    return titles


def most_negative_titles(data):
    """Computes the most negative titles.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
     Returns:
        titles(list): List of strings containing the most negative titles,
                      include all ties.
    """
    titles = []
    most_neg = 100
    for key,val in data.items():
        score = val[1]
        if score < most_neg:
            most_neg = score
            titles.clear()
            titles.append(val[0])
        elif score == most_neg:
            titles.append(val[0])
    return titles


def compute_word_positivity(data):
    """Computes average word positivity.
    Args:
        data(dict): Key: article_id(int),
                    Value: [title(str), positivity_score(float)](list)
    Returns:
        word_dict(dict): Key: word(str), value: word_index(int)
        word_avg(numpy.ndarray): numpy array where element
                                 #word_dict[word] is the
                                 average word positivity for word.
    """
    word_dict = {}
    # word_avg = None
    # word_avg = word_score / word_count

    temp = {}
    for key,val in data.items():
        title = val[0].split()
        for word in title:
            if word not in temp:
                temp[word] = [1.0, val[1]]
            else:
                temp[word][0] += 1.0
                temp[word][1] += val[1]

    word_avg= np.zeros(len(temp))
    idx = 0
    for key,val in temp.items():
        if key not in word_dict:
            word_dict[key] = idx
            word_avg[idx] = float(val[1])/float(val[0])
            idx += 1
        
    return word_dict, word_avg


def most_postivie_words(word_dict, word_avg):
    """Computes the most positive words.
    Args:
        word_dict(dict): output from compute_word_positivity.
        word_avg(numpy.ndarray): output from compute_word_positivity.
    Returns:
        words(list):
    """
    words = []
    most_pos = max(word_avg)
    res = np.where(word_avg == most_pos)[0]
    for i in range(len(res)):
        word = [word for word in word_dict if word_dict[word]==res[i]]
        words.append(word[0])
    return words


def most_negative_words(word_dict, word_avg):
    """Computes the most negative words.
    Args:
        word_dict(dict): output from compute_word_positivity.
        word_avg(numpy.ndarray): output from compute_word_positivity.
    Returns:
        words(list):
    """
    words = []
    most_neg = min(word_avg)
    res = np.where(word_avg == most_neg)[0]
    for i in range(len(res)):
        word = [word for word in word_dict if word_dict[word]==res[i]]
        words.append(word[0])
    return words
