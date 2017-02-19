'''

Assignment #2

    - Reading and tokenizing English-language text in Python.

'''

import re

def tokenize(string, lowercase=False):
    """Extract words from a string containing English words.

    Handling of hyphenation, contractions, and numbers is left to your
    discretion.

    Args:
        string (str): A string containing English.
        lowercase (bool, optional): Convert words to lowercase.

    Returns:
        list: A list of words.

    """

    #result = [i.lower() if lowercase else i for i in re.split(r"(\d+|-|\w'|\w+)", string) if i];


    result = [];
    tokens = re.split(r"(\d+|-|\w'|\w+)", string);
    for index, i in enumerate(tokens):
        if i:
            if i == "'":
                result.append("'" + tokens[index + 1]);
                continue;
            elif tokens[index - 1] == "'":
                continue;
            elif lowercase:
                result.append(i.lower());
            else:
                result.append(i);

    return result


def argmax(sequence):
    """Return the index of the highest value in a list.

    This is a warmup exercise.

    Args:
        sequence (list): A list of numeric values.

    Returns:
        int: The index of the highest value in `sequence`.

    """

    import operator
    index, value = max(enumerate(sequence), key=operator.itemgetter(1))

    return index


def shared_words(text1, text2):
    """Identify shared words in two texts written in English.

    Your function must make use of the `tokenize` function above.

    Args:
        text1 (str): A string containing English.
        text2 (str): A string containing English.

    Returns:
        set: A set with words appearing in both `text1` and `text2`.

    """

    list1 = tokenize(text1.strip(' '))
    list2 = tokenize(text2.strip(' '))

    list3 = set(list1) & set(list2)
    list3.remove(' ');

    return list3

def shared_words_from_filenames(filename1, filename2):
    """Identify shared words in two texts stored on disk.

    Your function must make use of the `tokenize` function above.

    For each filename you will need to `open` file and read the file's
    contents.

    There are two sample text files in the `data/` directory which you can use
    to practice on.

    Args:
        filename1 (str): A string containing English.
        filename2 (str): A string containing English.

    Returns:
        set: A set with words appearing in both texts.

    """

    """
    filename1 = tokenize(text1)
    filename2 = tokenize(text2)

    list3 = set(filename1) & set(filename2)

    return list3

    """
    with open(filename1, encoding="utf8") as f1, open(filename2, encoding="utf8") as f2:

        wordsFile1 = [];
        wordsFile2 = [];
        result = [];

        lines = [line.strip() for line in f1]  # create a set of words from file 1
        for line in lines:
            tokenizedline = tokenize(line.replace('\ufeff', ''));
            for word in tokenizedline:
                wordsFile1.append(word);

        lines = [line.strip() for line in f2]  # create a set of words from file 1
        for line in lines:
            tokenizedline = tokenize(line.replace('\ufeff', ''));
            for word in tokenizedline:
                wordsFile2.append(word);

            # now loop over each line of other file

        for word in wordsFile1:
            if word in wordsFile2 and word != ' ':  # if word in File 1 is found in File 2 then print it
                result.append(word)

    return result


def text2wordfreq(string, lowercase=False):
    """Calculate word frequencies for a text written in English.

    Handling of hyphenation and contractions is left to your discretion.

    Your function must make use of the `tokenize` function above.

    Args:
        string (str): A string containing English.
        lowercase (bool, optional): Convert words to lowercase before calculating their
            frequency.

    Returns:
        dict: A dictionary with words as keys and frequencies as values.

    """


    from collections import Counter
    lst = Counter(tokenize(string, lowercase)).most_common()

    dictLst = dict(lst)

    return dictLst


def lexical_density(string):
    """Calculate the lexical density of a string containing English words.

    The lexical density of a sequence is defined to be the number of
    unique words divided by the number of total words. The lexical
    density of the sentence "The dog ate the hat." is 4/5.

    Ignore capitalization. For example, "The" should be counted as the same
    type as "the".

    This function should use the `text2wordfreq` function.

    Args:
        string (str): A string containing English.

    Returns:
        float: Lexical density.

    """
    # YOUR CODE HERE

    from collections import Counter
    tokenizedStr = string.strip('.').split(' ')

    c = Counter(tokenize(string.strip('.'), True));

    data = list(c);
    data.remove(' ');

    result = len(data)/len(tokenizedStr)
    return result


def hashtags(string):
    """Extract hashtags from a string.

    For example, the string `"RT @HouseGOP: The #StateOfTheUnion is strong."`
    contains the hashtag `#StateOfTheUnion`.

    Args:
        string (str): A string containing English.

    Returns:
        list: A list, possibly empty, containing hashtags.

    """

    lst = string.split(' ');
    result = [li for li in lst if li.startswith('#')];

    return result

def jaccard_similarity(text1, text2):
    """Calculate Jaccard Similarity between two texts.

    The Jaccard similarity (coefficient) or Jaccard index is defined to be the
    ratio between the size of the intersection between two sets and the size of
    the union between two sets. In this case, the two sets we consider are the
    set of words extracted from `text1` and `text2` respectively.

    This function should ignore capitalization. A word with a capital
    letter should be treated the same as a word without a capital letter.

    Args:
        text1 (str): A string containing English words.
        text2 (str): A string containing English words.

    Returns:
        float: Jaccard similarity

    """

    set1 = set(text1.split());
    set2 = set(text2.split());

    num = set.intersection(set1, set2);
    denom = set.union(set1, set2);

    return len(num)/len(denom);


# DO NOT EDIT CODE BELOW THIS LINE

import unittest

import numpy as np


class TestAssignment2(unittest.TestCase):

    def test_argmax(self):
        self.assertEqual(argmax([0, 1, 2]), 2)
        self.assertEqual(argmax([3, 1, 2]), 0)

    def test_tokenize(self):
        words = tokenize("Colorless green ideas sleep furiously.", True)
        self.assertIn('green', words)
        self.assertIn('colorless', words)
        words = tokenize('The rain  in spain is  mainly in the plain.', False)
        self.assertIn('The', words)
        self.assertIn('rain', words)

    def test_text2wordfreq(self):
        counts = text2wordfreq("Colorless green ideas sleep furiously. Green ideas in trees.", True)
        self.assertEqual(counts['green'], 2)
        self.assertEqual(counts['sleep'], 1)
        self.assertIn('colorless', counts)
        self.assertNotIn('hello', counts)


    def test_shared_words(self):
        self.assertEqual(shared_words('the hat', 'red hat'), {'hat'})


    def test_shared_words_from_filenames(self):
        # the use of the os.path functions is required so that filenames work
        # on Windows and Unix/Linux systems.
        import os
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        filename1 = os.path.join(data_dir, '1984-chp01.txt')
        filename2 = os.path.join(data_dir, 'animal-farm-chp01.txt')
        words = shared_words_from_filenames(filename1, filename2)
        self.assertGreater(len(words), 3)
        self.assertIn('already', words)


    def test_lexical_density(self):
        self.assertAlmostEqual(lexical_density("The cat"), 1)
        self.assertAlmostEqual(lexical_density("The cat in the hat."), 4/5)
        tweet = """RT @HouseGOP: The #StateOfTheUnion isn't strong for the 8.7 million Americans out of work. #SOTU http://t.co/aa7FWRCdyn"""
        self.assertEqual(len(hashtags(tweet)), 2)

    def test_jaccard_similarity(self):
        text1 = "Eight million Americans"
        text2 = "Americans in the South"
        self.assertAlmostEqual(jaccard_similarity(text1, text2), 1/6)

    def test_hashtags(self):
        tweet = """RT @HouseGOP: The #StateOfTheUnion isn't strong for the 8.7 million Americans out of work. #SOTU http://t.co/aa7FWRCdyn"""
        self.assertEqual(len(hashtags(tweet)), 2)

if __name__ == '__main__':
    unittest.main()
