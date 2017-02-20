'''

Assignment #3 (Version 1.2)

    Working with Adjacency matrices, normalization and data extraction


This assignment requires the following packages:

- numpy
- pandas
- scikit-learn

'''

import os
import string


def load_nytimes_document_term_matrix_and_labels():
    """Load New York Times art and music articles.

    Articles are stored in a document-term matrix.

    See ``data/nytimes-art-music-simple.csv`` for the details.

    This function returns a tuple of two values. The Pythonic way to call this
    function is as follows:

        document_term_matrix, labels = load_nytimes_document_term_matrix_and_labels()

    Returns:
        (array, list): A document term matrix (as a Numpy array) and a list of labels.
    """

    import pandas as pd
    nytimes = pd.read_csv(os.path.join('data', 'nytimes-art-music-simple.csv'), index_col=0)
    labels = [document_name.rstrip(string.digits) for document_name in nytimes.index]
    return nytimes.values, labels


def normalize_document_term_matrix(document_term_matrix):
    """Normalize a document-term matrix by length.

    Each row in `document_term_matrix` is a vector of counts. Divide each
    vector by its length. Length, in this context, is just the sum of the
    counts or the Manhattan norm.

    For example, a single vector $(0, 1, 0, 1)$ normalized by length is $(0,
    0.5, 0, 0.5)$.

    Args:
        document_term_matrix (array): A document-term matrix of counts

    Returns:
        array: A length-normalized document-term matrix of counts

    """

    for row in document_term_matrix:
        row = np.linalg.norm(row)

    return document_term_matrix


def distance_matrix(document_term_matrix):
    """Calculate a NxN distance matrix given a document-term matrix with N rows.

    Each row in `document_term_matrix` is a vector of counts. Calculate the
    Euclidean distance between each pair of rows.

    Args:
        document_term_matrix (array): A document-term matrix of counts

    Returns:
        array: A square matrix of distances.

    """

    from sklearn.metrics.pairwise import euclidean_distances
    return euclidean_distances(document_term_matrix, document_term_matrix)


def jaccard_similarity_matrix(document_term_matrix):
    """Calculate a NxN similarity matrix given a document-term matrix with N rows.

    Each row in `document_term_matrix` is a vector of counts. Calculate the
    Jaccard similarity between each pair of rows.

    Args:
        document_term_matrix (array): A document-term matrix of counts

    Returns:
        array: A square matrix of similarities.

    """

    from sklearn.metrics.pairwise import pairwise_distances
    return (1 - pairwise_distances(document_term_matrix, metric="jaccard"))


def nearest_neighbors_classifier(new_vector, document_term_matrix, labels):
    """Return a predicted label for `new_vector`.

    You may use either Euclidean distance or Jaccard similarity.

    Args:
        new_vector (array): A vector of length V
        document_term_matrix (array): An array with shape (N, V)
        labels (list of str): List of N labels for the rows of `document_term_matrix`.

    Returns:
        str: Label predicted by the nearest neighbor classifier.

    """

    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(metric='euclidean')
    neigh.fit(document_term_matrix, labels)
    lbl = neigh.predict(new_vector)

    return lbl

def extract_hashtags(tweet):
    """Extract hashtags from a string.

    For example, the string `"RT @HouseGOP: The #StateOfTheUnion is strong."`
    contains the hashtag `#StateOfTheUnion`.

    The method used here needs to be robust. For example, the following tweet does
    not contain a hashtag: "This tweet contains a # but not a hashtag."

    Args:
        tweet (str): A tweet in English.

    Returns:
        list: A list, possibly empty, containing hashtags.

    """
    
    results = []
    lst = tweet.split(' ')

    for token in lst:
        if token.startswith('#'):
            if not (len(token) == 2 and token[1].isdigit()):
                results.append(token)

    return results


def extract_mentions(tweet):
    """Extract @mentions from a string.

    For example, the string `"RT @HouseGOP: The #StateOfTheUnion is strong."`
    contains the mention ``@HouseGOP``.

    The method used here needs to be robust. For example, the following tweet
    does not contain an @mention: "This tweet contains an email address,
    user@example.net."

    Args:
        tweet (str): A tweet in English.

    Returns:
        list: A list, possibly empty, containing @mentions.

    """

    import re

    lst = re.split(' |:', tweet);
    result = [li for li in lst if li.startswith('@') and len(li) > 1];

    return result


def adjacency_matrix_from_edges(pairs):
    """Construct and adjacency matrix from a list of edges.

    An adjacency matrix is a square matrix which records edges between vertices.

    This function turns a list of edges, represented using pairs of comparable
    elements (e.g., strings, integers), into a square adjacency matrix.

    For example, the list of pairs ``[('a', 'b'), ('b', 'c')]`` defines a tree
    with root node 'b' which may be represented by the adjacency matrix:

    ```
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]]
    ```

    where rows and columns correspond to the vertices ``['a', 'b', 'c']``.

    Vertices should be ordered using the usual Python sorting functions. That
    is vertices with string names should be alphabetically ordered and vertices
    with numeric identifiers should be sorted in ascending order.

    Args:
        pairs (list of [int] or list of [str]): Pairs of edges

    Returns:
        (array, list): Adjacency matrix and list of vertices. Note
            that this function returns *two* separate values, a Numpy
            array and a list.

    """

    edgeList = [];
    edgeListInts = [];
    for edge in pairs:
        for vertex in edge:
            edgeListInts.append(ord(vertex) - 97);
            if vertex not in edgeList:
                edgeList.append(vertex);

    N = len(edgeList);
    matrix = [[0 for i in range(N)] for j in range(N)];

    i = 0;
    while i < N:
        matrix[edgeListInts[i]][edgeListInts[i] + 1] = 1;
        matrix[edgeListInts[i] + 1][edgeListInts[i]] = 1;
        i += 2;

    return matrix, edgeList

def mentions_adjacency_matrix(list_of_mentions):
    """Construct an adjacency matrix given lists of mentions.

    Given the following list of mentions:

    - [@nytimes]
    - [@nytimes, @washtimes]
    - [@foxandfriends]
    - [@nytimes]
    - [@washtimes, @foxandfriends]

    One would expect as a result the following adjacency matrix:

      [[ 0.,  1.,  0.,  1.,  0.],
       [ 1.,  0.,  0.,  1.,  1.],
       [ 0.,  0.,  0.,  0.,  1.],
       [ 1.,  1.,  0.,  0.,  0.],
       [ 0.,  1.,  1.,  0.,  0.]]

    Where we can see, for example, that the third node is connected to the
    fifth node (because both mention ``@foxandfriends``).

    Args:
        list_of_mentions (list of list of str): List of mentions

    Returns:
        array: An adjacency matrix.

    """

    N = len(list_of_mentions);
    result = [[0 for i in range(N)] for j in range(N)];

    list_of_mentions_copy = list_of_mentions;
    edgeList = [];

    for i, mentions in enumerate(list_of_mentions):
        for j, mentionsCopy in enumerate(list_of_mentions_copy):
            common = set.intersection(set(mentions), set(mentionsCopy));
            if len(common) > 0 and i != j:
                result[i][j] = 1;
            else:
                result[i][j] = 0;

    return np.array(result);


# DO NOT EDIT CODE BELOW THIS LINE

import unittest

import numpy as np


class TestAssignment3(unittest.TestCase):


    def test_extract_hashtags1(self):
        tweet = """RT @HouseGOP: The #StateOfTheUnion isn't strong for the 8.7 million Americans out of work. #SOTU http://t.co/aa7FWRCdyn"""
        self.assertEqual(len(extract_hashtags(tweet)), 2)

    def test_extract_mentions1(self):
        tweet = """RT @HouseGOP: The #StateOfTheUnion isn't strong for the 8.7 million Americans out of work. #SOTU http://t.co/aa7FWRCdyn"""
        self.assertEqual(len(extract_mentions(tweet)), 1)
        self.assertIn('@HouseGOP', extract_mentions(tweet))

    def test_adjacency_matrix_from_edges1(self):
        pairs = [('a', 'b'), ('b', 'c')]
        expected = np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])
        A, nodes = adjacency_matrix_from_edges(pairs)
        self.assertEqual(nodes, ['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(A, expected)

    def test_normalize_document_term_matrix1(self):
        dtm, _ = load_nytimes_document_term_matrix_and_labels()
        self.assertEqual(dtm.shape, normalize_document_term_matrix(dtm).shape)

    def test_distance_matrix1(self):
        dtm, _ = load_nytimes_document_term_matrix_and_labels()
        dist = distance_matrix(dtm)
        self.assertEqual(dist.shape[0], dist.shape[1])

    def test_jaccard_similarity_matrix(self):
        dtm, _ = load_nytimes_document_term_matrix_and_labels()
        similarity = jaccard_similarity_matrix(dtm)
        self.assertEqual(similarity.shape[0], similarity.shape[1])

    def test_mentions_adjacency_matrix1(self):
        list_of_mentions = [['@nytimes'], ['@nytimes']]
        A = mentions_adjacency_matrix(list_of_mentions)
        self.assertTrue(isinstance(A, np.ndarray))

if __name__ == '__main__':
    unittest.main()
