'''

Assignment #1

    - Getting familiar with Python and Git

'''


def character_proportion(character, text):
    """Calculate the proportion of characters in text which are `character`.

    The character "o" occurs twice in the six-character string "Boston". Thus
    the proportion of characters which are "o" is one third.

    Args:
        character: A character (aka a string of length 1) such as "o".
        text: A string.

    Returns:
        float: The proportion of characters in `text` which are `character`.

    """
    
    count = 0;
    strlen = len(text);
    
    for c in range(0, strlen):
        if text[c] == character:
            count = count + 1;

    result = count/strlen;

    print("\n{} {}".format("Result of function character_proportion : ", result));
    
    return result


def random_sum(n):
    """Calculate the sum of `n` random numbers.

    This is an artificial exercise to introduce you to the Python ``random``
    module.

    Draw `n` numbers from between 0 and 1 and return their sum.

    Args:
        n: A non-negative integer.

    Returns:
        float: The sum of `n` random numbers.

    """
    
    import random

    sum = 0;
    for c in range(0, n): 
        sum = sum + random.uniform(0.0,1.0);

    print("\n{} {}".format("Result of function random_sum : ", sum));
    return sum;


def character_proportion_monte_carlo(character, text, n):
    """Calculate the proportion of characters in text which are `character` using random sampling.

    See the docstring for `character_proportion` for an introduction.

    In very long texts (e.g., all of Wikipedia) calculating the proportion of
    characters which are a given character is costly. It is less costly to
    estimate the proportion by randomly sampling characters from the text and
    calculating the proportion of characters in the random sample which are the
    given character.

    Take `n` characters at random *without replacement* from `text` and return
    the proportion of these `n` characters which are `character.

    Args:
        character: A character (aka a string of length 1) such as "o".
        text: A string.
        n: An integer greater than 0.

    Returns:
        float: The *estimated* proportion of characters in `text` which are `character`.

    """
    
    import random

    count = 0;
    strlen = len(text);
    
    rand_smpl = [text[i] for i in random.sample(range(strlen), n)]

    result = character_proportion(character,rand_smpl);
    
    print("\n{} {}".format("Result of function character_proportion_monte_carlo : ", result));
    return result


#Unit tests

import unittest


class TestAssignment1(unittest.TestCase):

    def test_character_proportion1(self):
        self.assertGreater(character_proportion('o', 'Boston'), 0)
        self.assertLess(character_proportion('o', 'Boston'), 1)
        self.assertEqual(character_proportion('o', 'Boston'), 1/3)

    def test_random_sum1(self):
        self.assertIsNotNone(random_sum(100))
        self.assertGreater(random_sum(100), 0)
        self.assertGreater(random_sum(1000), 5)

    def test_character_proportion_monte_carlo1(self):
        self.assertGreaterEqual(character_proportion('o', 'Boston'), 0)
        self.assertLessEqual(character_proportion('o', 'Boston'), 1)
        self.assertGreaterEqual(character_proportion_monte_carlo('o', 'Boston', 6), 0)

if __name__ == '__main__':
    unittest.main()
