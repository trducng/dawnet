# Utility code to aid testing
# @author: _john
# ============================================================================
import os
import sys
import unittest


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        A little bit hack to make it easier to know which test is taken
        """
        super(TestCase, cls).setUpClass()

        process = os.popen('stty size', 'r')
        _, columns = process.read().split()
        columns = int(columns) - 5
        process.close()

        class_name = str(cls.__name__)
        class_name = class_name.center(int(columns))

        print("")
        print(class_name)


def testprogress(func): 
    """
    Show test name and progress during testing
    """
    def func_wrapper(*args, **kwargs):
        sys.stdout.write('\r')
        sys.stdout.write("Performing {} ... ".format(func.__name__))
        sys.stdout.flush()
        func(*args, **kwargs)
        print("DONE")
    return func_wrapper
