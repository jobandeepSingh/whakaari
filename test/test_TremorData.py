import unittest
from unittest.mock import patch

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from whakaari import TremorData

# GIFLENS-https://media1.giphy.com/media/DNBljVTPd8qBO/200.gif
class TestTremorData(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    @patch('whakaari.TremorData._assess')
    def test_constructor(self, mock_assess):
        '''Need to get data file and run _assess method'''
        
        # TODO Test for the data file??? Only really need to test the path is right? Or even that?
        #print(td.file)]

        # def side_effect():
        #     print("Showing how patch side effects work")
        # mock_assess.side_effect = side_effect

        td = TremorData()
        mock_assess.assert_called_once() # Good test
        # mock_assess.assert_not_called() # Shows what a failed test looks like
        del td

    def test_assess(self):
        # Make sure that correct eruption data is read
        
        # If no data file then update with default times

        # Reads a csv wtih correct start and end times

    def test_update(self):
        pass

    def test_compute_transforms(self):
        pass

    def test_is_eruption_in(self):
        pass

    def test_get_data(self):
        pass


if __name__ == "__main__":
    unittest.main()