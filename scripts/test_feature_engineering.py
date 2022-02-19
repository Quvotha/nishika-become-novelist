import unittest

import pandas as pd

from feature_extraction import TextFeatureExtractor, PostHistoryFeatureExtractor


class TestTextFeatureExtractor(unittest.TestCase):

    def test_transform(self):
        input_ = pd.DataFrame(
            index=[1, 0],
            data=[
                '\n'.join(['ABC', 'DEF', 'GHI']),
                '\n'.join(['ABC', '', 'DEF', 'GHI']),
            ],
            columns=['text']
        )
        expected = pd.DataFrame(
            index=input_.index,
            data={
                'txt_len_text': [11, 12],
                'txt_nlines_text': [3, 4],
                'txt_nblines_text': [0, 1],
                'txt_sblines_text': [0., 0.25],
                'txt_avelen_text': [3., 3.]
            }
        )
        expected['txt_len_text'] = expected['txt_len_text'].astype('int')
        expected['txt_nlines_text'] = expected['txt_nlines_text'].astype('int')
        expected['txt_nblines_text'] = expected['txt_nblines_text'].astype('int')
        expected['txt_sblines_text'] = expected['txt_sblines_text'].astype('float32')
        expected['txt_avelen_text'] = expected['txt_avelen_text'].astype('float32')
        transformer = TextFeatureExtractor(['text']).fit(input_)
        output = transformer.transform(input_)
        pd.testing.assert_frame_equal(output, expected)


if __name__ == '__main__':
    unittest.main()
