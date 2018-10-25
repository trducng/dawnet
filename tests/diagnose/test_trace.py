import numpy as np
import torch
import torch.nn as nn

import dawnet.diagnose.trace as trace
import dawnet.utils.tests as tests


class TestTrace(tests.TestCase):

    @tests.testprogress
    def test_trace_maxpool2d_get_output(self):
        """Test `trace.trace_maxpool2d` with various conditions"""

        # kernel_size 2, stride 2, padding 0
        input_map = np.arange(32).astype(np.float32).reshape(1,2,4,4)
                                                        # pylint: disable=E1101
        input_map = torch.FloatTensor(input_map)
        expected_result = torch.FloatTensor(
            [[[0, 0, 0, 0], [0, 5, 0, 7], [0, 0, 0, 0], [0, 13, 0, 15]],
            [[0, 0, 0, 0], [0, 21, 0, 23], [0, 0, 0, 0], [0, 29, 0, 31]]]
        ).unsqueeze(0)
        self._test_trace_maxpool2d_get_output(input_map, expected_result,2,2,0)

    @tests.testprogress
    def test_trace_maxpool2d_get_idx(self):
        """Test `trace.trace_maxpool2d` with various conditions"""

        # kernel_size 2, stride 2, padding 0
        input_map = np.arange(32).astype(np.float32).reshape(1,2,4,4)
                                                        # pylint: disable=E1101
        input_map = torch.FloatTensor(input_map)
        expected_result = set([(0, 0), (0, 1), (1, 0), (1, 1)])
        indices = [0, 0]
        self._test_trace_maxpool2d_get_idx(
            input_map, indices, expected_result, 2, 2, 0)

    def _test_trace_maxpool2d_get_output(self, input_map, expected_result,
                                         kernel_size, stride, padding):
        """Test `trace.trace_maxpool2d`"""
        maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding,
            return_indices=True)
        output_map, indices = maxpool(input_map)
        input_map_result = trace.trace_maxpool2d(indices, output_map,
            kernel_size, stride, padding)

        self.assertEqual(
                                                        # pylint: disable=E1101
            torch.all(torch.eq(input_map_result, expected_result)).item(),
            1
        )

    def _test_trace_maxpool2d_get_idx(self, input_map, indices, expected_result,
                                      kernel_size, stride, padding):
        """Test `trace.trace_maxpool2d`"""
        maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding)
        output_map = maxpool(input_map)
        corners = trace.trace_maxpool2d(
            indices, output_map, kernel_size, stride, padding)

        self.assertSetEqual(set(corners), set(expected_result))
