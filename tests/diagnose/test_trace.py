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

    @tests.testprogress
    def test_trace_conv2d(self):
        """Test `trace_conv2d` with various arguments"""

        # test unequal input map
        self._test_trace_conv2d(indices=(10, 1), input_map=(15, 10), 
            kernel_size=3, stride=1, padding=(1,2))

        # test normal VALID padding
        self._test_trace_conv2d(indices=(5, 5), input_map=(62, 62),
            kernel_size=3, stride=1, padding=(0, 0))
        
        # test normal SAME padding
        self._test_trace_conv2d(indices=(5, 5), input_map=(62, 62),
            kernel_size=3, stride=1, padding=(1, 1))

        # test stride != 1
        self._test_trace_conv2d(indices=(2, 3), input_map=(62, 62),
            kernel_size=3, stride=2, padding=1)

        # test other kernel size
        self._test_trace_conv2d(indices=(3, 2), input_map=(62, 62),
            kernel_size=(6, 3), stride=2, padding=1)
        
        # test stride == kernel
        self._test_trace_conv2d(indices=(10, 12), input_map=(62, 62),
            kernel_size=2, stride=2, padding=0)
        
        # test stride > kernel
        self._test_trace_conv2d(indices=(12, 10), input_map=(64, 64),
            kernel_size=2, stride=4, padding=1)

    def _test_trace_conv2d(self, indices, input_map, kernel_size, stride=None,
        padding=0, dilation=1):
        """Test `trace_conv2d`"""
        if isinstance(padding, int):
            padding = [padding, padding]

        input_map = np.arange(int(np.prod(input_map))).reshape(1,1,*input_map)
                                                        # pylint: disable=E1101
        input_map = torch.FloatTensor(input_map)
        conv = nn.Conv2d(in_channels=input_map.shape[1],
                         out_channels=1,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation)
        output_map = conv(input_map)
        input_indices = trace.trace_conv2d(indices,
            (output_map.shape[2], output_map.shape[3]),
            kernel_size,
            stride, padding, dilation)
        top, bottom = input_indices[0][0], input_indices[-1][0] + 1
        left, right = input_indices[0][1], input_indices[-1][1] + 1

                                                        # pylint: disable=E1101
        expected = torch.sum(output_map[:,:,indices[0],indices[1]])
        target_y = min(padding[0], indices[0])
        target_x = min(padding[1], indices[1])
        conv.stride = (1, 1)
        actual = torch.sum(
            conv(input_map[:,:,top:bottom,left:right])
            [:,:,target_y,target_x])

        # print('Expected {}, actual {}'.format(expected, actual))
        # print('top {} bottom {} left {} right {}'.format(top, bottom, left, right))
        self.assertAlmostEqual(float(expected), float(actual))

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
