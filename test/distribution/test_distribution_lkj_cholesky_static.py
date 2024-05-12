# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import parameterize
from distribution import config

import paddle
from paddle.distribution import lkj_cholesky

paddle.enable_static()

np.random.seed(2024)
paddle.seed(2024)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'concentration'),
    [('one-dim', 1.0)],
)
class TestLKJCholeskyShapeOneDim(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.conc = paddle.static.data(
                'concentration',
                (),
                'float32',
            )
            self.feeds = {
                'concentration': self.concentration,
            }

    def gen_cases(self):
        extra_shape = (
            self._paddle_lkj_cholesky.dim,
            self._paddle_lkj_cholesky.dim,
        )
        cases = [
            {
                'input': (),
                'expect': () + extra_shape,
            },
            {
                'input': (2, 2),
                'expect': (2, 2) + extra_shape,
            },
        ]
        return cases

    def test_onion_sample_shape(self):
        sample_method = 'onion'
        self._test_sample_shape_dim(sample_method)

    def test_cvine_sample_shape(self):
        sample_method = 'cvine'
        self._test_sample_shape_dim(sample_method)

    def _test_sample_shape_dim(self, sample_method):
        for dim in range(2, 4):
            self._test_sample_shape(dim, sample_method)

    def _test_sample_shape(self, dim, sample_method):
        with paddle.static.program_guard(self.program):
            self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(
                dim, self.conc, sample_method
            )
            cases = self.gen_cases()
            for case in cases:
                [data] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_lkj_cholesky.sample(
                        case.get('input')
                    ),
                )
            self.assertTrue(tuple(data.shape) == case.get('expect'))


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'concentration'),
    [
        (
            'multi',
            parameterize.xrand(
                (2,),
                dtype='float32',
                max=1.0,
                min=0,
            ),
        ),
    ],
)
class TestLKJCholeskyShapeMulti(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.conc = paddle.static.data(
                'concentration',
                self.concentration.shape,
                self.concentration.dtype,
            )
            self.feeds = {
                'concentration': self.concentration,
            }

    def gen_cases(self):
        extra_shape = (
            len(self.concentration),
            self._paddle_lkj_cholesky.dim,
            self._paddle_lkj_cholesky.dim,
        )
        cases = [
            {
                'input': (),
                'expect': () + extra_shape,
            },
        ]
        return cases

    def test_onion_sample_shape(self):
        sample_method = 'onion'
        self._test_sample_shape_dim(sample_method)

    def test_cvine_sample_shape(self):
        sample_method = 'cvine'
        self._test_sample_shape_dim(sample_method)

    def _test_sample_shape_dim(self, sample_method):
        for dim in range(2, 4):
            self._test_sample_shape(dim, sample_method)

    def _test_sample_shape(self, dim, sample_method):
        with paddle.static.program_guard(self.program):
            self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(
                dim, self.conc, sample_method
            )
            cases = self.gen_cases()
            for case in cases:
                [data] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_lkj_cholesky.sample(
                        case.get('input')
                    ),
                )
            self.assertTrue(tuple(data.shape) == case.get('expect'))


if __name__ == '__main__':
    unittest.main()
