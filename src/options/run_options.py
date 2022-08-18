#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .base_options import BaseOptions


class RunOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=50, help='how many test inputs to run')
        parser.add_argument('--n_samples', type=int, default=5, help='#samples')
        #parser.add_argument('--no_encode', action='store_true', help='do not produce encoded inputs')
        #parser.add_argument('--sync', action='store_true', help='use the same latent code for different inputs')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--input_path', type=str, help='input path')
        parser.add_argument('--output_path', type=str, help='output path')
        parser.add_argument('--z0', type=str, help='first latent code of input path')
        parser.add_argument('--z1', type=str, help='last latent code of input path')

        self.isTrain = False
        return parser
