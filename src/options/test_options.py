#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # lambda parameters
        self.isTrain = False
        return parser
