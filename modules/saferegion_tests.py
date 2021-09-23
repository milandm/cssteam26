import torch
import torch.nn as nn
from modules.saferegion import SafeRegion1d, SafeRegion2d, SafeRegion3d

import unittest

torch.set_printoptions(precision=10)


def compare_with_bn(sr, bn):
    err = False
    if not torch.allclose(sr.running_mean, bn.running_mean):
        print('Diff in running_mean: {} vs {}'.format(sr.running_mean, bn.running_mean))
        err = True

    if not torch.allclose(sr.running_var, bn.running_var):
        print('Diff in running_var: {} vs {}'.format(sr.running_var, bn.running_var))
        err = True

    return err


class TestSafeRegion(unittest.TestCase):
    def test_safe_region_1d(self):
        # C from an expected input of size (N, C)
        c = 10
        safe_region = SafeRegion1d(c, eps=0, momentum=None)
        batch_norm = nn.BatchNorm1d(c, eps=0, momentum=None, affine=False)
        for i in range(10):
            r1 = -100
            r2 = 100
            x = (r1 - r2) * torch.rand((13, c), dtype=torch.float) + r2
            safe_region(x)
            batch_norm(x)
        err = compare_with_bn(safe_region, batch_norm)
        self.assertEqual(err, False)

    def test_safe_region_2d(self):
        # C from an expected input of size (N, C, H, W)
        c = 3
        safe_region = SafeRegion2d(c, eps=0, momentum=None)
        batch_norm = nn.BatchNorm2d(c, eps=0, momentum=None, affine=False)
        r1 = -100
        r2 = 100
        for i in range(100):
            x = (r1 - r2) * torch.rand((100, c, 16, 16), dtype=torch.float) + r2
            safe_region(x)
            batch_norm(x)
        err = compare_with_bn(safe_region, batch_norm)
        self.assertEqual(err, False)

    def test_safe_region_3d(self):
        # C from an expected 5D input
        c = 100
        safe_region = SafeRegion3d(c, eps=0, momentum=None)
        batch_norm = nn.BatchNorm3d(c, eps=0, momentum=None, affine=False)
        r1 = -100
        r2 = 100
        for i in range(100):
            x = (r1 - r2) * torch.rand((100, c, 16, 16, 4), dtype=torch.float) + r2
            safe_region(x)
            batch_norm(x)
        err = compare_with_bn(safe_region, batch_norm)
        self.assertEqual(err, False)


if __name__ == '__main__':
    unittest.main()
