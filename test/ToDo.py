#!/usr/bin/env python
# ------------------------------------------------------
#
#  ChirPy 0.1
#
#  https://hartree.chimie.ens.fr/sjaehnigen/ChirPy.git
#
#  2010-2016 Arne Scherrer
#  2014-2019 Sascha Jähnigen
#
#
# ------------------------------------------------------


class TestMathematics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestStatistics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestPhysics(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestClasses(unittest.TestCase):
    # Later for trajectory class: load vels, find unknown elements

    def setUp(self):
        pass

    def tearDown(self):
        pass

# also test the Iterators (important if they return correct shapes, especially
# CPMD)

#    @unittest.expectedFailure
#    def test_fail(self):
#        self.assertEqual(1, 0, "broken")


class TestInterfaces(unittest.TestCase):

    def setUp(self):
        self.dir = _test_dir + '/read_write'

    def tearDown(self):
        pass

    def test_cpmdReader(self):
        for _i, _n in zip(['GEOMETRY', 'MOMENTS', 'TRAJECTORY'],
                          [(1, 208, 6), (5, 288, 9), (6, 208, 6)]):

            data = cpmd.cpmdReader(self.dir + '/' + _i,
                                   filetype=_i,
                                   kinds=['X']*_n[1])['data']

            self.assertTrue(np.array_equal(
                data,
                np.genfromtxt(self.dir + '/data_' + _i).reshape(_n)
                ))

        # Some Negatives
        with self.assertRaises(ValueError):
            data = cpmd.cpmdReader(self.dir + '/MOMENTS_broken',
                                   filetype='MOMENTS',
                                   kinds=['X']*288)['data']
            data = cpmd.cpmdReader(self.dir + '/MOMENTS',
                                   filetype='MOMENTS',
                                   kinds=['X']*286)['data']
        # Test range
        data = cpmd.cpmdReader(self.dir + '/' + _i,
                               filetype='TRAJECTORY',
                               kinds=['X']*_n[1],
                               range=(2, 3, 8),
                               )['data']
        self.assertTrue(np.array_equal(
            data,
            np.genfromtxt(self.dir + '/data_TRAJECTORY').reshape(_n)[2:8:3]
            ))

    def test_cpmdWriter(self):
        data = cpmd.cpmdReader(self.dir + '/TRAJECTORY',
                               filetype='TRAJECTORY',
                               kinds=['X']*208)['data']

        cpmd.cpmdWriter(self.dir + '/OUT', data, write_atoms=False)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            data2 = cpmd.cpmdReader(self.dir + '/OUT',
                                    filetype='TRAJECTORY',
                                    kinds=cpmd.cpmd_kinds_from_file(self.dir + '/OUT')
                                    )['data']
        self.assertTrue(np.array_equal(data, data2))



class TestGenerators(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


# Test the binaries (all arguments)

class TestVisualisation(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestMdanalysis(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


class TestTest:
    def setUp(self):
        # Change paths after moving file
        self.dir = _test_dir

    def tearDown(self):
        pass

    def test_test(self):
        pass


if __name__ == '__main__':
    # os.system('bash %s/check_methods.sh %s/..' % (_test_dir, _test_dir))
    unittest.main()
