from multiprocessing import Process
import unittest


class MyTestCase(unittest.TestCase):
    def test_all(self):
        from .. import tuning_settings
        from ..tuning import main

        for k in tuning_settings.__dict__.keys():
            if not isinstance(getattr(tuning_settings, k),
                              tuning_settings.Experiment):
                continue
            print("Testing {}".format(k))
            p = Process(target=main,
                        args=(k, ),
                        kwargs={"testrun": True})
            p.start()
            p.join()
            p.terminate()

            self.assertIsNotNone(p.exitcode)
            self.assertIs(p.exitcode, 0)

if __name__ == '__main__':
    unittest.main()
