import unittest
from project_config import ROOT,load_config
from sim.edca.profile_table import load_profiles
from sim.edca.cw_growth import Growth


class EdcaTests(unittest.TestCase):
    def test_profiles_and_beb_equivalence(self) -> None:
        profiles=load_profiles(ROOT/"configs/profiles.json")
        self.assertEqual(len(profiles),16)
        for p in profiles:
            for v in p.values():
                actual=Growth(1,2,2,7).table(v["CWmin"],v["CWmax"],7)
                expected=[min(v["CWmax"],(v["CWmin"]+1)*2**n-1) for n in range(8)]
                self.assertEqual(actual,expected)

    def test_balanced_profiles_are_legal_and_include_low_priority_actions(self) -> None:
        profiles=load_profiles(ROOT/"configs/profiles_v2_balanced.json")
        default=load_profiles(ROOT/"configs/profiles.json")[0]
        self.assertEqual(len(profiles),16)
        self.assertEqual(profiles[0],default)
        self.assertTrue(any(p["BE"]["AIFSN"]<default["BE"]["AIFSN"] or
                            p["BE"]["CWmin"]<default["BE"]["CWmin"] or
                            p["BE"]["TXOP_ms"]>0 for p in profiles[1:]))
        self.assertTrue(any(p["BK"]["AIFSN"]<default["BK"]["AIFSN"] or
                            p["BK"]["CWmin"]<default["BK"]["CWmin"] or
                            p["BK"]["TXOP_ms"]>0 for p in profiles[1:]))

    def test_piecewise_saturation_and_ceil(self) -> None:
        g=Growth(2,1.2,2.6,5)
        self.assertEqual(g.table(15,1023,7),[15,19,23,62,163,426,426,426])
        with self.assertRaises(ValueError): Growth(1,2,1.2,7)

    def test_time_scales_remain_separate(self) -> None:
        config=load_config(ROOT/"configs/quick.yaml")
        self.assertEqual(config["timing"]["history_s"] / config["timing"]["sta_sample_s"],10)
        self.assertEqual(config["timing"]["decision_s"],1)


if __name__=="__main__": unittest.main()
