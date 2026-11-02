from copy import deepcopy
from io import StringIO

import numpy as np

from project_config import ROOT, load_config
from rl.tgax_environment import TgaxControlEnvironment, configured_default_action
from sim.backend.base_backend import ControlAction
from sim.backend.tgax_ns3_backend import TgaxNs3Backend


def split_config():
    config=deepcopy(load_config(ROOT/"configs/tgax_ddqn_full40_multiac_v2.yaml"))
    config["action"]["profile_application_mode"]="split_ap_sta"
    config["action"]["default_action"]=[16,16,0,0,0,0]
    return config


def bare_environment(config):
    env=TgaxControlEnvironment.__new__(TgaxControlEnvironment)
    env.config=config
    env.profile_application_mode="split_ap_sta"
    env.profile_branch_count=2
    return env


def test_split_action_encoding_and_role_specific_fallback():
    config=split_config(); env=bare_environment(config)
    action=np.asarray([16,7,2,1,2,2])
    features=env.action_features(action)
    assert features.shape==(38,)
    decoded=env.decode(action,custom=True)
    assert decoded.profile==0 and decoded.custom_growth is False
    assert decoded.sta_profile==7 and decoded.sta_custom_growth is True
    assert (decoded.alpha,decoded.beta1,decoded.beta2,decoded.gamma)==(3,2.0,2.8,7)
    np.testing.assert_array_equal(configured_default_action(config),[16,16,0,0,0,0])


def test_split_backend_emits_step2_protocol():
    backend=TgaxNs3Backend.__new__(TgaxNs3Backend)
    backend.process=type("Process",(),{"stdin":StringIO()})()
    backend.offset=0.; backend.time=0.; backend.raw=StringIO()
    lines=iter(["END 1\n"])
    backend._read=lambda: next(lines)
    action=ControlAction(3,True,2,1.6,2.8,6,sta_profile=8,sta_custom_growth=False)
    backend._exchange(action,1.0)
    assert backend.process.stdin.getvalue()=="STEP2 1.0 3 1 8 0 2 16 28 6\n"


def test_cpp_backend_applies_role_specific_profiles():
    source=(ROOT/"ns3_ext/tgax-control.cc").read_text(encoding="utf-8")
    assert 'verb == "STEP2"' in source
    assert 'const uint32_t profile = isAp ? apProfile : staProfile;' in source
    assert 'const bool custom = isAp ? apCustom : staCustom;' in source
