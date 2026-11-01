"""Apply repeatable CW-growth and queue-accounting patches to pinned ns-3.44."""
from __future__ import annotations

import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fix_queue_drop_trace(root: Path) -> None:
    """Restore Queue's rejection accounting; do not change the drop decision."""
    path=root/"src/wifi/model/wifi-mac-queue.cc"
    old=path.read_text(encoding="utf-8")
    marker="// Research accounting fix: scheduler rejection bypasses Queue::DoEnqueue."
    if marker in old:
        return
    anchor='''        // the given item must be dropped
        SetMaxSize(currSize);
        return false;'''
    if old.count(anchor)!=1:
        raise RuntimeError("Unexpected ns-3 queue rejection implementation")
    new=old.replace(anchor,'''        // the given item must be dropped
        SetMaxSize(currSize);
        // Research accounting fix: scheduler rejection bypasses Queue::DoEnqueue.
        DropBeforeEnqueue(item);
        return false;''')
    path.write_text(new,encoding="utf-8",newline="\n")
    (ROOT/"ns3_ext/queue_drop_trace.patch").write_text("".join(difflib.unified_diff(
        old.splitlines(True),new.splitlines(True),fromfile="a/src/wifi/model/wifi-mac-queue.cc",
        tofile="b/src/wifi/model/wifi-mac-queue.cc")),encoding="utf-8")


def add_nonht_suite(root: Path) -> None:
    """Add a separate suite using the unchanged upstream non-HT test cases."""
    path=root/"src/wifi/test/wifi-txop-test.cc"
    old=path.read_text(encoding="utf-8")
    if "ResearchNonHtTxopTestSuite" in old:
        return
    addition='''
// A separate scope-specific suite; the original wifi-txop suite remains unchanged.
class ResearchNonHtTxopTestSuite : public TestSuite
{
  public:
    ResearchNonHtTxopTestSuite() : TestSuite("wifi-txop-nonht", Type::UNIT)
    {
        AddTestCase(new WifiTxopTest({true,true,false,false,true}), TestCase::Duration::QUICK);
        AddTestCase(new WifiTxopTest({true,false,true,false,false}), TestCase::Duration::QUICK);
        AddTestCase(new WifiTxopTest({true,true,true,true,true}), TestCase::Duration::QUICK);
    }
};
static ResearchNonHtTxopTestSuite g_researchNonHtTxopTestSuite;
'''
    path.write_text(old+addition,encoding="utf-8",newline="\n")
    diff="".join(difflib.unified_diff(old.splitlines(True),(old+addition).splitlines(True),
                 fromfile="a/src/wifi/test/wifi-txop-test.cc",tofile="b/src/wifi/test/wifi-txop-test.cc"))
    (ROOT/"ns3_ext/nonht_test.patch").write_text(diff,encoding="utf-8")


def add_internal_collision_trace(root: Path) -> None:
    """Expose EDCAF virtual collisions without changing the contention decision."""
    h = root / "src/wifi/model/txop.h"
    cc = root / "src/wifi/model/txop.cc"
    fem = root / "src/wifi/model/frame-exchange-manager.cc"
    old_h, old_cc, old_fem = (p.read_text(encoding="utf-8") for p in (h, cc, fem))
    marker = 'AddTraceSource("InternalCollision"'
    if marker in old_cc:
        return
    method_anchor = "    void UpdateFailedCw(uint8_t linkId);"
    if old_h.count(method_anchor) != 1:
        raise RuntimeError("Unexpected Txop collision trace method anchor")
    new_h = old_h.replace(method_anchor, method_anchor + '''

    // Trace-only research hook for an EDCAF losing an internal virtual collision.
    void NotifyInternalCollisionTrace(uint8_t linkId) { m_internalCollisionTrace(linkId); }
''')
    callback_anchor = "    CwValueTracedCallback m_cwTrace;           //!< CW trace value"
    if new_h.count(callback_anchor) != 1:
        raise RuntimeError("Unexpected Txop collision trace callback anchor")
    new_h = new_h.replace(callback_anchor, callback_anchor + '''
    TracedCallback<uint8_t /* linkId */> m_internalCollisionTrace; //!< internal EDCAF collision
''')
    type_anchor = '''            .AddTraceSource("CwTrace",
                            "Trace source for contention window values",
                            MakeTraceSourceAccessor(&Txop::m_cwTrace),
                            "ns3::Txop::CwValueTracedCallback");'''
    if old_cc.count(type_anchor) != 1:
        raise RuntimeError("Unexpected Txop TypeId trace anchor")
    new_cc = old_cc.replace(type_anchor, '''            .AddTraceSource("CwTrace",
                            "Trace source for contention window values",
                            MakeTraceSourceAccessor(&Txop::m_cwTrace),
                            "ns3::Txop::CwValueTracedCallback")
            .AddTraceSource("InternalCollision",
                            "Trace source fired when this EDCAF loses an internal collision",
                            MakeTraceSourceAccessor(&Txop::m_internalCollisionTrace),
                            "ns3::TracedCallback::Uint8");''')
    fem_anchor = '''FrameExchangeManager::NotifyInternalCollision(Ptr<Txop> txop)
{
    NS_LOG_FUNCTION(this);
'''
    if old_fem.count(fem_anchor) != 1:
        raise RuntimeError("Unexpected internal collision implementation anchor")
    new_fem = old_fem.replace(fem_anchor, fem_anchor + '''
    // Observation only: notify the losing EDCAF before retry/CW state is updated.
    txop->NotifyInternalCollisionTrace(m_linkId);
''')
    diff = ""
    for path, before, after in ((h, old_h, new_h), (cc, old_cc, new_cc), (fem, old_fem, new_fem)):
        rel = path.relative_to(root).as_posix()
        diff += "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                             fromfile="a/" + rel, tofile="b/" + rel))
        path.write_text(after, encoding="utf-8", newline="\n")
    (ROOT / "ns3_ext/internal_collision_trace.patch").write_text(diff, encoding="utf-8")


def patch(root: Path = ROOT / "vendor/ns-3.44") -> None:
    fix_queue_drop_trace(root)
    add_nonht_suite(root)
    h = root / "src/wifi/model/txop.h"
    cc = root / "src/wifi/model/txop.cc"
    old_h, old_cc = h.read_text(encoding="utf-8"), cc.read_text(encoding="utf-8")
    if "SetResearchCwGrowth" in old_h:
        add_internal_collision_trace(root)
        return
    anchor = "    void UpdateFailedCw(uint8_t linkId);"
    assert old_h.count(anchor) == 1
    new_h = old_h.replace(anchor, anchor + '''

    // Research extension: disabled by default, standard retry/ACK logic is unchanged.
    void SetResearchCwGrowth(bool enabled, uint32_t alpha, uint32_t beta1Tenths,
                             uint32_t beta2Tenths, uint32_t gamma)
    {
        m_researchGrowth = enabled;
        m_growthAlpha = alpha;
        m_growthBeta1 = beta1Tenths;
        m_growthBeta2 = beta2Tenths;
        m_growthGamma = gamma;
    }
''')
    anchor_h = "    DroppedMpdu m_droppedMpduCallback;"
    assert new_h.count(anchor_h) == 1
    new_h = new_h.replace(anchor_h, '''    bool m_researchGrowth{false};
    uint32_t m_growthAlpha{1}, m_growthBeta1{20}, m_growthBeta2{20}, m_growthGamma{7};
''' + anchor_h)
    anchor_cc = '''        link.cw =
            std::min(GetMaxCw(linkId), (1 << link.staRetryCount) * (GetMinCw(linkId) + 1) - 1);'''
    assert old_cc.count(anchor_cc) == 1
    new_cc = old_cc.replace(anchor_cc, '''        if (m_researchGrowth)
        {
            if (link.staRetryCount <= m_growthGamma)
            {
                const uint64_t beta = link.staRetryCount <= m_growthAlpha
                                          ? m_growthBeta1 : m_growthBeta2;
                // Integer ceil for decimal candidates avoids floating-point drift.
                link.cw = std::min<uint64_t>(GetMaxCw(linkId),
                                            (beta * (link.cw + 1) + 9) / 10 - 1);
            }
        }
        else
        {
            link.cw = std::min(GetMaxCw(linkId),
                              (1 << link.staRetryCount) * (GetMinCw(linkId) + 1) - 1);
        }''')
    diff = ""
    for path, before, after in [(h, old_h, new_h), (cc, old_cc, new_cc)]:
        rel = path.relative_to(root).as_posix()
        diff += "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile="a/" + rel, tofile="b/" + rel))
        path.write_text(after, encoding="utf-8", newline="\n")
    (ROOT / "ns3_ext/cw_growth.patch").write_text(diff, encoding="utf-8")
    add_internal_collision_trace(root)


if __name__ == "__main__":
    patch()
