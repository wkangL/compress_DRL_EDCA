// SPDX-License-Identifier: GPL-2.0-only
// Real ns-3 Wi-Fi MAC backend. No learning framework is linked into this process.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/propagation-module.h"
#include "ns3/wifi-module.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <set>

using namespace ns3;

class MeasurementTag : public Tag
{
  public:
    uint32_t sta{0}, ac{0}, bytes{0};
    uint64_t created{0};
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("MeasurementTag").SetParent<Tag>().AddConstructor<MeasurementTag>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override { return GetTypeId(); }
    uint32_t GetSerializedSize() const override { return 20; }
    void Serialize(TagBuffer b) const override
    { b.WriteU32(sta); b.WriteU32(ac); b.WriteU32(bytes); b.WriteU64(created); }
    void Deserialize(TagBuffer b) override
    { sta=b.ReadU32(); ac=b.ReadU32(); bytes=b.ReadU32(); created=b.ReadU64(); }
    void Print(std::ostream& os) const override { os << sta << ':' << ac; }
};
NS_OBJECT_ENSURE_REGISTERED(MeasurementTag);

struct Params { uint32_t aifs, cwmin, cwmax; double txopMs; };
struct Counters
{
    uint64_t retry=0, failed=0, discarded=0, offered=0, attempts=0, timeout=0, internalCollision=0;
    uint64_t delivered=0, successes=0, offeredPackets=0;
    uint64_t deadlineEligible=0, onTimeDelivered=0;
    double delaySum=0;
    uint64_t overflow=0, expired=0, otherDrop=0;
    Counters& operator+=(const Counters& x)
    {
        retry+=x.retry; failed+=x.failed; discarded+=x.discarded;
        offered+=x.offered; attempts+=x.attempts; timeout+=x.timeout;
        delivered+=x.delivered; successes+=x.successes; offeredPackets+=x.offeredPackets;
        delaySum+=x.delaySum; overflow+=x.overflow; expired+=x.expired; otherDrop+=x.otherDrop;
        internalCollision+=x.internalCollision;
        deadlineEligible+=x.deadlineEligible; onTimeDelivered+=x.onTimeDelivered;
        return *this;
    }
};
struct Candidate { double generated; int64_t txop; };
struct AirFrame { uint32_t sender, receiver; uint64_t uid; double end; bool data; };
struct DeadlineRecord { uint32_t node, ac; bool matured{false}; };

class Experiment
{
  public:
    std::map<std::string,std::string> cfg;
    std::vector<std::array<Params,4>> profiles;
    std::vector<Ptr<WifiNetDevice>> devices; // AP=0, STA i has node index i+1
    std::vector<std::array<Counters,4>> counters;
    // Accounting schema 2: exact per-STEP totals, separate from sampled state.
    std::vector<std::array<Counters,4>> sampledTotals, lastStepTotals;
    std::vector<std::array<uint64_t,4>> queuedData, lastStepPending;
    bool samplePending=false;
    std::vector<std::array<Ptr<ExponentialRandomVariable>,4>> arrivals;
    std::vector<std::array<Ptr<LogNormalRandomVariable>,4>> lengths;
    std::vector<std::array<double,4>> baseRates;
    std::vector<double> lastReportGeneration;
    std::vector<int64_t> lastReportTxop;
    std::unordered_map<uint64_t,Candidate> candidates;
    std::unordered_map<uint64_t,DeadlineRecord> deadlines;
    std::vector<AirFrame> air;
    std::set<std::pair<uint32_t,uint64_t>> overlap, countedCollision;
    std::ofstream trace;
    uint32_t n=0, enhanced=0, profile=0;
    bool started=false;
    double warmup=0, sample=0, reportMin=0;
    const std::array<AcIndex,4> acs{AC_VO,AC_VI,AC_BE,AC_BK};
    const std::array<uint8_t,4> tids{6,4,0,1};

    double Num(const std::string& key) const { return std::stod(cfg.at(key)); }
    double Now() const { return Simulator::Now().GetSeconds()-warmup; }

    void Load(const std::string& path)
    {
        std::ifstream f(path);
        NS_ABORT_MSG_IF(!f,"Cannot read configuration");
        std::string line;
        while(std::getline(f,line))
        {
            if(line.empty()) continue;
            auto split=line.find('=');
            NS_ABORT_MSG_IF(split==std::string::npos,"Malformed configuration");
            cfg[line.substr(0,split)]=line.substr(split+1);
        }
        std::ifstream p(cfg.at("profiles"));
        std::array<Params,4> row;
        while(p >> row[0].aifs >> row[0].cwmin >> row[0].cwmax >> row[0].txopMs)
        {
            for(int c=1;c<4;++c) p >> row[c].aifs >> row[c].cwmin >> row[c].cwmax >> row[c].txopMs;
            NS_ABORT_MSG_IF(!p,"Incomplete profile");
            profiles.push_back(row);
        }
        NS_ABORT_MSG_IF(profiles.empty(),"No profiles");
        n=Num("num_sta"); enhanced=std::lround(Num("enhanced_fraction")*n);
        warmup=Num("warmup_s"); sample=Num("sample_s"); reportMin=Num("report_min_s");
        if(cfg.at("trace")!="none")
        {
            trace.open(cfg.at("trace"));
            trace << "time_s,node,event,ac,value,uid\n" << std::setprecision(12);
        }
    }

    void Trace(uint32_t node, const std::string& event, int ac, double value, uint64_t uid=0)
    {
        if(started && trace) trace << Now() << ',' << node << ',' << event << ',' << ac << ',' << value << ',' << uid << '\n';
    }

    static bool Receive(Ptr<NetDevice> device, Ptr<const Packet> p, uint16_t protocol, const Address& source)
    { return true; }

    void Ack(uint32_t node, Ptr<const WifiMpdu> mpdu)
    {
        MeasurementTag tag;
        if(!started || !mpdu->GetPacket()->PeekPacketTag(tag)) return;
        auto& count=counters[node][tag.ac];
        const double delay=(Simulator::Now().GetNanoSeconds()-tag.created)/1e6;
        count.delaySum+=delay; ++count.successes;
        // Delivered payload measured at MAC successful completion; no IP overhead included.
        count.delivered+=tag.bytes;
        std::cout << "D " << Now() << ' ' << tag.sta << ' ' << tag.ac << ' ' << tag.bytes << ' ' << delay << ' ' << node << '\n';
        Trace(node,"ACKED",tag.ac,delay,mpdu->GetPacket()->GetUid());
        ResolveDeadline(mpdu->GetPacket()->GetUid(),true);
        auto it=candidates.find(mpdu->GetPacket()->GetUid());
        if(it!=candidates.end())
        {
            std::cout << "R " << Now() << ' ' << node-1 << ' ' << it->second.generated << ' ' << it->second.txop << '\n';
            candidates.erase(it);
        }
    }

    void Drop(uint32_t node, WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
    {
        MeasurementTag tag;
        if(!started || !mpdu->GetPacket()->PeekPacketTag(tag)) return;
        auto& c=counters[node][tag.ac];
        if(reason==WIFI_MAC_DROP_REACHED_RETRY_LIMIT) ++c.failed;
        else
        {
            ++c.discarded;
            if(reason==WIFI_MAC_DROP_FAILED_ENQUEUE) ++c.overflow;
            else if(reason==WIFI_MAC_DROP_EXPIRED_LIFETIME) ++c.expired;
            else ++c.otherDrop;
        }
        auto pending=candidates.find(mpdu->GetPacket()->GetUid());
        if(pending!=candidates.end())
        {
            std::cout << "X " << Now() << ' ' << node-1 << ' ' << pending->second.txop << '\n';
            candidates.erase(pending);
        }
        Trace(node,"DROP",tag.ac,reason,mpdu->GetPacket()->GetUid());
        ResolveDeadline(mpdu->GetPacket()->GetUid(),false);
    }

    void ResolveDeadline(uint64_t uid, bool acked)
    {
        auto it=deadlines.find(uid);
        if(it==deadlines.end()) return;
        if(!it->second.matured)
        {
            auto& c=counters[it->second.node][it->second.ac];
            ++c.deadlineEligible;
            if(acked) ++c.onTimeDelivered;
        }
        deadlines.erase(it);
    }

    void DeadlineReached(uint64_t uid)
    {
        auto it=deadlines.find(uid);
        if(it==deadlines.end() || it->second.matured) return;
        ++counters[it->second.node][it->second.ac].deadlineEligible;
        it->second.matured=true;
    }

    void Timeout(uint32_t node,uint8_t reason,Ptr<const WifiMpdu> mpdu,const WifiTxVector& vector)
    {
        MeasurementTag tag;
        if(!started || !mpdu->GetPacket()->PeekPacketTag(tag)) return;
        ++counters[node][tag.ac].timeout;
        Trace(node,"TIMEOUT",tag.ac,reason,mpdu->GetPacket()->GetUid());
    }

    void Tx(uint32_t node, Ptr<const Packet> p, double power)
    {
        if(!started) return;
        WifiMacHeader hdr;
        p->PeekHeader(hdr);
        MeasurementTag tag;
        std::string name=hdr.IsRts()?"RTS":hdr.IsCts()?"CTS":hdr.IsAck()?"ACK":hdr.IsData()?"DATA":"MGMT";
        Trace(node,name,-1,hdr.IsRetry(),p->GetUid());
        if(!hdr.IsQosData() || !p->PeekPacketTag(tag)) return;
        auto& c=counters[node][tag.ac];
        ++c.attempts;
        if(hdr.IsRetry()) ++c.retry;
        auto txop=devices[node]->GetMac()->GetQosTxop(acs[tag.ac])->GetTxopStartTime(0);
        int64_t txopId=txop?txop->GetNanoSeconds():Simulator::Now().GetNanoSeconds();
        if(node>0 && node<=enhanced && Now()-lastReportGeneration[node]>=reportMin &&
           txopId!=lastReportTxop[node] && !candidates.count(p->GetUid()))
        {
            candidates[p->GetUid()]={Now(),txopId};
            std::cout << "G " << Now() << ' ' << node-1 << ' ' << txopId << '\n';
            lastReportGeneration[node]=Now(); lastReportTxop[node]=txopId;
        }
    }

    bool Reaches(uint32_t sender,uint32_t receiver) const
    {
        if(sender==receiver) return true;
        return !(Num("hidden")!=0 && sender>0 && receiver>0 && ((sender<=n/2)!=(receiver<=n/2)));
    }

    void AirTx(uint32_t node,WifiConstPsduMap psdus,WifiTxVector vector,double power)
    {
        if(!started) return;
        auto psdu=psdus.begin()->second;
        const auto& hdr=psdu->GetHeader(0);
        uint32_t receiver=n+1;
        for(uint32_t i=0;i<=n;++i)
            if(devices[i]->GetAddress()==hdr.GetAddr1()) receiver=i;
        auto p=psdu->GetPacket();
        uint64_t uid=psdu->GetPayload(0)->GetUid();
        overlap.erase({receiver,uid}); countedCollision.erase({receiver,uid});
        double end=Now()+WifiPhy::CalculateTxDuration(psdus,vector,WIFI_PHY_BAND_5GHZ).GetSeconds();
        air.erase(std::remove_if(air.begin(),air.end(),[&](const AirFrame& f){return f.end<=Now();}),air.end());
        for(const auto& f:air)
        {
            if(f.receiver<=n && Reaches(node,f.receiver)) overlap.insert({f.receiver,f.uid});
            if(receiver<=n && Reaches(f.sender,receiver)) overlap.insert({receiver,uid});
        }
        const bool data=hdr.IsQosData() || hdr.IsRts();
        air.push_back({node,receiver,uid,end,data});
        Simulator::Schedule(Seconds(end-Now())+NanoSeconds(1),[this,receiver,uid](){
            overlap.erase({receiver,uid}); countedCollision.erase({receiver,uid});
        });
        // A=attempted DATA/RTS at its intended receiver, not ACK/control overhead.
        if(data && receiver<=n) std::cout << "A " << Now() << ' ' << node << ' ' << receiver << ' ' << uid << '\n';
    }

    void RxDrop(uint32_t node,Ptr<const Packet> packet,WifiPhyRxfailureReason reason)
    {
        if(!started) return;
        WifiMacHeader hdr; packet->PeekHeader(hdr);
        if(!(hdr.IsQosData() || hdr.IsRts()) || hdr.GetAddr1()!=devices[node]->GetAddress()) return;
        auto key=std::make_pair(node,packet->GetUid());
        bool interfered=overlap.count(key)>0;
        MeasurementTag tag; const bool tagged=packet->PeekPacketTag(tag);
        // Emit only actual intended-receiver failures with independently observed overlapping energy.
        if(interfered && countedCollision.insert(key).second)
            std::cout << "C " << Now() << ' ' << node << ' ' << packet->GetUid() << ' ' << static_cast<int>(reason)
                      << ' ' << (tagged?static_cast<int>(tag.ac):-1) << ' ' << (tagged?static_cast<int>(tag.sta):-1) << '\n';
        std::cout << "F " << Now() << ' ' << node << ' ' << interfered << ' ' << static_cast<int>(reason)
                  << ' ' << (tagged?static_cast<int>(tag.ac):-1) << ' ' << (tagged?static_cast<int>(tag.sta):-1) << '\n';
        Trace(node,interfered?"COLLISION_RX":"OTHER_RX_FAIL",-1,reason,packet->GetUid());
    }

    void Cw(uint32_t node,uint32_t ac,uint32_t value,uint8_t link)
    { Trace(node,"CW",ac,value); }

    void Txop(uint32_t node,uint32_t ac,Time start,Time duration,uint8_t link)
    { Trace(node,"TXOP",ac,duration.GetMicroSeconds()); }

    void InternalCollision(uint32_t node,uint32_t ac,uint8_t link)
    {
        if(!started) return;
        ++counters[node][ac].internalCollision;
        Trace(node,"INTERNAL_COLLISION",ac,link);
    }

    void State(Time start,Time duration,WifiPhyState state)
    {
        if(!started || state==WifiPhyState::IDLE || duration.IsZero()) return;
        // The PHY may report an interval ending or starting at the current time.
        std::cout << "B " << start.GetSeconds()-warmup << ' ' << duration.GetSeconds() << ' ' << static_cast<int>(state) << '\n';
    }

    void Sample()
    {
        samplePending=true;
        for(uint32_t node=0;node<=n;++node)
            for(uint32_t ac=0;ac<4;++ac)
                devices[node]->GetMac()->GetQosTxop(acs[ac])->GetWifiMacQueue()->WipeAllExpiredMpdus();
        // Expired callbacks are ScheduleNow events; include them in this sample.
        Simulator::ScheduleNow(&Experiment::EmitSample,this);
        Simulator::Schedule(Seconds(sample),&Experiment::Sample,this);
    }

    void EmitSample()
    {
        for(uint32_t node=0;node<=n;++node)
        {
            for(uint32_t ac=0;ac<4;++ac)
            {
                auto& c=counters[node][ac];
                auto queue=devices[node]->GetMac()->GetQosTxop(acs[ac])->GetWifiMacQueue();
                Trace(node,"QUEUE_PACKETS",ac,queuedData[node][ac]);
                if(Now()<sample+1e-6)
                {
                    auto q=devices[node]->GetMac()->GetQosTxop(acs[ac]);
                    Trace(node,"QUEUE_LIMIT",ac,queue->GetMaxSize().GetValue());
                    Trace(node,"QUEUE_TTL",ac,queue->GetMaxDelay().GetSeconds());
                    Trace(node,"ACTUAL_CWMIN",ac,q->GetMinCw(0));
                    Trace(node,"ACTUAL_CWMAX",ac,q->GetMaxCw(0));
                    Trace(node,"ACTUAL_AIFSN",ac,q->GetAifsn(0));
                    Trace(node,"ACTUAL_TXOP_US",ac,q->GetTxopLimit().GetMicroSeconds());
                }
                std::cout << "S " << Now() << ' ' << node << ' ' << ac << ' ' << queue->GetNBytes()
                    << ' ' << (c.successes?c.delaySum/c.successes:0) << ' ' << c.retry << ' ' << c.failed
                    << ' ' << c.discarded << ' ' << (c.successes>0) << ' ' << c.offered << ' ' << c.attempts
                    << ' ' << c.timeout << ' ' << c.delivered << ' ' << c.offeredPackets << '\n';
                sampledTotals[node][ac]+=c; c=Counters{};
            }
        }
        samplePending=false;
    }

    void Enqueued(uint32_t node, uint32_t ac, Ptr<const WifiMpdu> mpdu)
    {
        MeasurementTag tag;
        if(started && mpdu->GetPacket()->PeekPacketTag(tag)) ++queuedData[node][ac];
    }

    void Dequeued(uint32_t node, uint32_t ac, Ptr<const WifiMpdu> mpdu)
    {
        MeasurementTag tag;
        if(!started || !mpdu->GetPacket()->PeekPacketTag(tag)) return;
        NS_ABORT_MSG_IF(queuedData[node][ac]==0,"Unmatched tagged packet dequeue");
        --queuedData[node][ac];
    }

    Counters Total(uint32_t node, uint32_t ac) const
    {
        auto total=sampledTotals[node][ac]; total+=counters[node][ac]; return total;
    }

    void FinishStep(uint32_t round)
    {
        bool balanced=!samplePending;
        for(uint32_t node=0;node<=n;++node)
            for(uint32_t ac=0;ac<4;++ac)
            {
                auto c=Total(node,ac);
                balanced &= c.offeredPackets==c.successes+c.failed+c.discarded+queuedData[node][ac];
            }
        // A queue removal may precede its ScheduleNow Expired callback. Drain
        // only current-time callbacks; never add time or manufacture a loss.
        if(!balanced)
        {
            NS_ABORT_MSG_IF(round>=64,"Packet accounting failed at STEP boundary");
            Simulator::ScheduleNow(&Experiment::FinishStep,this,round+1);
            return;
        }
        for(uint32_t node=0;node<=n;++node)
            for(uint32_t ac=0;ac<4;++ac)
            {
                auto c=Total(node,ac); const auto& p=lastStepTotals[node][ac];
                // P: time,node,AC,offered,ACKed,overflow,expired,retry-drop,
                // other-drop,pending-start,pending-end,offered-bytes,ACKed-bytes,
                // retries,attempts,timeouts,internal-collisions,deadline-eligible,
                // on-time-ACKed. Counts are deltas for this STEP.
                std::cout << "P " << Now() << ' ' << node << ' ' << ac
                    << ' ' << c.offeredPackets-p.offeredPackets << ' ' << c.successes-p.successes
                    << ' ' << c.overflow-p.overflow << ' ' << c.expired-p.expired
                    << ' ' << c.failed-p.failed << ' ' << c.otherDrop-p.otherDrop
                    << ' ' << lastStepPending[node][ac] << ' ' << queuedData[node][ac]
                    << ' ' << c.offered-p.offered << ' ' << c.delivered-p.delivered
                    << ' ' << c.retry-p.retry << ' ' << c.attempts-p.attempts << ' ' << c.timeout-p.timeout
                    << ' ' << c.internalCollision-p.internalCollision
                    << ' ' << c.deadlineEligible-p.deadlineEligible
                    << ' ' << c.onTimeDelivered-p.onTimeDelivered << '\n';
                lastStepTotals[node][ac]=c; lastStepPending[node][ac]=queuedData[node][ac];
            }
        Simulator::Stop();
    }

    double LoadScale(uint32_t sta) const
    {
        if(Num("dynamic") == 0) return 1.0;
        const double period=Num("dynamic_period_s");
        const double offset=Num("dynamic_stagger") != 0 ? period*sta/std::max(1u,n) : 0.0;
        const double phase=std::fmod(std::max(0.0,Now())+offset,period);
        return phase < period/2 ? Num("dynamic_low_scale") : Num("dynamic_high_scale");
    }

    double NextInterval(uint32_t node,uint32_t sta,uint32_t ac)
    {
        const double rate=baseRates[node][ac]*LoadScale(sta);
        NS_ABORT_MSG_IF(rate<=0,"Traffic state must retain a positive rate");
        arrivals[node][ac]->SetAttribute("Mean",DoubleValue(1/rate));
        return arrivals[node][ac]->GetValue();
    }

    void Send(uint32_t node,uint32_t sta,uint32_t ac)
    {
        uint32_t bytes=Num("payload"+std::to_string(ac));
        if(ac==1 && Num("video_variable")!=0)
            bytes=std::clamp(lengths[node][ac]->GetValue(),Num("video_min"),Num("video_max"));
        auto packet=Create<Packet>(bytes);
        MeasurementTag tag; tag.sta=sta; tag.ac=ac; tag.bytes=bytes;
        tag.created=Simulator::Now().GetNanoSeconds(); packet->AddPacketTag(tag);
        SocketPriorityTag priority; priority.SetPriority(tids[ac]); packet->AddPacketTag(priority);
        counters[node][ac].offered+=bytes;
        ++counters[node][ac].offeredPackets;
        const double deadlineMs=Num("deadline"+std::to_string(ac)+"_ms");
        if(deadlineMs>0)
        {
            deadlines[packet->GetUid()]={node,ac,false};
            Simulator::Schedule(MilliSeconds(deadlineMs),&Experiment::DeadlineReached,this,packet->GetUid());
        }
        Trace(node,"OFFERED",ac,bytes,packet->GetUid());
        auto dest=devices[node==0?sta+1:0]->GetAddress();
        devices[node]->Send(packet,dest,0x88b5);
        // Independent exponential processes per STA/direction/AC; random streams are assigned explicitly.
        Simulator::Schedule(Seconds(NextInterval(node,sta,ac)),&Experiment::Send,this,node,sta,ac);
    }

    void Apply(uint32_t index,bool growth,uint32_t alpha,uint32_t b1,uint32_t b2,uint32_t gamma)
    {
        NS_ABORT_MSG_IF(index>=profiles.size() || b2<b1 || alpha>gamma,"Invalid action");
        profile=index;
        for(uint32_t node=0;node<=n;++node)
            for(uint32_t ac=0;ac<4;++ac)
            {
                const auto& p=profiles[index][ac];
                auto q=devices[node]->GetMac()->GetQosTxop(acs[ac]);
                q->SetMaxCw(p.cwmax); q->SetMinCw(p.cwmin); q->SetAifsn(p.aifs);
                q->SetTxopLimit(MicroSeconds(std::llround(p.txopMs*1000)));
                q->SetResearchCwGrowth(growth && (node==0 || node<=enhanced),alpha,b1,b2,gamma);
            }
    }

    void Setup()
    {
        RngSeedManager::SetSeed(1); RngSeedManager::SetRun(Num("seed"));
        NodeContainer nodes; nodes.Create(n+1);
        MobilityHelper mobility; mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel"); mobility.Install(nodes);
        auto loss=CreateObject<MatrixPropagationLossModel>(); loss->SetDefaultLoss(Num("visible_loss_db"));
        if(Num("hidden")!=0)
            for(uint32_t i=1;i<=n;++i) for(uint32_t j=i+1;j<=n;++j)
                if((i<=n/2)!=(j<=n/2)) loss->SetLoss(nodes.Get(i)->GetObject<MobilityModel>(),nodes.Get(j)->GetObject<MobilityModel>(),Num("hidden_loss_db"));
        auto channel=CreateObject<YansWifiChannel>(); channel->SetPropagationLossModel(loss);
        channel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());
        YansWifiPhyHelper phy; phy.SetChannel(channel);
        phy.Set("TxPowerStart",DoubleValue(Num("tx_power_dbm"))); phy.Set("TxPowerEnd",DoubleValue(Num("tx_power_dbm")));
        phy.Set("ChannelSettings",StringValue("{36, 20, BAND_5GHZ, 0}"));
        WifiHelper wifi; wifi.SetStandard(WIFI_STANDARD_80211a);
        wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager","DataMode",StringValue(cfg.at("data_mode")),
            "ControlMode",StringValue(cfg.at("control_mode")),"RtsCtsThreshold",UintegerValue(Num("rts")?0:65535));
        WifiMacHelper mac; Ssid ssid("research-bss");
        mac.SetType("ns3::ApWifiMac","Ssid",SsidValue(ssid),"QosSupported",BooleanValue(true));
        auto ap=wifi.Install(phy,mac,nodes.Get(0)); devices.push_back(DynamicCast<WifiNetDevice>(ap.Get(0)));
        mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),"QosSupported",BooleanValue(true),"ActiveProbing",BooleanValue(false));
        NetDeviceContainer all(ap);
        for(uint32_t i=1;i<=n;++i)
        {
            auto d=wifi.Install(phy,mac,nodes.Get(i)); all.Add(d); devices.push_back(DynamicCast<WifiNetDevice>(d.Get(0)));
        }
        wifi.AssignStreams(all,100);
        counters.resize(n+1); arrivals.resize(n+1); lengths.resize(n+1); baseRates.resize(n+1);
        sampledTotals.resize(n+1); lastStepTotals.resize(n+1);
        queuedData.resize(n+1); lastStepPending.resize(n+1);
        lastReportGeneration.assign(n+1,-1e30); lastReportTxop.assign(n+1,-1);
        for(uint32_t i=0;i<=n;++i)
        {
            auto dev=devices[i]; auto m=dev->GetMac();
            dev->SetReceiveCallback(MakeCallback(&Experiment::Receive));
            m->SetFrameRetryLimit(Num("retry_limit"));
            m->TraceConnectWithoutContext("AckedMpdu",MakeCallback(&Experiment::Ack,this).Bind(i));
            m->TraceConnectWithoutContext("DroppedMpdu",MakeCallback(&Experiment::Drop,this).Bind(i));
            m->TraceConnectWithoutContext("MpduResponseTimeout",MakeCallback(&Experiment::Timeout,this).Bind(i));
            dev->GetPhy()->TraceConnectWithoutContext("PhyTxBegin",MakeCallback(&Experiment::Tx,this).Bind(i));
            dev->GetPhy()->TraceConnectWithoutContext("PhyTxPsduBegin",MakeCallback(&Experiment::AirTx,this).Bind(i));
            dev->GetPhy()->TraceConnectWithoutContext("PhyRxDrop",MakeCallback(&Experiment::RxDrop,this).Bind(i));
            if(i==0) dev->GetPhy()->GetState()->TraceConnectWithoutContext("State",MakeCallback(&Experiment::State,this));
            for(uint32_t c=0;c<4;++c)
            {
                auto q=m->GetQosTxop(acs[c]);
                q->GetWifiMacQueue()->SetMaxSize(QueueSize(std::to_string(static_cast<int>(Num("queue"+std::to_string(c))))+"p"));
                q->GetWifiMacQueue()->SetMaxDelay(Seconds(Num("queue_delay_s")));
                q->GetWifiMacQueue()->TraceConnectWithoutContext("Enqueue",MakeCallback(&Experiment::Enqueued,this).Bind(i,c));
                q->GetWifiMacQueue()->TraceConnectWithoutContext("Dequeue",MakeCallback(&Experiment::Dequeued,this).Bind(i,c));
                q->TraceConnectWithoutContext("CwTrace",MakeCallback(&Experiment::Cw,this).Bind(i,c));
                q->TraceConnectWithoutContext("TxopTrace",MakeCallback(&Experiment::Txop,this).Bind(i,c));
                q->TraceConnectWithoutContext("InternalCollision",MakeCallback(&Experiment::InternalCollision,this).Bind(i,c));
                arrivals[i][c]=CreateObject<ExponentialRandomVariable>(); arrivals[i][c]->SetStream(10000+i*16+c);
                lengths[i][c]=CreateObject<LogNormalRandomVariable>(); lengths[i][c]->SetStream(20000+i*16+c);
                double sigma=Num("video_sigma");
                lengths[i][c]->SetAttribute("Mu",DoubleValue(std::log(Num("payload1"))-sigma*sigma/2));
                lengths[i][c]->SetAttribute("Sigma",DoubleValue(sigma));
            }
        }
        Apply(0,false,1,20,20,7);
        Simulator::Stop(Seconds(warmup)); Simulator::Run();
        started=true;
        Simulator::Schedule(Seconds(sample),&Experiment::Sample,this);
        for(uint32_t sta=0;sta<n;++sta) for(uint32_t c=0;c<4;++c)
        {
            double rate=Num("offered_mbps")*1e6*Num("mix"+std::to_string(c))/(8*Num("mean"+std::to_string(c))*n);
            double ul=Num("ul_fraction");
            if(rate*ul>0)
            {
                baseRates[sta+1][c]=rate*ul;
                Simulator::Schedule(Seconds(NextInterval(sta+1,sta,c)),&Experiment::Send,this,sta+1,sta,c);
            }
            if(rate*(1-ul)>0)
            {
                // Superposition of equal independent destination flows at AP.
                baseRates[0][c]=rate*(1-ul);
                Simulator::Schedule(Seconds(NextInterval(0,sta,c)),&Experiment::Send,this,0,sta,c);
            }
        }
    }
};

int main(int argc,char** argv)
{
    std::string config; CommandLine cmd; cmd.AddValue("config","Flat backend configuration",config); cmd.Parse(argc,argv);
    NS_ABORT_MSG_IF(config.empty(),"--config required");
    std::cout << std::setprecision(12);
    Experiment experiment; experiment.Load(config); experiment.Setup();
    std::cout << "READY 4\n" << std::flush;
    std::string command;
    while(std::getline(std::cin,command))
    {
        if(command=="CLOSE") break;
        std::istringstream in(command); std::string verb;
        double duration; uint32_t profile,alpha,b1,b2,gamma; bool custom;
        in >> verb >> duration >> profile >> custom >> alpha >> b1 >> b2 >> gamma;
        NS_ABORT_MSG_IF(!in || verb!="STEP" || duration<=0,"Malformed STEP command");
        experiment.Apply(profile,custom,alpha,b1,b2,gamma);
        // The barrier verifies all tagged packets before returning at exactly
        // the requested time, including deferred final-sample expiry callbacks.
        Simulator::Schedule(Seconds(duration), [&experiment](){
            Simulator::ScheduleNow(&Experiment::FinishStep,&experiment,0); });
        Simulator::Run();
        std::cout << "END " << experiment.Now() << '\n' << std::flush;
    }
    Simulator::Destroy();
}
