// IEEE TGax Scenario 3 (Indoor Small BSSs), Default EDCA environment.
// Research CW-growth hooks are deliberately not enabled in this program.

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TgaxScenario3");

namespace
{

constexpr double PI = 3.141592653589793238462643383279502884;

std::ofstream g_appEvents;
std::ofstream g_macEvents;
std::ofstream g_backoffEvents;
std::ofstream g_phyStates;
std::ofstream g_queueSamples;
std::ofstream g_macPacketEvents;
std::ofstream g_macAccessDelay;
bool g_fullTrace{true};

// Packet tag used by direct-MAC sources.  It lets the existing MAC traces
// classify an MSDU without requiring an IPv4 header and keeps effective
// payload bytes separate from the LLC header added below.
class MacFullBufferTag : public Tag
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::MacFullBufferTag")
                                .SetParent<Tag>()
                                .SetGroupName("Wifi")
                                .AddConstructor<MacFullBufferTag>();
        return tid;
    }
    TypeId GetInstanceTypeId() const override { return GetTypeId(); }
    uint32_t GetSerializedSize() const override { return 5; }
    void Serialize(TagBuffer buffer) const override
    {
        buffer.WriteU8(m_ac);
        buffer.WriteU32(m_payloadBytes);
    }
    void Deserialize(TagBuffer buffer) override
    {
        m_ac = buffer.ReadU8();
        m_payloadBytes = buffer.ReadU32();
    }
    void Print(std::ostream& stream) const override
    {
        stream << "ac=" << +m_ac << ",payload=" << m_payloadBytes;
    }
    void Set(uint8_t ac, uint32_t payloadBytes)
    {
        m_ac = ac;
        m_payloadBytes = payloadBytes;
    }
    uint8_t GetAc() const { return m_ac; }
    uint32_t GetPayloadBytes() const { return m_payloadBytes; }

  private:
    uint8_t m_ac{};
    uint32_t m_payloadBytes{};
};

NS_OBJECT_ENSURE_REGISTERED(MacFullBufferTag);

struct MacPending
{
    double arrival{};
    uint32_t bss{};
    std::string direction;
    std::string ac;
    uint32_t bytes{};
    uint32_t txAttempts{};
};

struct MacOutcome
{
    uint64_t arrived{};
    uint64_t acked{};
    uint64_t failed{};
    uint64_t unfinished{};
    uint64_t failedEnqueue{};
    uint64_t expiredLifetime{};
    uint64_t retryLimit{};
    uint64_t qosOldPacket{};
    uint64_t otherFailure{};
    uint64_t arrivedBytes{};
    uint64_t ackedBytes{};
    uint64_t failedBytes{};
    uint64_t unfinishedBytes{};
    uint64_t txAttempts{};
    uint64_t retransmissions{};
    uint64_t deadlineEligible{};
    uint64_t onTimeDelivered{};
    std::vector<double> serviceDelayMs;
};

using OutcomeKey = std::tuple<uint32_t, std::string, std::string>; // BSS, UL/DL, AC
struct DeadlineStatus
{
    OutcomeKey key;
    bool matured{};
};
std::map<std::pair<std::string, uint64_t>, MacPending> g_macPending;
std::map<std::pair<std::string, uint64_t>, double> g_accessPending;
std::map<OutcomeKey, MacOutcome> g_macOutcomes;
std::map<std::pair<std::string, uint64_t>, DeadlineStatus> g_deadlines;
uint64_t g_duplicateAdmissions{};
std::set<std::pair<std::string, uint64_t>> g_seenTcpAckMpdus;
std::map<OutcomeKey, uint64_t> g_tcpAckMpdus;
using NodeAcKey = std::pair<std::string, std::string>;
struct FullBufferSourceStats
{
    uint64_t arrived{}, acked{}, failed{}, unfinished{}, ackedBytes{};
};
struct FullBufferQueueStats
{
    uint64_t samples{}, nonempty{}, totalPackets{}, minimum{std::numeric_limits<uint64_t>::max()}, maximum{};
};
struct AmpduStats
{
    uint64_t psdus{}, ampduPsdus{}, mpdus{}, maximumMpdus{};
};
std::set<NodeAcKey> g_fullBufferSources;
std::map<NodeAcKey, FullBufferSourceStats> g_fullBufferSourceStats;
std::map<NodeAcKey, FullBufferQueueStats> g_fullBufferQueueStats;
std::map<std::string, AmpduStats> g_ampduStats;
std::map<std::string, std::set<uint32_t>> g_cwValues;
std::ofstream g_competitionSamples;
double g_measureStart{0.0};
double g_measureEnd{0.0};

std::vector<std::string>
Split(const std::string& line)
{
    std::vector<std::string> values;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        values.push_back(item);
    }
    if (!line.empty() && line.back() == ',')
    {
        values.emplace_back();
    }
    return values;
}

std::vector<std::map<std::string, std::string>>
ReadCsv(const std::string& path)
{
    std::ifstream stream(path);
    NS_ABORT_MSG_IF(!stream, "Cannot open CSV: " << path);
    std::string line;
    NS_ABORT_MSG_IF(!std::getline(stream, line), "Empty CSV: " << path);
    const auto header = Split(line);
    std::vector<std::map<std::string, std::string>> rows;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto fields = Split(line);
        NS_ABORT_MSG_IF(fields.size() != header.size(), "Malformed CSV row in " << path);
        std::map<std::string, std::string> row;
        for (std::size_t i = 0; i < header.size(); ++i)
        {
            row.emplace(header[i], fields[i]);
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

std::string
AcName(AcIndex ac)
{
    switch (ac)
    {
    case AC_VO:
        return "VO";
    case AC_VI:
        return "VI";
    case AC_BE:
        return "BE";
    case AC_BK:
        return "BK";
    default:
        return "NA";
    }
}

uint8_t
TosForAc(const std::string& ac)
{
    if (ac == "VO")
    {
        return 0xc0; // user priority 6 -> AC_VO
    }
    if (ac == "VI")
    {
        return 0xb8; // user priority 5 -> AC_VI
    }
    if (ac == "BK")
    {
        return 0x20; // user priority 1 -> AC_BK
    }
    return 0x00; // AC_BE
}

uint64_t
SplitMix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

class TgaxIndoorPropagationLossModel : public PropagationLossModel
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TgaxIndoorPropagationLossModel")
                                .SetParent<PropagationLossModel>()
                                .SetGroupName("Propagation")
                                .AddConstructor<TgaxIndoorPropagationLossModel>();
        return tid;
    }

    void SetSeed(uint64_t seed)
    {
        m_seed = seed;
    }

    double GetLoss(Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
    {
        const auto pa = a->GetPosition();
        const auto pb = b->GetPosition();
        const double dx = pa.x - pb.x;
        const double dy = pa.y - pb.y;
        const double dz = pa.z - pb.z;
        const double d = std::max(1.0, std::sqrt(dx * dx + dy * dy + dz * dz));
        const double loss = 40.05 + 20.0 * std::log10(5.0 / 2.4) +
                            20.0 * std::log10(std::min(d, 10.0)) +
                            (d > 10.0 ? 35.0 * std::log10(d / 10.0) : 0.0);
        return loss + Shadow(pa, pb);
    }

  private:
    double DoCalcRxPower(double txPowerDbm,
                         Ptr<MobilityModel> a,
                         Ptr<MobilityModel> b) const override
    {
        return txPowerDbm - GetLoss(a, b);
    }

    int64_t DoAssignStreams(int64_t) override
    {
        return 0;
    }

    double Shadow(Vector a, Vector b) const
    {
        std::array<int64_t, 3> left{std::llround(a.x * 1000),
                                    std::llround(a.y * 1000),
                                    std::llround(a.z * 1000)};
        std::array<int64_t, 3> right{std::llround(b.x * 1000),
                                     std::llround(b.y * 1000),
                                     std::llround(b.z * 1000)};
        if (right < left)
        {
            std::swap(left, right);
        }
        uint64_t hash = 0xcbf29ce484222325ULL ^ m_seed;
        for (const auto coordinate : {left[0], left[1], left[2], right[0], right[1], right[2]})
        {
            const uint64_t bits = static_cast<uint64_t>(coordinate);
            for (unsigned shift = 0; shift < 64; shift += 8)
            {
                hash ^= static_cast<uint8_t>((bits >> shift) & 0xff);
                hash *= 0x100000001b3ULL;
            }
        }
        const uint64_t x1 = SplitMix64(hash);
        const uint64_t x2 = SplitMix64(hash ^ 0xd1b54a32d192ed03ULL);
        const double u1 = (static_cast<long double>(x1) + 0.5L) / 18446744073709551616.0L;
        const double u2 = (static_cast<long double>(x2) + 0.5L) / 18446744073709551616.0L;
        return 5.0 * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
    }

    uint64_t m_seed{1};
};

NS_OBJECT_ENSURE_REGISTERED(TgaxIndoorPropagationLossModel);

struct FlowRow
{
    uint32_t id{};
    uint32_t streamId{};
    uint32_t bss{};
    std::string service;
    std::string model;
    std::string direction;
    std::string transport;
    std::string ac;
    uint32_t sta{};
    std::string source;
    std::string sink;
};

class TgaxSourceApplication : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::TgaxSourceApplication")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<TgaxSourceApplication>();
        return tid;
    }

    void Setup(const FlowRow& row, Ipv4Address peer, uint16_t port, int64_t stream)
    {
        m_row = row;
        m_peer = peer;
        m_port = port;
        m_uniform = CreateObject<UniformRandomVariable>();
        m_exponential = CreateObject<ExponentialRandomVariable>();
        m_weibull = CreateObject<WeibullRandomVariable>();
        m_gamma = CreateObject<GammaRandomVariable>();
        m_extreme = CreateObject<LargestExtremeValueRandomVariable>();
        m_lognormal = CreateObject<LogNormalRandomVariable>();
        m_pareto = CreateObject<ParetoRandomVariable>();
        m_uniform->SetStream(stream);
        m_exponential->SetStream(stream + 1);
        m_weibull->SetStream(stream + 2);
        m_gamma->SetStream(stream + 3);
        m_extreme->SetStream(stream + 4);
        m_lognormal->SetStream(stream + 5);
        m_pareto->SetStream(stream + 6);
        if (m_row.model == "FTP")
        {
            m_segmentSize = m_uniform->GetValue() < 0.76 ? 1460 : 536;
        }
        else if (m_row.model == "HTTP_REQ")
        {
            m_segmentSize = 350;
        }
        m_httpPersistent = m_uniform->GetValue() < 0.5;
    }

    uint64_t GetOfferedBytes() const
    {
        return m_offeredBytes;
    }

  private:
    void StartApplication() override
    {
        m_running = true;
        // Generate application demand independently of TCP connection progress.
        // Bytes generated before Connected() stay pending and are sent once the
        // connection is established, preserving the traffic-model clock.
        if (m_row.transport == "TCP")
        {
            m_started = true;
            m_event = Simulator::Schedule(MilliSeconds(m_uniform->GetValue(0.0, 20.0)),
                                          &TgaxSourceApplication::StartModel,
                                          this);
        }
        OpenSocket();
    }

    void StopApplication() override
    {
        m_running = false;
        Simulator::Cancel(m_event);
        if (m_socket)
        {
            m_socket->Close();
        }
    }

    void OpenSocket()
    {
        const bool tcp = m_row.transport == "TCP";
        m_socket = Socket::CreateSocket(GetNode(),
                                        tcp ? TcpSocketFactory::GetTypeId()
                                            : UdpSocketFactory::GetTypeId());
        m_socket->SetIpTos(TosForAc(m_row.ac));
        if (tcp)
        {
            m_socket->SetAttribute("SegmentSize", UintegerValue(m_segmentSize));
        }
        if (tcp && m_row.model == "FTP")
        {
            m_socket->SetAttribute("InitialCwnd", UintegerValue(1));
        }
        m_socket->SetConnectCallback(MakeCallback(&TgaxSourceApplication::Connected, this),
                                     MakeCallback(&TgaxSourceApplication::ConnectFailed, this));
        m_socket->SetSendCallback(MakeCallback(&TgaxSourceApplication::CanSend, this));
        const int status = m_socket->Connect(InetSocketAddress(m_peer, m_port));
        if (!tcp && status == 0)
        {
            Connected(m_socket);
        }
    }

    void Connected(Ptr<Socket>)
    {
        if (!m_running)
        {
            return;
        }
        if (m_row.transport == "TCP")
        {
            Drain();
            return;
        }
        if (m_started)
        {
            return;
        }
        m_started = true;
        m_event = Simulator::Schedule(MilliSeconds(m_uniform->GetValue(0.0, 20.0)),
                                      &TgaxSourceApplication::StartModel,
                                      this);
    }

    void ConnectFailed(Ptr<Socket>)
    {
        NS_FATAL_ERROR("TCP/UDP connect failed for TGax flow " << m_row.id);
    }

    void CanSend(Ptr<Socket>, uint32_t)
    {
        Drain();
    }

    void StartModel()
    {
        if (m_row.model == "VOIP")
        {
            m_voipActive = m_uniform->GetValue() < 0.5;
            m_stateEnd = Simulator::Now() + Seconds(m_exponential->GetValue(1.25, 0.0));
            VoipTick();
        }
        else if (m_row.model == "GMG")
        {
            GamingTick();
        }
        else if (m_row.model == "BV3" || m_row.model == "BV6")
        {
            VideoTick();
        }
        else if (m_row.model == "FTP")
        {
            FtpSession();
        }
        else if (m_row.model == "HTTP")
        {
            HttpSession();
        }
        else if (m_row.model == "HTTP_REQ")
        {
            HttpRequest();
        }
    }

    void LogGeneration(uint64_t bytes, const std::string& detail)
    {
        if (!g_fullTrace)
        {
            return;
        }
        g_appEvents << std::fixed << std::setprecision(9) << Simulator::Now().GetSeconds()
                    << ",GENERATE," << m_row.id << ',' << m_row.bss << ',' << m_row.service
                    << ',' << m_row.model << ',' << m_row.direction << ',' << m_row.transport
                    << ',' << m_row.ac << ',' << bytes << ',' << detail << '\n';
    }

    void Generate(uint64_t bytes, const std::string& detail)
    {
        if (!m_running)
        {
            return;
        }
        m_offeredBytes += bytes;
        LogGeneration(bytes, detail);
        if (m_row.transport == "UDP")
        {
            const int sent = m_socket->Send(Create<Packet>(static_cast<uint32_t>(bytes)));
            NS_ABORT_MSG_IF(sent != static_cast<int>(bytes), "UDP TGax send failed");
            return;
        }
        m_pendingBytes += bytes;
        Drain();
    }

    void Drain()
    {
        if (!m_running || m_row.transport != "TCP" || !m_socket)
        {
            return;
        }
        while (m_pendingBytes > 0)
        {
            const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(m_pendingBytes, m_segmentSize));
            const int sent = m_socket->Send(Create<Packet>(chunk));
            if (sent <= 0)
            {
                return;
            }
            m_pendingBytes -= static_cast<uint32_t>(sent);
        }
        if (m_sessionPending)
        {
            m_sessionPending = false;
            const double wait = m_row.model == "FTP" ? m_exponential->GetValue(180.0, 0.0)
                                                       : m_exponential->GetValue(30.0, 0.0);
            m_event = Simulator::Schedule(Seconds(wait),
                                          m_row.model == "FTP" ? &TgaxSourceApplication::FtpSession
                                                                : &TgaxSourceApplication::HttpSession,
                                          this);
        }
    }

    void VoipTick()
    {
        if (!m_running)
        {
            return;
        }
        if (Simulator::Now() >= m_stateEnd)
        {
            m_voipActive = !m_voipActive;
            m_stateEnd = Simulator::Now() + Seconds(m_exponential->GetValue(1.25, 0.0));
            if (g_fullTrace)
            {
                g_appEvents << Simulator::Now().GetSeconds() << ",STATE," << m_row.id << ','
                            << m_row.bss << ',' << m_row.service << ',' << m_row.model << ','
                            << m_row.direction << ',' << m_row.transport << ',' << m_row.ac
                            << ",0," << (m_voipActive ? "ACTIVE" : "SILENCE") << '\n';
            }
        }
        Generate(m_voipActive ? 33 : 7, m_voipActive ? "ACTIVE" : "SILENCE");
        double intervalMs = m_voipActive ? 20.0 : 160.0;
        if (m_row.direction == "DL")
        {
            const double u = m_uniform->GetValue(-0.5, 0.5);
            const double jitter = -5.11 * std::copysign(std::log(1.0 - 2.0 * std::abs(u)), u);
            intervalMs = std::max(0.001, intervalMs + jitter);
        }
        m_event = Simulator::Schedule(MilliSeconds(intervalMs),
                                      &TgaxSourceApplication::VoipTick,
                                      this);
    }

    void GamingTick()
    {
        if (!m_running)
        {
            return;
        }
        const bool dl = m_row.direction == "DL";
        const double size = m_extreme->GetValue(dl ? 390.0 : 158.0, dl ? 89.0 : 26.2);
        Generate(static_cast<uint64_t>(std::max(1.0, std::round(size))), "GMG_PACKET");
        const double iat = m_extreme->GetValue(dl ? 15.0 : 23.5, dl ? 7.0 : 10.5);
        m_event = Simulator::Schedule(MilliSeconds(std::max(0.001, iat)),
                                      &TgaxSourceApplication::GamingTick,
                                      this);
    }

    void VideoTick()
    {
        if (!m_running)
        {
            return;
        }
        const bool bv6 = m_row.model == "BV6";
        const double lambda = bv6 ? 54210.0 : 20850.0;
        const double bitrate = bv6 ? 15.6e6 : 6.0e6;
        const double shape = 0.8099;
        const uint64_t bytes = static_cast<uint64_t>(std::max(1.0, std::round(m_weibull->GetValue(lambda, shape, 0.0))));
        const double delayMs = m_gamma->GetValue(0.2463, 60.227);
        Simulator::Schedule(MilliSeconds(delayMs),
                            &TgaxSourceApplication::Generate,
                            this,
                            bytes,
                            "VIDEO_FRAME");
        const double meanBytes = lambda * std::tgamma(1.0 + 1.0 / shape);
        const double interval = meanBytes * 8.0 / bitrate;
        m_event = Simulator::Schedule(Seconds(interval),
                                      &TgaxSourceApplication::VideoTick,
                                      this);
    }

    uint64_t TruncatedLognormal(double mu, double sigma, uint64_t minimum, uint64_t maximum)
    {
        for (unsigned attempt = 0; attempt < 100000; ++attempt)
        {
            const double value = m_lognormal->GetValue(mu, sigma);
            if (value >= minimum && value <= maximum)
            {
                return static_cast<uint64_t>(std::round(value));
            }
        }
        NS_FATAL_ERROR("Truncated lognormal rejection sampler failed");
        return minimum;
    }

    void FtpSession()
    {
        if (!m_running)
        {
            return;
        }
        const uint64_t bytes = TruncatedLognormal(14.45, 0.35, 1, 5 * 1024 * 1024);
        m_sessionPending = true;
        Generate(bytes, "FTP_FILE");
    }

    void HttpSession()
    {
        if (!m_running)
        {
            return;
        }
        uint64_t bytes = TruncatedLognormal(8.37, 1.37, 100, 2 * 1024 * 1024);
        const double raw = m_pareto->GetValue(2.0, 1.1, 55.0);
        const uint32_t objects = std::min<uint32_t>(53, static_cast<uint32_t>(std::max(0.0, std::floor(raw - 2.0))));
        for (uint32_t i = 0; i < objects; ++i)
        {
            bytes += TruncatedLognormal(6.17, 2.36, 50, 2 * 1024 * 1024);
        }
        const double parsing = m_exponential->GetValue(0.13, 0.0);
        m_event = Simulator::Schedule(Seconds(parsing),
                                      &TgaxSourceApplication::BeginHttpPage,
                                      this,
                                      bytes,
                                      objects);
    }

    void BeginHttpPage(uint64_t bytes, uint32_t objects)
    {
        m_sessionPending = true;
        Generate(bytes,
                 "HTTP_PAGE_OBJECTS=" + std::to_string(objects) +
                 (m_httpPersistent ? ";HTTP11_PERSISTENT" : ";HTTP10_STYLE"));
    }

    void HttpRequest()
    {
        if (!m_running)
        {
            return;
        }
        Generate(350, "HTTP_REQUEST_350B");
        m_event = Simulator::Schedule(Seconds(m_exponential->GetValue(30.0, 0.0)),
                                      &TgaxSourceApplication::HttpRequest,
                                      this);
    }

    FlowRow m_row;
    Ipv4Address m_peer;
    uint16_t m_port{};
    Ptr<Socket> m_socket;
    Ptr<UniformRandomVariable> m_uniform;
    Ptr<ExponentialRandomVariable> m_exponential;
    Ptr<WeibullRandomVariable> m_weibull;
    Ptr<GammaRandomVariable> m_gamma;
    Ptr<LargestExtremeValueRandomVariable> m_extreme;
    Ptr<LogNormalRandomVariable> m_lognormal;
    Ptr<ParetoRandomVariable> m_pareto;
    EventId m_event;
    Time m_stateEnd;
    uint64_t m_offeredBytes{};
    uint64_t m_pendingBytes{};
    uint32_t m_segmentSize{1460};
    bool m_running{};
    bool m_started{};
    bool m_voipActive{};
    bool m_sessionPending{};
    bool m_httpPersistent{};
};

NS_OBJECT_ENSURE_REGISTERED(TgaxSourceApplication);

class MacSaturationApplication : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::MacSaturationApplication")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<MacSaturationApplication>();
        return tid;
    }

    void Setup(Ptr<WifiNetDevice> device,
               const std::string& ac,
               const std::vector<Mac48Address>& destinations,
               uint32_t lowWatermark,
               uint32_t highWatermark,
               uint32_t refillIntervalUs,
               int64_t stream)
    {
        m_device = device;
        m_ac = ac;
        m_destinations = destinations;
        m_lowWatermark = lowWatermark;
        m_highWatermark = highWatermark;
        m_refillIntervalUs = refillIntervalUs;
        m_extreme = CreateObject<LargestExtremeValueRandomVariable>();
        m_uniform = CreateObject<UniformRandomVariable>();
        m_extreme->SetStream(stream);
        m_uniform->SetStream(stream + 1);
    }

    uint64_t GetInjectedPackets() const { return m_injectedPackets; }
    uint64_t GetInjectedPayloadBytes() const { return m_injectedPayloadBytes; }
    uint64_t GetRefillEvents() const { return m_refillEvents; }

  private:
    void StartApplication() override
    {
        m_running = true;
        Fill();
    }

    void StopApplication() override
    {
        m_running = false;
        Simulator::Cancel(m_event);
    }

    AcIndex Ac() const
    {
        if (m_ac == "VO") return AC_VO;
        if (m_ac == "VI") return AC_VI;
        if (m_ac == "BK") return AC_BK;
        return AC_BE;
    }

    uint8_t Priority() const
    {
        if (m_ac == "VO") return 6;
        if (m_ac == "VI") return 5;
        if (m_ac == "BK") return 1;
        return 0;
    }

    uint8_t Tid() const { return Priority(); }

    uint8_t AcNumber() const
    {
        if (m_ac == "VO") return 0;
        if (m_ac == "VI") return 1;
        if (m_ac == "BE") return 2;
        return 3;
    }

    uint32_t PacketBytes()
    {
        // Full buffer changes only the arrival process.  Payload lengths retain
        // representative per-service characteristics and always fit in one
        // legal MSDU; file/page sizes are never used as a single MAC frame.
        if (m_ac == "VO") return 33; // TGax active VoIP payload
        if (m_ac == "VI")
        {
            const double gaming = m_extreme->GetValue(158.0, 26.2); // TGax uplink gaming
            return static_cast<uint32_t>(std::clamp(std::round(gaming), 64.0, 1500.0));
        }
        if (m_ac == "BE") return m_uniform->GetInteger(800, 1500);
        return m_uniform->GetInteger(1200, 1500);
    }

    void Fill()
    {
        if (!m_running || m_destinations.empty()) return;
        const auto queue = m_device->GetMac()->GetQosTxop(Ac())->GetWifiMacQueue();
        if (queue->GetNPackets() <= m_lowWatermark)
        {
            ++m_refillEvents;
            uint32_t guard = 0;
            while (queue->GetNPackets() < m_highWatermark && guard++ < m_highWatermark * 2)
            {
                const uint32_t payloadBytes = PacketBytes();
                auto packet = Create<Packet>(payloadBytes);
                LlcSnapHeader llc;
                llc.SetType(0x88b5); // experimental local payload, no IP/TCP/UDP header
                packet->AddHeader(llc);
                MacFullBufferTag tag;
                tag.Set(AcNumber(), payloadBytes);
                packet->AddPacketTag(tag);
                const auto destination = m_destinations[m_nextDestination++ % m_destinations.size()];
                auto mac = m_device->GetMac();
                // WifiNetDevice::DoSend normally fires MacTx before calling
                // WifiMac::Enqueue.  This direct source deliberately bypasses
                // NetDevice/IP, so preserve the same admission trace explicitly.
                mac->NotifyTx(packet);
                mac->Enqueue(packet, destination, mac->GetAddress(), Tid());
                ++m_injectedPackets;
                m_injectedPayloadBytes += payloadBytes;
            }
        }
        m_event = Simulator::Schedule(MicroSeconds(m_refillIntervalUs),
                                      &MacSaturationApplication::Fill,
                                      this);
    }

    Ptr<WifiNetDevice> m_device;
    Ptr<LargestExtremeValueRandomVariable> m_extreme;
    Ptr<UniformRandomVariable> m_uniform;
    std::string m_ac;
    std::vector<Mac48Address> m_destinations;
    uint32_t m_lowWatermark{32};
    uint32_t m_highWatermark{64};
    uint32_t m_refillIntervalUs{100};
    uint64_t m_nextDestination{};
    uint64_t m_injectedPackets{};
    uint64_t m_injectedPayloadBytes{};
    uint64_t m_refillEvents{};
    EventId m_event;
    bool m_running{};
};

NS_OBJECT_ENSURE_REGISTERED(MacSaturationApplication);

// TGax traffic clocks and size distributions feeding WifiMac directly.  This
// deliberately removes TCP/UDP/IP while retaining the application demand
// process.  Large video frames, files and pages are kept as demand and split
// into legal MSDUs before controlled admission to the ordinary QoS queue.
class MacArrivalApplication : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::MacArrivalApplication")
                                .SetParent<Application>()
                                .SetGroupName("Applications")
                                .AddConstructor<MacArrivalApplication>();
        return tid;
    }

    void Setup(const FlowRow& row,
               Ptr<WifiNetDevice> device,
               Mac48Address destination,
               uint32_t highWatermark,
               uint32_t admissionPollUs,
               int64_t stream)
    {
        m_row = row;
        m_device = device;
        m_destination = destination;
        m_highWatermark = highWatermark;
        m_admissionPollUs = admissionPollUs;
        m_uniform = CreateObject<UniformRandomVariable>();
        m_exponential = CreateObject<ExponentialRandomVariable>();
        m_weibull = CreateObject<WeibullRandomVariable>();
        m_gamma = CreateObject<GammaRandomVariable>();
        m_extreme = CreateObject<LargestExtremeValueRandomVariable>();
        m_lognormal = CreateObject<LogNormalRandomVariable>();
        m_pareto = CreateObject<ParetoRandomVariable>();
        m_pollUniform = CreateObject<UniformRandomVariable>();
        m_uniform->SetStream(stream);
        m_exponential->SetStream(stream + 1);
        m_weibull->SetStream(stream + 2);
        m_gamma->SetStream(stream + 3);
        m_extreme->SetStream(stream + 4);
        m_lognormal->SetStream(stream + 5);
        m_pareto->SetStream(stream + 6);
        m_pollUniform->SetStream(stream + 7);
        if (m_row.model == "FTP")
        {
            m_segmentSize = m_uniform->GetValue() < 0.76 ? 1460 : 536;
        }
        else if (m_row.model == "HTTP_REQ")
        {
            m_segmentSize = 350;
        }
        m_httpPersistent = m_uniform->GetValue() < 0.5;
    }

    uint64_t GetGeneratedEvents() const { return m_generatedEvents; }
    uint64_t GetGeneratedBytes() const { return m_generatedBytes; }
    uint64_t GetInjectedPackets() const { return m_injectedPackets; }
    uint64_t GetInjectedBytes() const { return m_injectedBytes; }
    uint64_t GetPendingPackets() const { return m_pending.size(); }
    uint64_t GetPendingBytes() const { return m_pendingBytes; }
    uint64_t GetMaximumPendingBytes() const { return m_maximumPendingBytes; }

  private:
    void StartApplication() override
    {
        m_running = true;
        m_modelEvent = Simulator::Schedule(MilliSeconds(m_uniform->GetValue(0.0, 20.0)),
                                           &MacArrivalApplication::StartModel,
                                           this);
        m_admissionEvent = Simulator::Schedule(MicroSeconds(m_pollUniform->GetValue(1.0, m_admissionPollUs)),
                                               &MacArrivalApplication::AdmissionTick,
                                               this);
    }

    void StopApplication() override
    {
        m_running = false;
        Simulator::Cancel(m_modelEvent);
        Simulator::Cancel(m_admissionEvent);
    }

    AcIndex Ac() const
    {
        if (m_row.ac == "VO") return AC_VO;
        if (m_row.ac == "VI") return AC_VI;
        if (m_row.ac == "BK") return AC_BK;
        return AC_BE;
    }

    uint8_t Tid() const
    {
        if (m_row.ac == "VO") return 6;
        if (m_row.ac == "VI") return 5;
        if (m_row.ac == "BK") return 1;
        return 0;
    }

    uint8_t AcNumber() const
    {
        if (m_row.ac == "VO") return 0;
        if (m_row.ac == "VI") return 1;
        if (m_row.ac == "BE") return 2;
        return 3;
    }

    void StartModel()
    {
        if (!m_running) return;
        if (m_row.model == "VOIP")
        {
            m_voipActive = m_uniform->GetValue() < 0.5;
            m_stateEnd = Simulator::Now() + Seconds(m_exponential->GetValue(1.25, 0.0));
            VoipTick();
        }
        else if (m_row.model == "GMG") GamingTick();
        else if (m_row.model == "BV3" || m_row.model == "BV6") VideoTick();
        else if (m_row.model == "FTP") FtpSession();
        else if (m_row.model == "HTTP") HttpSession();
        else if (m_row.model == "HTTP_REQ") HttpRequest();
    }

    void LogGeneration(uint64_t bytes, const std::string& detail)
    {
        if (!g_fullTrace) return;
        g_appEvents << std::fixed << std::setprecision(9) << Simulator::Now().GetSeconds()
                    << ",GENERATE," << m_row.id << ',' << m_row.bss << ',' << m_row.service
                    << ',' << m_row.model << ',' << m_row.direction << ",NONE," << m_row.ac
                    << ',' << bytes << ',' << detail << '\n';
    }

    void Generate(uint64_t bytes, const std::string& detail)
    {
        if (!m_running) return;
        ++m_generatedEvents;
        m_generatedBytes += bytes;
        LogGeneration(bytes, detail);
        uint64_t remaining = bytes;
        while (remaining > 0)
        {
            const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(remaining, m_segmentSize));
            m_pending.push_back(chunk);
            m_pendingBytes += chunk;
            remaining -= chunk;
        }
        m_maximumPendingBytes = std::max(m_maximumPendingBytes, m_pendingBytes);
    }

    void AdmissionTick()
    {
        if (!m_running) return;
        const auto queue = m_device->GetMac()->GetQosTxop(Ac())->GetWifiMacQueue();
        // One packet per source poll prevents a single AP-side flow from
        // monopolising a shared AC queue.  Randomised polls are reproducible
        // and independent of the traffic-model RNG streams.
        if (!m_pending.empty() && queue->GetNPackets() < m_highWatermark)
        {
            const uint32_t payloadBytes = m_pending.front();
            m_pending.pop_front();
            m_pendingBytes -= payloadBytes;
            auto packet = Create<Packet>(payloadBytes);
            LlcSnapHeader llc;
            llc.SetType(0x88b5);
            packet->AddHeader(llc);
            MacFullBufferTag tag;
            tag.Set(AcNumber(), payloadBytes);
            packet->AddPacketTag(tag);
            auto mac = m_device->GetMac();
            mac->NotifyTx(packet);
            mac->Enqueue(packet, m_destination, mac->GetAddress(), Tid());
            ++m_injectedPackets;
            m_injectedBytes += payloadBytes;
        }
        if (m_pending.empty() && m_sessionPending)
        {
            m_sessionPending = false;
            const double wait = m_row.model == "FTP" ? m_exponential->GetValue(180.0, 0.0)
                                                       : m_exponential->GetValue(30.0, 0.0);
            m_modelEvent = Simulator::Schedule(Seconds(wait),
                                               m_row.model == "FTP" ? &MacArrivalApplication::FtpSession
                                                                     : &MacArrivalApplication::HttpSession,
                                               this);
        }
        const double factor = m_pollUniform->GetValue(0.5, 1.5);
        m_admissionEvent = Simulator::Schedule(MicroSeconds(std::max(1.0, factor * m_admissionPollUs)),
                                               &MacArrivalApplication::AdmissionTick,
                                               this);
    }

    void VoipTick()
    {
        if (!m_running) return;
        if (Simulator::Now() >= m_stateEnd)
        {
            m_voipActive = !m_voipActive;
            m_stateEnd = Simulator::Now() + Seconds(m_exponential->GetValue(1.25, 0.0));
            if (g_fullTrace)
            {
                g_appEvents << Simulator::Now().GetSeconds() << ",STATE," << m_row.id << ','
                            << m_row.bss << ',' << m_row.service << ',' << m_row.model << ','
                            << m_row.direction << ",NONE," << m_row.ac << ",0,"
                            << (m_voipActive ? "ACTIVE" : "SILENCE") << '\n';
            }
        }
        Generate(m_voipActive ? 33 : 7, m_voipActive ? "ACTIVE" : "SILENCE");
        double intervalMs = m_voipActive ? 20.0 : 160.0;
        if (m_row.direction == "DL")
        {
            const double u = m_uniform->GetValue(-0.5, 0.5);
            const double jitter = -5.11 * std::copysign(std::log(1.0 - 2.0 * std::abs(u)), u);
            intervalMs = std::max(0.001, intervalMs + jitter);
        }
        m_modelEvent = Simulator::Schedule(MilliSeconds(intervalMs), &MacArrivalApplication::VoipTick, this);
    }

    void GamingTick()
    {
        if (!m_running) return;
        const bool dl = m_row.direction == "DL";
        const double size = m_extreme->GetValue(dl ? 390.0 : 158.0, dl ? 89.0 : 26.2);
        Generate(static_cast<uint64_t>(std::max(1.0, std::round(size))), "GMG_PACKET");
        const double iat = m_extreme->GetValue(dl ? 15.0 : 23.5, dl ? 7.0 : 10.5);
        m_modelEvent = Simulator::Schedule(MilliSeconds(std::max(0.001, iat)),
                                           &MacArrivalApplication::GamingTick,
                                           this);
    }

    void VideoTick()
    {
        if (!m_running) return;
        const bool bv6 = m_row.model == "BV6";
        const double lambda = bv6 ? 54210.0 : 20850.0;
        const double bitrate = bv6 ? 15.6e6 : 6.0e6;
        const double shape = 0.8099;
        const uint64_t bytes = static_cast<uint64_t>(std::max(1.0, std::round(m_weibull->GetValue(lambda, shape, 0.0))));
        const double delayMs = m_gamma->GetValue(0.2463, 60.227);
        Simulator::Schedule(MilliSeconds(delayMs), &MacArrivalApplication::Generate,
                            this, bytes, "VIDEO_FRAME");
        const double meanBytes = lambda * std::tgamma(1.0 + 1.0 / shape);
        m_modelEvent = Simulator::Schedule(Seconds(meanBytes * 8.0 / bitrate),
                                           &MacArrivalApplication::VideoTick,
                                           this);
    }

    uint64_t TruncatedLognormal(double mu, double sigma, uint64_t minimum, uint64_t maximum)
    {
        for (unsigned attempt = 0; attempt < 100000; ++attempt)
        {
            const double value = m_lognormal->GetValue(mu, sigma);
            if (value >= minimum && value <= maximum) return static_cast<uint64_t>(std::round(value));
        }
        NS_FATAL_ERROR("Truncated lognormal rejection sampler failed");
        return minimum;
    }

    void FtpSession()
    {
        if (!m_running) return;
        m_sessionPending = true;
        Generate(TruncatedLognormal(14.45, 0.35, 1, 5 * 1024 * 1024), "FTP_FILE");
    }

    void HttpSession()
    {
        if (!m_running) return;
        uint64_t bytes = TruncatedLognormal(8.37, 1.37, 100, 2 * 1024 * 1024);
        const double raw = m_pareto->GetValue(2.0, 1.1, 55.0);
        const uint32_t objects = std::min<uint32_t>(53, static_cast<uint32_t>(std::max(0.0, std::floor(raw - 2.0))));
        for (uint32_t i = 0; i < objects; ++i)
        {
            bytes += TruncatedLognormal(6.17, 2.36, 50, 2 * 1024 * 1024);
        }
        m_modelEvent = Simulator::Schedule(Seconds(m_exponential->GetValue(0.13, 0.0)),
                                           &MacArrivalApplication::BeginHttpPage,
                                           this, bytes, objects);
    }

    void BeginHttpPage(uint64_t bytes, uint32_t objects)
    {
        m_sessionPending = true;
        Generate(bytes, "HTTP_PAGE_OBJECTS=" + std::to_string(objects) +
                        (m_httpPersistent ? ";HTTP11_PERSISTENT" : ";HTTP10_STYLE"));
    }

    void HttpRequest()
    {
        if (!m_running) return;
        Generate(350, "HTTP_REQUEST_350B");
        m_modelEvent = Simulator::Schedule(Seconds(m_exponential->GetValue(30.0, 0.0)),
                                           &MacArrivalApplication::HttpRequest,
                                           this);
    }

    FlowRow m_row;
    Ptr<WifiNetDevice> m_device;
    Mac48Address m_destination;
    Ptr<UniformRandomVariable> m_uniform;
    Ptr<UniformRandomVariable> m_pollUniform;
    Ptr<ExponentialRandomVariable> m_exponential;
    Ptr<WeibullRandomVariable> m_weibull;
    Ptr<GammaRandomVariable> m_gamma;
    Ptr<LargestExtremeValueRandomVariable> m_extreme;
    Ptr<LogNormalRandomVariable> m_lognormal;
    Ptr<ParetoRandomVariable> m_pareto;
    std::deque<uint32_t> m_pending;
    EventId m_modelEvent;
    EventId m_admissionEvent;
    Time m_stateEnd;
    uint32_t m_segmentSize{1460};
    uint32_t m_highWatermark{64};
    uint32_t m_admissionPollUs{500};
    uint64_t m_pendingBytes{};
    uint64_t m_generatedEvents{};
    uint64_t m_generatedBytes{};
    uint64_t m_injectedPackets{};
    uint64_t m_injectedBytes{};
    uint64_t m_maximumPendingBytes{};
    bool m_running{};
    bool m_voipActive{};
    bool m_sessionPending{};
    bool m_httpPersistent{};
};

NS_OBJECT_ENSURE_REGISTERED(MacArrivalApplication);

bool
DiscardFullBufferPayload(Ptr<NetDevice>, Ptr<const Packet>, uint16_t, const Address&)
{
    // The experiment intentionally terminates the payload at the receiving
    // WifiNetDevice.  No IPv4, UDP, TCP or Socket object participates.
    return true;
}

struct NodeInfo
{
    std::string label;
    std::string role;
    uint32_t bss{};
    int32_t sta{-1};
    uint16_t channel{42};
    Vector position;
    Ptr<Node> node;
    Ptr<WifiNetDevice> device;
    Ipv4Address address;
};

struct RuntimeFlow
{
    FlowRow row;
    uint16_t port{};
    Ptr<TgaxSourceApplication> source;
    Ptr<PacketSink> sink;
};

struct QueueAggregate
{
    uint64_t samples{};
    uint64_t totalPackets{};
    uint64_t maximum{};
};

std::map<uint32_t, double> g_busySeconds;
std::map<uint32_t, QueueAggregate> g_queueAggregate;
using QueueKey = std::tuple<uint32_t, std::string, std::string>; // BSS, UL/DL, AC
std::map<QueueKey, QueueAggregate> g_queueByDirectionAc;
std::vector<NodeInfo*> g_nodesForQueue;

std::string
PhyStateName(WifiPhyState state)
{
    std::ostringstream stream;
    stream << state;
    return stream.str();
}

void
StateTrace(std::string label, uint32_t bss, Time start, Time duration, WifiPhyState state)
{
    const double left = std::max(g_measureStart, start.GetSeconds());
    const double right = std::min(g_measureEnd, (start + duration).GetSeconds());
    if (right > left && state != WifiPhyState::IDLE && state != WifiPhyState::SLEEP)
    {
        g_busySeconds[bss] += right - left;
    }
    if (g_fullTrace)
    {
        g_phyStates << std::fixed << std::setprecision(9) << start.GetSeconds() << ','
                    << duration.GetSeconds() << ',' << label << ',' << bss << ','
                    << PhyStateName(state) << '\n';
    }
}

void
BackoffTrace(std::string label, uint32_t bss, std::string ac, uint32_t slots, uint8_t linkId)
{
    if (g_fullTrace)
    {
        g_backoffEvents << std::fixed << std::setprecision(9) << Simulator::Now().GetSeconds()
                        << ',' << label << ',' << bss << ',' << ac << ',' << slots << ','
                        << +linkId << '\n';
    }
}

void
CwTrace(uint32_t bss, std::string ac, uint32_t value, uint8_t)
{
    const double now = Simulator::Now().GetSeconds();
    if (bss == 0 && now >= g_measureStart && now < g_measureEnd)
    {
        g_cwValues[ac].insert(value);
    }
}

std::string
FrameType(const WifiMacHeader& header)
{
    if (header.IsBlockAck()) return "BLOCK_ACK";
    if (header.IsBlockAckReq()) return "BLOCK_ACK_REQ";
    if (header.IsAck()) return "MAC_ACK";
    if (header.IsRts()) return "RTS";
    if (header.IsCts()) return "CTS";
    if (header.IsBeacon()) return "BEACON";
    if (header.IsAssocReq()) return "ASSOC_REQ";
    if (header.IsAssocResp()) return "ASSOC_RESP";
    if (header.IsQosData()) return "QOS_DATA";
    if (header.IsData()) return "DATA";
    return "OTHER";
}

std::string
PacketAc(Ptr<const Packet> original)
{
    MacFullBufferTag fullBuffer;
    if (original->PeekPacketTag(fullBuffer))
    {
        switch (fullBuffer.GetAc())
        {
        case 0: return "VO";
        case 1: return "VI";
        case 2: return "BE";
        case 3: return "BK";
        default: return "NA";
        }
    }
    SocketPriorityTag priority;
    if (original->PeekPacketTag(priority))
    {
        const uint8_t value = priority.GetPriority();
        if (value == 6 || value == 7) return "VO";
        if (value == 4 || value == 5) return "VI";
        if (value == 1 || value == 2) return "BK";
        return "BE";
    }
    auto packet = original->Copy();
    LlcSnapHeader llc;
    if (!packet->RemoveHeader(llc))
    {
        return "NA";
    }
    Ipv4Header ipv4;
    if (!packet->RemoveHeader(ipv4))
    {
        return "NA";
    }
    const uint8_t tos = ipv4.GetTos();
    if (tos == 0xc0) return "VO";
    if (tos == 0xb8) return "VI";
    if (tos == 0x20) return "BK";
    return "BE";
}

uint32_t
EffectivePayloadBytes(Ptr<const Packet> packet)
{
    MacFullBufferTag fullBuffer;
    return packet->PeekPacketTag(fullBuffer) ? fullBuffer.GetPayloadBytes() : packet->GetSize();
}

void
DeadlineReached(std::string label, uint64_t uid)
{
    const auto key = std::make_pair(label, uid);
    const auto found = g_deadlines.find(key);
    if (found == g_deadlines.end() || found->second.matured) return;
    ++g_macOutcomes[found->second.key].deadlineEligible;
    found->second.matured = true;
}

void
ResolveDeadline(const std::pair<std::string, uint64_t>& key, bool acked)
{
    const auto found = g_deadlines.find(key);
    if (found == g_deadlines.end()) return;
    if (!found->second.matured)
    {
        auto& outcome = g_macOutcomes[found->second.key];
        ++outcome.deadlineEligible;
        if (acked) ++outcome.onTimeDelivered;
    }
    g_deadlines.erase(found);
}

std::string
DropReason(WifiMacDropReason reason)
{
    switch (reason)
    {
    case WIFI_MAC_DROP_FAILED_ENQUEUE: return "FAILED_ENQUEUE";
    case WIFI_MAC_DROP_EXPIRED_LIFETIME: return "EXPIRED_LIFETIME";
    case WIFI_MAC_DROP_REACHED_RETRY_LIMIT: return "RETRY_LIMIT";
    case WIFI_MAC_DROP_QOS_OLD_PACKET: return "QOS_OLD_PACKET";
    }
    return "UNKNOWN";
}

void
MacArrivalTrace(std::string label, std::string role, uint32_t bss, Ptr<const Packet> packet)
{
    const auto key = std::make_pair(label, packet->GetUid());
    const auto ac = PacketAc(packet);
    const double now = Simulator::Now().GetSeconds();
    if (now >= g_measureStart && now < g_measureEnd && ac != "NA")
    {
        const uint32_t payloadBytes = EffectivePayloadBytes(packet);
        MacPending pending{now, bss, role == "AP" ? "DL" : "UL", ac, payloadBytes};
        // WifiMac::MacTx can be observed again for the same packet while it is
        // internally re-forwarded.  A MAC service customer is identified by
        // node + packet UID, so count its first admission only.
        if (g_macPending.emplace(key, pending).second)
        {
            if (g_fullTrace)
            {
                g_accessPending[key] = now;
            }
            auto& outcome = g_macOutcomes[{bss, pending.direction, ac}];
            ++outcome.arrived;
            outcome.arrivedBytes += payloadBytes;
            if (const NodeAcKey source{label, ac}; g_fullBufferSources.contains(source))
            {
                ++g_fullBufferSourceStats[source].arrived;
            }
            const double deadlineMs = ac == "VO" ? 20.0 : ac == "VI" ? 50.0 : 0.0;
            if (deadlineMs > 0)
            {
                g_deadlines[key] = DeadlineStatus{{bss, pending.direction, ac}, false};
                Simulator::Schedule(MilliSeconds(deadlineMs), &DeadlineReached, label, packet->GetUid());
            }
        }
        else
        {
            ++g_duplicateAdmissions;
        }
    }
    if (g_fullTrace)
    {
        g_macPacketEvents << std::fixed << std::setprecision(9) << now
                          << ',' << label << ',' << bss << ",ARRIVAL," << packet->GetUid()
                          << ",NONE," << ac << ',' << EffectivePayloadBytes(packet) << '\n';
    }
}

void
MacTxDropTrace(std::string label, uint32_t bss, Ptr<const Packet> packet)
{
    const auto key = std::make_pair(label, packet->GetUid());
    if (const auto found = g_macPending.find(key); found != g_macPending.end())
    {
        auto& outcome = g_macOutcomes[{found->second.bss, found->second.direction, found->second.ac}];
        ++outcome.failed;
        ++outcome.failedEnqueue;
        outcome.failedBytes += found->second.bytes;
        if (const NodeAcKey source{label, found->second.ac}; g_fullBufferSources.contains(source))
        {
            ++g_fullBufferSourceStats[source].failed;
        }
        g_macPending.erase(found);
        g_accessPending.erase(key);
        ResolveDeadline(key, false);
    }
    if (g_fullTrace)
    {
        g_macPacketEvents << std::fixed << std::setprecision(9) << Simulator::Now().GetSeconds()
                          << ',' << label << ',' << bss << ",MAC_TX_DROP," << packet->GetUid()
                          << ",PRE_QUEUE," << PacketAc(packet) << ',' << EffectivePayloadBytes(packet) << '\n';
    }
}

void
AckedMpduTrace(std::string label, uint32_t bss, Ptr<const WifiMpdu> mpdu)
{
    const auto uid = mpdu->GetPacket()->GetUid();
    const auto& header = mpdu->GetHeader();
    const std::string ac = header.IsQosData() ? AcName(QosUtilsMapTidToAc(header.GetQosTid())) : "NA";
    const auto key = std::make_pair(label, uid);
    if (const auto found = g_macPending.find(key); found != g_macPending.end())
    {
        auto& outcome = g_macOutcomes[{found->second.bss, found->second.direction, found->second.ac}];
        ++outcome.acked;
        outcome.txAttempts += found->second.txAttempts;
        outcome.retransmissions += found->second.txAttempts > 0 ? found->second.txAttempts - 1 : 0;
        outcome.ackedBytes += found->second.bytes;
        outcome.serviceDelayMs.push_back((Simulator::Now().GetSeconds() - found->second.arrival) * 1000.0);
        if (const NodeAcKey source{label, found->second.ac}; g_fullBufferSources.contains(source))
        {
            auto& stats = g_fullBufferSourceStats[source];
            ++stats.acked;
            stats.ackedBytes += found->second.bytes;
        }
        g_macPending.erase(found);
        g_accessPending.erase(key);
        ResolveDeadline(key, true);
    }
    if (g_fullTrace)
    {
        g_macPacketEvents << std::fixed << std::setprecision(9) << Simulator::Now().GetSeconds()
                          << ',' << label << ',' << bss << ",ACKED," << uid << ",NONE,"
                          << ac << ',' << EffectivePayloadBytes(mpdu->GetPacket()) << '\n';
    }
}

void
DroppedMpduTrace(std::string label,
                 uint32_t bss,
                 WifiMacDropReason reason,
                 Ptr<const WifiMpdu> mpdu)
{
    const auto uid = mpdu->GetPacket()->GetUid();
    const auto& header = mpdu->GetHeader();
    const std::string ac = header.IsQosData() ? AcName(QosUtilsMapTidToAc(header.GetQosTid())) : "NA";
    const auto key = std::make_pair(label, uid);
    if (const auto found = g_macPending.find(key); found != g_macPending.end())
    {
        auto& outcome = g_macOutcomes[{found->second.bss, found->second.direction, found->second.ac}];
        ++outcome.failed;
        outcome.txAttempts += found->second.txAttempts;
        outcome.retransmissions += found->second.txAttempts > 0 ? found->second.txAttempts - 1 : 0;
        outcome.failedBytes += found->second.bytes;
        if (const NodeAcKey source{label, found->second.ac}; g_fullBufferSources.contains(source))
        {
            ++g_fullBufferSourceStats[source].failed;
        }
        switch (reason)
        {
        case WIFI_MAC_DROP_FAILED_ENQUEUE: ++outcome.failedEnqueue; break;
        case WIFI_MAC_DROP_EXPIRED_LIFETIME: ++outcome.expiredLifetime; break;
        case WIFI_MAC_DROP_REACHED_RETRY_LIMIT: ++outcome.retryLimit; break;
        case WIFI_MAC_DROP_QOS_OLD_PACKET: ++outcome.qosOldPacket; break;
        default: ++outcome.otherFailure; break;
        }
        g_macPending.erase(found);
        g_accessPending.erase(key);
        ResolveDeadline(key, false);
    }
    if (g_fullTrace)
    {
        g_macPacketEvents << std::fixed << std::setprecision(9) << Simulator::Now().GetSeconds()
                          << ',' << label << ',' << bss << ",DROPPED," << uid << ','
                          << DropReason(reason) << ',' << ac << ','
                          << EffectivePayloadBytes(mpdu->GetPacket()) << '\n';
    }
}

bool
IsTcpAckOnly(Ptr<const WifiMpdu> mpdu)
{
    if (!mpdu->GetHeader().IsQosData())
    {
        return false;
    }
    auto packet = mpdu->GetPacket()->Copy();
    LlcSnapHeader llc;
    if (!packet->RemoveHeader(llc))
    {
        return false;
    }
    Ipv4Header ipv4;
    if (!packet->RemoveHeader(ipv4) || ipv4.GetProtocol() != 6)
    {
        return false;
    }
    TcpHeader tcp;
    if (!packet->RemoveHeader(tcp))
    {
        return false;
    }
    return packet->GetSize() == 0 && (tcp.GetFlags() & TcpHeader::ACK);
}

void
PhyTxTrace(std::string label,
           std::string role,
           uint32_t bss,
           WifiConstPsduMap psduMap,
           WifiTxVector txVector,
           double txPowerW)
{
    uint32_t bytes = 0;
    uint32_t mpdus = 0;
    bool tcpAck = false;
    std::string type = "EMPTY";
    std::string ac = "NA";
    for (const auto& [staId, psdu] : psduMap)
    {
        bytes += psdu->GetSize();
        mpdus += psdu->GetNMpdus();
        if (psdu->GetNMpdus())
        {
            const auto& header = psdu->GetHeader(0);
            type = FrameType(header);
            if (header.IsQosData())
            {
                ac = AcName(QosUtilsMapTidToAc(header.GetQosTid()));
            }
            for (const auto& mpdu : *PeekPointer(psdu))
            {
                const auto key = std::make_pair(label, mpdu->GetPacket()->GetUid());
                if (const auto pending = g_macPending.find(key); pending != g_macPending.end())
                {
                    ++pending->second.txAttempts;
                }
                const auto& mpduHeader = mpdu->GetHeader();
                const bool tcpAckOnly = mpduHeader.IsQosData() && IsTcpAckOnly(mpdu);
                const double now = Simulator::Now().GetSeconds();
                if (tcpAckOnly && now >= g_measureStart && now <= g_measureEnd &&
                    g_seenTcpAckMpdus.insert(key).second)
                {
                    const std::string mpduAc = AcName(QosUtilsMapTidToAc(mpduHeader.GetQosTid()));
                    ++g_tcpAckMpdus[{bss, role == "AP" ? "DL" : "UL", mpduAc}];
                }
                if (!g_fullTrace)
                {
                    continue;
                }
                tcpAck = tcpAck || tcpAckOnly;
                const auto arrival = g_accessPending.find(key);
                if (arrival != g_accessPending.end())
                {
                    const std::string mpduAc = mpduHeader.IsQosData()
                                                   ? AcName(QosUtilsMapTidToAc(mpduHeader.GetQosTid()))
                                                   : "NA";
                    if (g_fullTrace)
                    {
                        g_macAccessDelay << std::fixed << std::setprecision(9)
                                         << Simulator::Now().GetSeconds() << ',' << label << ',' << bss
                                         << ',' << mpdu->GetPacket()->GetUid() << ',' << mpduAc << ','
                                         << (Simulator::Now().GetSeconds() - arrival->second) * 1000.0
                                         << ',' << mpdu->GetSize() << '\n';
                    }
                    g_accessPending.erase(arrival); // first PHY start only; retries are not new samples
                }
            }
        }
    }
    const Time duration = WifiPhy::CalculateTxDuration(psduMap, txVector, WIFI_PHY_BAND_5GHZ);
    const double now = Simulator::Now().GetSeconds();
    if (bss == 0 && role == "STA" && type == "QOS_DATA" && ac != "NA" &&
        now >= g_measureStart && now < g_measureEnd)
    {
        auto& stats = g_ampduStats[ac];
        ++stats.psdus;
        stats.ampduPsdus += mpdus > 1;
        stats.mpdus += mpdus;
        stats.maximumMpdus = std::max<uint64_t>(stats.maximumMpdus, mpdus);
    }
    if (g_fullTrace)
    {
        g_macEvents << std::fixed << std::setprecision(9) << Simulator::Now().GetSeconds()
                    << ',' << duration.GetSeconds() << ',' << label << ',' << bss << ',' << type
                    << ',' << ac << ',' << mpdus << ',' << bytes << ',' << (mpdus > 1)
                    << ',' << tcpAck << ',' << txVector.GetMode().GetUniqueName() << ','
                    << +txVector.GetNss() << ',' << txVector.GetChannelWidth() << ','
                    << 10.0 * std::log10(txPowerW * 1000.0) << '\n';
    }
}

void
SampleQueues()
{
    std::set<std::string> activeStas;
    std::set<std::string> activeNodes;
    uint64_t activeQueues = 0;
    uint64_t activeQueuePackets = 0;
    for (auto* info : g_nodesForQueue)
    {
        for (AcIndex ac : {AC_VO, AC_VI, AC_BE, AC_BK})
        {
            const auto queue = info->device->GetMac()->GetQosTxop(ac)->GetWifiMacQueue();
            const uint64_t packets = queue->GetNPackets();
            if (g_fullTrace)
            {
                g_queueSamples << std::fixed << std::setprecision(9) << Simulator::Now().GetSeconds()
                               << ',' << info->label << ',' << info->bss << ',' << info->role << ','
                               << AcName(ac) << ',' << packets << ',' << queue->GetNBytes() << '\n';
            }
            if (Simulator::Now().GetSeconds() >= g_measureStart)
            {
                auto& aggregate = g_queueAggregate[info->bss];
                ++aggregate.samples;
                aggregate.totalPackets += packets;
                aggregate.maximum = std::max(aggregate.maximum, packets);
                auto& detailed = g_queueByDirectionAc[{info->bss,
                                                       info->role == "AP" ? "DL" : "UL",
                                                       AcName(ac)}];
                ++detailed.samples;
                detailed.totalPackets += packets;
                detailed.maximum = std::max(detailed.maximum, packets);
                const NodeAcKey source{info->label, AcName(ac)};
                if (g_fullBufferSources.contains(source))
                {
                    auto& sourceStats = g_fullBufferQueueStats[source];
                    ++sourceStats.samples;
                    sourceStats.nonempty += packets > 0;
                    sourceStats.totalPackets += packets;
                    sourceStats.minimum = std::min(sourceStats.minimum, packets);
                    sourceStats.maximum = std::max(sourceStats.maximum, packets);
                    if (packets > 0)
                    {
                        activeNodes.insert(info->label);
                        if (info->role == "STA")
                        {
                            activeStas.insert(info->label);
                        }
                        ++activeQueues;
                        activeQueuePackets += packets;
                    }
                }
            }
        }
    }
    if (Simulator::Now().GetSeconds() >= g_measureStart &&
        Simulator::Now().GetSeconds() <= g_measureEnd + 1e-9)
    {
        g_competitionSamples << std::fixed << std::setprecision(9)
                             << Simulator::Now().GetSeconds() << ',' << activeStas.size() << ','
                             << activeNodes.size() << ','
                             << activeQueues << ',' << activeQueuePackets << '\n';
    }
    if (Simulator::Now().GetSeconds() + 0.1 <= g_measureEnd + 1e-9)
    {
        Simulator::Schedule(MilliSeconds(100), &SampleQueues);
    }
}

void
ConfigureMac(Ptr<WifiNetDevice> device)
{
    auto mac = device->GetMac();
    mac->SetAttribute("FrameRetryLimit", UintegerValue(10));
    mac->SetAttribute("MpduBufferSize", UintegerValue(64));
    for (const auto* name : {"VO_MaxAmpduSize", "VI_MaxAmpduSize", "BE_MaxAmpduSize", "BK_MaxAmpduSize"})
    {
        mac->SetAttribute(name, UintegerValue(65535));
    }
    for (const auto* name : {"VO_MaxAmsduSize", "VI_MaxAmsduSize", "BE_MaxAmsduSize", "BK_MaxAmsduSize"})
    {
        mac->SetAttribute(name, UintegerValue(0));
    }
}

void
ApplyTargetProfile(Ptr<WifiNetDevice> device, const std::string& profile)
{
    if (profile == "default")
    {
        return;
    }
    auto mac = device->GetMac();
    auto vo = mac->GetQosTxop(AC_VO);
    auto vi = mac->GetQosTxop(AC_VI);
    auto be = mac->GetQosTxop(AC_BE);
    if (profile == "rt-moderate")
    {
        vi->SetMinCw(3);
        vi->SetMaxCw(15);
        vi->SetTxopLimit(MicroSeconds(6016));
        return;
    }
    if (profile == "rt-aggressive")
    {
        vo->SetAifsn(1);
        vo->SetMinCw(1);
        vo->SetMaxCw(3);
        vo->SetTxopLimit(MicroSeconds(3264));
        vi->SetAifsn(1);
        vi->SetMinCw(3);
        vi->SetMaxCw(7);
        vi->SetTxopLimit(MicroSeconds(6016));
        return;
    }
    if (profile == "rt-protect")
    {
        vi->SetMinCw(3);
        vi->SetMaxCw(15);
        vi->SetTxopLimit(MicroSeconds(6016));
        be->SetAifsn(5);
        be->SetMinCw(31);
        be->SetMaxCw(1023);
        return;
    }
    NS_FATAL_ERROR("Unknown target EDCA profile: " << profile);
}

double
Percentile(const std::vector<double>& sorted, double probability)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    const double position = probability * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - lower;
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

void
MergeOutcome(MacOutcome& destination, const MacOutcome& source)
{
    destination.arrived += source.arrived;
    destination.acked += source.acked;
    destination.failed += source.failed;
    destination.unfinished += source.unfinished;
    destination.failedEnqueue += source.failedEnqueue;
    destination.expiredLifetime += source.expiredLifetime;
    destination.retryLimit += source.retryLimit;
    destination.qosOldPacket += source.qosOldPacket;
    destination.otherFailure += source.otherFailure;
    destination.arrivedBytes += source.arrivedBytes;
    destination.ackedBytes += source.ackedBytes;
    destination.failedBytes += source.failedBytes;
    destination.unfinishedBytes += source.unfinishedBytes;
    destination.txAttempts += source.txAttempts;
    destination.retransmissions += source.retransmissions;
    destination.deadlineEligible += source.deadlineEligible;
    destination.onTimeDelivered += source.onTimeDelivered;
    destination.serviceDelayMs.insert(destination.serviceDelayMs.end(),
                                      source.serviceDelayMs.begin(),
                                      source.serviceDelayMs.end());
}

} // namespace

int
main(int argc, char** argv)
{
    std::string topologyPath;
    std::string flowsPath;
    std::string outputPath{"tgax-s3-output"};
    double warmup = 5.0;
    double duration = 2.0;
    double drain = 0.5;
    uint32_t seed = 20261021;
    std::string traceLevel{"full"};
    std::string phyBackend{"spectrum"};
    std::string targetProfile{"default"};
    std::string trafficMode{"applications"};
    double saturationWarmup{1.0};
    uint32_t saturationLowWatermark{32};
    uint32_t saturationHighWatermark{64};
    uint32_t saturationRefillUs{100};
    CommandLine command(__FILE__);
    command.AddValue("topology", "Generated topology CSV", topologyPath);
    command.AddValue("flows", "Generated traffic assignment CSV", flowsPath);
    command.AddValue("output", "Output directory", outputPath);
    command.AddValue("warmup", "Association/application warm-up seconds", warmup);
    command.AddValue("duration", "Measurement seconds", duration);
    command.AddValue("drain", "Post-measurement TCP drain seconds", drain);
    command.AddValue("seed", "Deterministic RNG seed", seed);
    command.AddValue("traceLevel", "Trace detail: full or summary", traceLevel);
    command.AddValue("phyBackend", "PHY channel backend: spectrum or yans", phyBackend);
    command.AddValue("targetProfile", "Fixed EDCA profile applied to AP 0 only", targetProfile);
    command.AddValue("trafficMode", "Traffic source: applications, mac-saturation, or mac-arrival", trafficMode);
    command.AddValue("saturationWarmup", "Direct-MAC source warm-up before measurement", saturationWarmup);
    command.AddValue("saturationLowWatermark", "Refill trigger in packets", saturationLowWatermark);
    command.AddValue("saturationHighWatermark", "Queue target after refill in packets", saturationHighWatermark);
    command.AddValue("saturationRefillUs", "Queue-watermark polling period in microseconds", saturationRefillUs);
    command.Parse(argc, argv);
    NS_ABORT_MSG_IF(topologyPath.empty() || flowsPath.empty(), "--topology and --flows are required");
    NS_ABORT_MSG_IF(traceLevel != "full" && traceLevel != "summary", "--traceLevel must be full or summary");
    NS_ABORT_MSG_IF(phyBackend != "spectrum" && phyBackend != "yans", "--phyBackend must be spectrum or yans");
    NS_ABORT_MSG_IF(targetProfile != "default" && targetProfile != "rt-moderate" &&
                        targetProfile != "rt-aggressive" && targetProfile != "rt-protect",
                    "Unknown --targetProfile");
    NS_ABORT_MSG_IF(trafficMode != "applications" && trafficMode != "mac-saturation" &&
                        trafficMode != "mac-arrival",
                    "--trafficMode must be applications, mac-saturation, or mac-arrival");
    NS_ABORT_MSG_IF(saturationLowWatermark == 0 ||
                        saturationHighWatermark <= saturationLowWatermark ||
                        saturationHighWatermark > 128,
                    "Require 0 < saturationLowWatermark < saturationHighWatermark <= 128");
    NS_ABORT_MSG_IF(saturationRefillUs == 0 || saturationRefillUs > 10000,
                    "--saturationRefillUs must be in [1,10000]");
    g_fullTrace = traceLevel == "full";
    std::filesystem::create_directories(outputPath);
    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(1);
    const bool directMac = trafficMode != "applications";
    const double sourceStart = warmup;
    g_measureStart = warmup + (directMac ? saturationWarmup : 0.0);
    g_measureEnd = g_measureStart + duration;

    g_appEvents.open(outputPath + "/application_events.csv");
    g_macEvents.open(outputPath + "/mac_timeline.csv");
    g_backoffEvents.open(outputPath + "/backoff_events.csv");
    g_phyStates.open(outputPath + "/phy_states.csv");
    g_queueSamples.open(outputPath + "/queue_samples.csv");
    g_macPacketEvents.open(outputPath + "/mac_packet_events.csv");
    g_macAccessDelay.open(outputPath + "/mac_access_delay.csv");
    g_competitionSamples.open(outputPath + "/competition_samples.csv");
    g_appEvents << "time_s,event,flow_id,bss,service,model,direction,transport,ac,bytes,detail\n";
    g_macEvents << "time_s,duration_s,node,bss,frame_type,ac,mpdus,bytes,is_ampdu,tcp_ack_only,mode,nss,width_mhz,tx_power_dbm\n";
    g_backoffEvents << "time_s,node,bss,ac,slots,link_id\n";
    g_phyStates << "start_s,duration_s,node,bss,state\n";
    g_queueSamples << "time_s,node,bss,role,ac,packets,bytes\n";
    g_macPacketEvents << "time_s,node,bss,event,uid,reason,ac,bytes\n";
    g_macAccessDelay << "phy_start_s,node,bss,uid,ac,access_delay_ms,mpdu_bytes\n";
    g_competitionSamples << "time_s,active_stas,active_nodes,active_source_queues,total_source_queue_packets\n";
    std::ofstream simulationConfig(outputPath + "/simulation_config.csv");
    simulationConfig << "seed,association_warmup_s,source_warmup_s,measurement_s,drain_s,trace_level,phy_backend,target_profile,traffic_mode,low_watermark,high_watermark,refill_us\n"
                     << seed << ',' << warmup << ','
                     << (directMac ? saturationWarmup : 0.0) << ','
                     << duration << ',' << drain << ','
                     << traceLevel << ',' << phyBackend << ',' << targetProfile << ','
                     << trafficMode << ',' << saturationLowWatermark << ','
                     << saturationHighWatermark << ',' << saturationRefillUs << '\n';

    std::map<std::string, NodeInfo> nodes;
    std::map<uint32_t, std::vector<std::string>> bssStaLabels;
    std::map<uint32_t, std::string> bssApLabel;
    NodeContainer allNodes;
    for (const auto& row : ReadCsv(topologyPath))
    {
        NodeInfo info;
        info.label = row.at("node_label");
        info.role = row.at("role");
        info.bss = std::stoul(row.at("bss"));
        info.sta = std::stoi(row.at("sta"));
        info.channel = std::stoul(row.at("channel"));
        info.position = Vector(std::stod(row.at("x")), std::stod(row.at("y")), std::stod(row.at("z")));
        info.node = CreateObject<Node>();
        allNodes.Add(info.node);
        if (info.role == "AP") bssApLabel[info.bss] = info.label;
        else bssStaLabels[info.bss].push_back(info.label);
        nodes.emplace(info.label, info);
    }

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);
    for (auto& [label, info] : nodes)
    {
        info.node->GetObject<MobilityModel>()->SetPosition(info.position);
    }
    if (trafficMode == "applications")
    {
        InternetStackHelper internet;
        internet.Install(allNodes);
    }

    auto loss = CreateObject<TgaxIndoorPropagationLossModel>();
    loss->SetSeed(seed);
    Ptr<MultiModelSpectrumChannel> spectrumChannel;
    Ptr<YansWifiChannel> yansChannel;
    if (phyBackend == "spectrum")
    {
        spectrumChannel = CreateObject<MultiModelSpectrumChannel>();
        spectrumChannel->AddPropagationLossModel(loss);
        spectrumChannel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());
    }
    else
    {
        yansChannel = CreateObject<YansWifiChannel>();
        yansChannel->SetPropagationLossModel(loss);
        yansChannel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());
    }

    int64_t stream = 1000;
    for (const auto& [bss, apLabel] : bssApLabel)
    {
        WifiHelper wifi;
        wifi.SetStandard(WIFI_STANDARD_80211ax);
        wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                     "DataMode", StringValue("HeMcs7"),
                                     "ControlMode", StringValue("OfdmRate6Mbps"),
                                     "RtsCtsThreshold", UintegerValue(65535));
        wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(800)));
        WifiMacHelper mac;
        // The fixed 7-BSS research configuration uses exactly 40 STAs per BSS.
        // Keep association deterministic so a STA cannot inherit another BSS's
        // advertised EDCA parameters after shadowing changes its strongest AP.
        const Ssid ssid("tgax-s3-bss-" + std::to_string(bss));

        NodeContainer staNodes;
        for (const auto& label : bssStaLabels[bss]) staNodes.Add(nodes.at(label).node);
        mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
        NodeContainer apNode(nodes.at(apLabel).node);
        NetDeviceContainer staDevices;
        NetDeviceContainer apDevices;
        if (phyBackend == "spectrum")
        {
            SpectrumWifiPhyHelper staPhy;
            staPhy.SetChannel(spectrumChannel);
            staPhy.Set("ChannelSettings", StringValue("{" + std::to_string(nodes.at(apLabel).channel) + ", 80, BAND_5GHZ, 0}"));
            staPhy.Set("TxPowerStart", DoubleValue(15.0));
            staPhy.Set("TxPowerEnd", DoubleValue(15.0));
            staPhy.Set("TxGain", DoubleValue(-2.0));
            staPhy.Set("RxGain", DoubleValue(-2.0));
            staPhy.Set("RxNoiseFigure", DoubleValue(7.0));
            staPhy.Set("Antennas", UintegerValue(1));
            staPhy.Set("MaxSupportedTxSpatialStreams", UintegerValue(1));
            staPhy.Set("MaxSupportedRxSpatialStreams", UintegerValue(1));
            staDevices = wifi.Install(staPhy, mac, staNodes);
            SpectrumWifiPhyHelper apPhy;
            apPhy.SetChannel(spectrumChannel);
            apPhy.Set("ChannelSettings", StringValue("{" + std::to_string(nodes.at(apLabel).channel) + ", 80, BAND_5GHZ, 0}"));
            apPhy.Set("TxPowerStart", DoubleValue(23.0102999566));
            apPhy.Set("TxPowerEnd", DoubleValue(23.0102999566));
            apPhy.Set("TxGain", DoubleValue(0.0));
            apPhy.Set("RxGain", DoubleValue(0.0));
            apPhy.Set("RxNoiseFigure", DoubleValue(7.0));
            apPhy.Set("Antennas", UintegerValue(2));
            apPhy.Set("MaxSupportedTxSpatialStreams", UintegerValue(1));
            apPhy.Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
            mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid),
                        "EnableBeaconJitter", BooleanValue(true),
                        "BeaconInterval", TimeValue(MicroSeconds(102400)));
            apDevices = wifi.Install(apPhy, mac, apNode);
        }
        else
        {
            YansWifiPhyHelper staPhy;
            staPhy.SetChannel(yansChannel);
            staPhy.Set("ChannelSettings", StringValue("{" + std::to_string(nodes.at(apLabel).channel) + ", 80, BAND_5GHZ, 0}"));
            staPhy.Set("TxPowerStart", DoubleValue(15.0));
            staPhy.Set("TxPowerEnd", DoubleValue(15.0));
            staPhy.Set("TxGain", DoubleValue(-2.0));
            staPhy.Set("RxGain", DoubleValue(-2.0));
            staPhy.Set("RxNoiseFigure", DoubleValue(7.0));
            staPhy.Set("Antennas", UintegerValue(1));
            staPhy.Set("MaxSupportedTxSpatialStreams", UintegerValue(1));
            staPhy.Set("MaxSupportedRxSpatialStreams", UintegerValue(1));
            staDevices = wifi.Install(staPhy, mac, staNodes);
            YansWifiPhyHelper apPhy;
            apPhy.SetChannel(yansChannel);
            apPhy.Set("ChannelSettings", StringValue("{" + std::to_string(nodes.at(apLabel).channel) + ", 80, BAND_5GHZ, 0}"));
            apPhy.Set("TxPowerStart", DoubleValue(23.0102999566));
            apPhy.Set("TxPowerEnd", DoubleValue(23.0102999566));
            apPhy.Set("TxGain", DoubleValue(0.0));
            apPhy.Set("RxGain", DoubleValue(0.0));
            apPhy.Set("RxNoiseFigure", DoubleValue(7.0));
            apPhy.Set("Antennas", UintegerValue(2));
            apPhy.Set("MaxSupportedTxSpatialStreams", UintegerValue(1));
            apPhy.Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
            mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid),
                        "EnableBeaconJitter", BooleanValue(true),
                        "BeaconInterval", TimeValue(MicroSeconds(102400)));
            apDevices = wifi.Install(apPhy, mac, apNode);
        }

        nodes.at(apLabel).device = DynamicCast<WifiNetDevice>(apDevices.Get(0));
        ConfigureMac(nodes.at(apLabel).device);
        if (bss == 0)
        {
            ApplyTargetProfile(nodes.at(apLabel).device, targetProfile);
        }
        for (std::size_t i = 0; i < bssStaLabels[bss].size(); ++i)
        {
            auto& info = nodes.at(bssStaLabels[bss][i]);
            info.device = DynamicCast<WifiNetDevice>(staDevices.Get(i));
            ConfigureMac(info.device);
        }
        stream += WifiHelper::AssignStreams(apDevices, stream);
        stream += WifiHelper::AssignStreams(staDevices, stream);

        if (trafficMode == "applications")
        {
            std::ostringstream network;
            network << "10." << (bss + 1) << ".0.0";
            Ipv4AddressHelper addresses;
            addresses.SetBase(network.str().c_str(), "255.255.255.0");
            const auto apIf = addresses.Assign(apDevices);
            const auto staIf = addresses.Assign(staDevices);
            nodes.at(apLabel).address = apIf.GetAddress(0);
            for (std::size_t i = 0; i < bssStaLabels[bss].size(); ++i)
            {
                nodes.at(bssStaLabels[bss][i]).address = staIf.GetAddress(i);
            }
        }
        else
        {
            nodes.at(apLabel).device->SetReceiveCallback(MakeCallback(&DiscardFullBufferPayload));
            for (const auto& label : bssStaLabels[bss])
            {
                nodes.at(label).device->SetReceiveCallback(MakeCallback(&DiscardFullBufferPayload));
            }
        }
    }

    for (auto& [label, info] : nodes)
    {
        g_nodesForQueue.push_back(&info);
        // Summary mode still inspects PSDU starts to count per-MPDU retries;
        // expensive frame decoding and text output stay guarded in the callback.
        info.device->GetPhy(0)->TraceConnectWithoutContext("PhyTxPsduBegin",
            MakeCallback(&PhyTxTrace).Bind(info.label, info.role, info.bss));
        info.device->GetMac()->TraceConnectWithoutContext(
            "MacTx", MakeCallback(&MacArrivalTrace).Bind(info.label, info.role, info.bss));
        info.device->GetMac()->TraceConnectWithoutContext(
            "MacTxDrop", MakeCallback(&MacTxDropTrace).Bind(info.label, info.bss));
        info.device->GetMac()->TraceConnectWithoutContext(
            "AckedMpdu", MakeCallback(&AckedMpduTrace).Bind(info.label, info.bss));
        info.device->GetMac()->TraceConnectWithoutContext(
            "DroppedMpdu", MakeCallback(&DroppedMpduTrace).Bind(info.label, info.bss));
        for (AcIndex ac : {AC_VO, AC_VI, AC_BE, AC_BK})
        {
            info.device->GetMac()->GetQosTxop(ac)->TraceConnectWithoutContext(
                "CwTrace", MakeCallback(&CwTrace).Bind(info.bss, AcName(ac)));
        }
        if (g_fullTrace)
        {
            for (AcIndex ac : {AC_VO, AC_VI, AC_BE, AC_BK})
            {
                info.device->GetMac()->GetQosTxop(ac)->TraceConnectWithoutContext(
                    "BackoffTrace", MakeCallback(&BackoffTrace).Bind(info.label, info.bss, AcName(ac)));
            }
        }
        if (info.role == "AP")
        {
            PointerValue pointer;
            info.device->GetPhy(0)->GetAttribute("State", pointer);
            auto state = pointer.Get<WifiPhyStateHelper>();
            state->TraceConnectWithoutContext("State", MakeCallback(&StateTrace).Bind(info.label, info.bss));
        }
    }

    std::vector<RuntimeFlow> runtime;
    struct SaturationRuntime
    {
        std::string source;
        std::string ac;
        Ptr<MacSaturationApplication> application;
    };
    std::vector<SaturationRuntime> saturationRuntime;
    struct MacArrivalRuntime
    {
        FlowRow row;
        Ptr<MacArrivalApplication> application;
    };
    std::vector<MacArrivalRuntime> macArrivalRuntime;
    const auto inputFlows = ReadCsv(flowsPath);
    if (trafficMode == "mac-saturation")
    {
        using SaturationKey = std::pair<std::string, std::string>; // source node, AC
        std::map<SaturationKey, std::vector<std::string>> assignments;
        for (const auto& row : inputFlows)
        {
            auto& destinations = assignments[{row.at("source"), row.at("ac")}];
            if (std::find(destinations.begin(), destinations.end(), row.at("sink")) == destinations.end())
            {
                destinations.push_back(row.at("sink"));
            }
        }
        int64_t saturationStream = 700000;
        for (const auto& [key, destinationLabels] : assignments)
        {
            const auto& [sourceLabel, ac] = key;
            g_fullBufferSources.insert({sourceLabel, ac});
            std::vector<Mac48Address> destinations;
            for (const auto& label : destinationLabels)
            {
                destinations.push_back(nodes.at(label).device->GetMac()->GetAddress());
            }
            auto source = CreateObject<MacSaturationApplication>();
            source->Setup(nodes.at(sourceLabel).device, ac, destinations,
                          saturationLowWatermark, saturationHighWatermark,
                          saturationRefillUs, saturationStream);
            saturationStream += 2;
            nodes.at(sourceLabel).node->AddApplication(source);
            source->SetStartTime(Seconds(sourceStart));
            source->SetStopTime(Seconds(g_measureEnd));
            saturationRuntime.push_back({sourceLabel, ac, source});
        }
    }
    else if (trafficMode == "mac-arrival")
    {
        int64_t arrivalStream = 800000;
        auto installArrival = [&](FlowRow flow)
        {
            g_fullBufferSources.insert({flow.source, flow.ac});
            auto source = CreateObject<MacArrivalApplication>();
            source->Setup(flow,
                          nodes.at(flow.source).device,
                          nodes.at(flow.sink).device->GetMac()->GetAddress(),
                          saturationHighWatermark,
                          saturationRefillUs,
                          arrivalStream);
            arrivalStream += 8;
            nodes.at(flow.source).node->AddApplication(source);
            source->SetStartTime(Seconds(sourceStart));
            source->SetStopTime(Seconds(g_measureEnd));
            macArrivalRuntime.push_back({flow, source});
        };
        for (const auto& row : inputFlows)
        {
            FlowRow flow;
            flow.id = std::stoul(row.at("flow_id"));
            flow.streamId = row.contains("stream_id") ? std::stoul(row.at("stream_id")) : flow.id;
            flow.bss = std::stoul(row.at("bss"));
            flow.service = row.at("service");
            flow.model = row.at("model");
            flow.direction = row.at("direction");
            flow.transport = "NONE";
            flow.ac = row.at("ac");
            flow.sta = std::stoul(row.at("sta"));
            flow.source = row.at("source");
            flow.sink = row.at("sink");
            installArrival(flow);
            if (flow.model == "HTTP")
            {
                FlowRow request = flow;
                request.id = 100000 + flow.id;
                request.service = flow.service + "_REQUEST";
                request.model = "HTTP_REQ";
                request.direction = flow.direction == "DL" ? "UL_REQUEST" : "DL_REQUEST";
                std::swap(request.source, request.sink);
                installArrival(request);
            }
        }
    }
    else for (const auto& row : inputFlows)
    {
        FlowRow flow;
        flow.id = std::stoul(row.at("flow_id"));
        flow.streamId = row.contains("stream_id") ? std::stoul(row.at("stream_id")) : flow.id;
        flow.bss = std::stoul(row.at("bss"));
        flow.service = row.at("service");
        flow.model = row.at("model");
        flow.direction = row.at("direction");
        flow.transport = row.at("transport");
        flow.ac = row.at("ac");
        flow.sta = std::stoul(row.at("sta"));
        flow.source = row.at("source");
        flow.sink = row.at("sink");
        const uint16_t port = static_cast<uint16_t>(10000 + flow.id);
        PacketSinkHelper sinkHelper(flow.transport == "TCP" ? "ns3::TcpSocketFactory"
                                                             : "ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), port));
        const auto sinkApps = sinkHelper.Install(nodes.at(flow.sink).node);
        sinkApps.Start(Seconds(0.1));
        sinkApps.Stop(Seconds(g_measureEnd + drain));
        auto source = CreateObject<TgaxSourceApplication>();
        source->Setup(flow, nodes.at(flow.sink).address, port, 100000 + flow.streamId * 8);
        nodes.at(flow.source).node->AddApplication(source);
        source->SetStartTime(Seconds(g_measureStart));
        source->SetStopTime(Seconds(g_measureEnd));
        runtime.push_back({flow, port, source, DynamicCast<PacketSink>(sinkApps.Get(0))});

        if (flow.model == "HTTP")
        {
            FlowRow request = flow;
            request.id = 100000 + flow.id;
            request.service = "D4_REQUEST";
            request.model = "HTTP_REQ";
            request.direction = "UL_REQUEST";
            std::swap(request.source, request.sink);
            const uint16_t requestPort = static_cast<uint16_t>(30000 + flow.id);
            PacketSinkHelper requestSink("ns3::TcpSocketFactory",
                                         InetSocketAddress(Ipv4Address::GetAny(), requestPort));
            const auto requestSinkApps = requestSink.Install(nodes.at(request.sink).node);
            requestSinkApps.Start(Seconds(0.1));
            requestSinkApps.Stop(Seconds(g_measureEnd + drain));
            auto requestSource = CreateObject<TgaxSourceApplication>();
            requestSource->Setup(request,
                                 nodes.at(request.sink).address,
                                 requestPort,
                                 900000 + flow.streamId * 8);
            nodes.at(request.source).node->AddApplication(requestSource);
            requestSource->SetStartTime(Seconds(g_measureStart));
            requestSource->SetStopTime(Seconds(g_measureEnd));
            runtime.push_back({request,
                               requestPort,
                               requestSource,
                               DynamicCast<PacketSink>(requestSinkApps.Get(0))});
        }
    }

    std::ofstream edcaRuntime(outputPath + "/edca_runtime.csv");
    edcaRuntime << "bss,role,node,ac,aifsn,cwmin,cwmax,txop_us,queue_max_size,queue_max_delay_ms,max_ampdu_bytes,max_amsdu_bytes,ba_buffer_mpdu,frame_retry_limit\n";
    for (const auto& [bss, apLabel] : bssApLabel)
    {
        for (const auto& label : {apLabel, bssStaLabels.at(bss).front()})
        {
            const auto& info = nodes.at(label);
            for (AcIndex ac : {AC_VO, AC_VI, AC_BE, AC_BK})
            {
                const auto txop = info.device->GetMac()->GetQosTxop(ac);
                const auto queue = txop->GetWifiMacQueue();
                edcaRuntime << bss << ',' << info.role << ',' << label << ',' << AcName(ac)
                            << ',' << +txop->GetAifsn(0) << ',' << txop->GetMinCw(0)
                            << ',' << txop->GetMaxCw(0) << ',' << txop->GetTxopLimit(0).GetMicroSeconds()
                            << ',' << queue->GetMaxSize() << ',' << queue->GetMaxDelay().GetMilliSeconds()
                            << ",65535,0,64,10\n";
            }
        }
    }

    FlowMonitorHelper flowMonitorHelper;
    Ptr<FlowMonitor> flowMonitor;
    if (g_fullTrace && trafficMode == "applications")
    {
        flowMonitor = flowMonitorHelper.InstallAll();
    }
    Simulator::Schedule(Seconds(0.1), &SampleQueues);
    Simulator::Stop(Seconds(g_measureEnd + drain));
    Simulator::Run();

    // Packets admitted during the measurement window that have neither been
    // acknowledged nor finally dropped by the end of the drain window are a
    // separate censored outcome.  They must not be silently counted as delay
    // successes or as failures.
    for (const auto& [key, pending] : g_macPending)
    {
        auto& outcome = g_macOutcomes[{pending.bss, pending.direction, pending.ac}];
        ++outcome.unfinished;
        outcome.unfinishedBytes += pending.bytes;
        outcome.txAttempts += pending.txAttempts;
        outcome.retransmissions += pending.txAttempts > 0 ? pending.txAttempts - 1 : 0;
        if (const NodeAcKey source{key.first, pending.ac}; g_fullBufferSources.contains(source))
        {
            ++g_fullBufferSourceStats[source].unfinished;
        }
    }

    if (flowMonitor)
    {
        flowMonitor->CheckForLostPackets();
        flowMonitor->SerializeToXmlFile(outputPath + "/flowmon.xml", true, true);
    }

    std::map<uint16_t, const RuntimeFlow*> runtimeByPort;
    for (const auto& item : runtime)
    {
        runtimeByPort[item.port] = &item;
    }
    std::ofstream ipFlowSummary(outputPath + "/ip_flow_summary.csv");
    ipFlowSummary << "flow_monitor_id,app_flow_id,bss,service,direction,transport,ac,source_ip,destination_ip,source_port,destination_port,tx_packets,rx_packets,lost_packets,tx_bytes,rx_bytes,mean_delay_ms,mean_jitter_ms\n";
    if (flowMonitor)
    {
        const auto classifier = DynamicCast<Ipv4FlowClassifier>(flowMonitorHelper.GetClassifier());
        for (const auto& [flowId, stats] : flowMonitor->GetFlowStats())
        {
            const auto tuple = classifier->FindFlow(flowId);
            const auto found = runtimeByPort.find(tuple.destinationPort);
            if (found == runtimeByPort.end())
            {
                continue; // reverse TCP ACK/control flow; MAC trace classifies it separately
            }
            const auto& item = *found->second;
            ipFlowSummary << flowId << ',' << item.row.id << ',' << item.row.bss << ','
                          << item.row.service << ',' << item.row.direction << ',' << item.row.transport
                          << ',' << item.row.ac << ',' << tuple.sourceAddress << ','
                          << tuple.destinationAddress << ',' << tuple.sourcePort << ','
                          << tuple.destinationPort << ',' << stats.txPackets << ',' << stats.rxPackets
                          << ',' << stats.lostPackets << ',' << stats.txBytes << ',' << stats.rxBytes
                          << ',' << (stats.rxPackets ? stats.delaySum.GetSeconds() * 1000.0 / stats.rxPackets : 0.0)
                          << ',' << (stats.rxPackets > 1 ? stats.jitterSum.GetSeconds() * 1000.0 / (stats.rxPackets - 1) : 0.0)
                          << '\n';
        }
    }

    std::ofstream flowSummary(outputPath + "/flow_summary.csv");
    flowSummary << "flow_id,bss,service,model,direction,transport,ac,source,sink,port,offered_bytes,received_bytes,delivery_fraction,throughput_mbps\n";
    struct BssResult { uint64_t offered{}; uint64_t received{}; };
    std::map<uint32_t, BssResult> totals;
    for (const auto& item : runtime)
    {
        const uint64_t offered = item.source->GetOfferedBytes();
        const uint64_t received = item.sink->GetTotalRx();
        totals[item.row.bss].offered += offered;
        totals[item.row.bss].received += received;
        flowSummary << item.row.id << ',' << item.row.bss << ',' << item.row.service << ','
                    << item.row.model << ',' << item.row.direction << ',' << item.row.transport
                    << ',' << item.row.ac << ',' << item.row.source << ',' << item.row.sink << ','
                    << item.port << ',' << offered << ',' << received << ','
                    << (offered ? static_cast<double>(received) / offered : 0.0) << ','
                    << received * 8.0 / duration / 1e6 << '\n';
    }
    if (directMac)
    {
        uint32_t summaryId = 0;
        for (const auto& [key, outcome] : g_macOutcomes)
        {
            const auto& [bss, direction, ac] = key;
            totals[bss].offered += outcome.arrivedBytes;
            totals[bss].received += outcome.ackedBytes;
            flowSummary << summaryId++ << ',' << bss << ','
                        << (trafficMode == "mac-saturation" ? "MAC_SAT" : "MAC_ARRIVAL")
                        << ",MAC," << direction
                        << ",NONE," << ac << ",MAC,MAC,0," << outcome.arrivedBytes << ','
                        << outcome.ackedBytes << ','
                        << (outcome.arrivedBytes ? static_cast<double>(outcome.ackedBytes) / outcome.arrivedBytes : 0.0)
                        << ',' << outcome.ackedBytes * 8.0 / duration / 1e6 << '\n';
        }
    }

    std::ofstream bssSummary(outputPath + "/bss_summary.csv");
    bssSummary << "bss,offered_bytes,received_bytes,delivery_fraction,throughput_mbps,busy_fraction,mean_queue_packets,max_queue_packets\n";
    for (const auto& [bss, ap] : bssApLabel)
    {
        const auto result = totals[bss];
        const auto queues = g_queueAggregate[bss];
        bssSummary << bss << ',' << result.offered << ',' << result.received << ','
                   << (result.offered ? static_cast<double>(result.received) / result.offered : 0.0)
                   << ',' << result.received * 8.0 / duration / 1e6 << ','
                   << g_busySeconds[bss] / duration << ','
                   << (queues.samples ? static_cast<double>(queues.totalPackets) / queues.samples : 0.0)
                   << ',' << queues.maximum << '\n';
    }

    std::ofstream queueSummary(outputPath + "/queue_summary.csv");
    queueSummary << "bss,direction,ac,samples,mean_packets,max_packets\n";
    for (const auto& [key, queues] : g_queueByDirectionAc)
    {
        const auto& [bss, direction, ac] = key;
        queueSummary << bss << ',' << direction << ',' << ac << ',' << queues.samples << ','
                     << (queues.samples ? static_cast<double>(queues.totalPackets) / queues.samples : 0.0)
                     << ',' << queues.maximum << '\n';
    }

    using ScopeKey = std::tuple<std::string, std::string, std::string>; // scope, direction, AC
    std::map<ScopeKey, MacOutcome> scopeOutcomes;
    for (const auto& [key, outcome] : g_macOutcomes)
    {
        const auto& [bss, direction, ac] = key;
        MergeOutcome(scopeOutcomes[{bss == 0 ? "TARGET" : "NEIGHBORS", direction, ac}], outcome);
        MergeOutcome(scopeOutcomes[{"ALL", direction, ac}], outcome);
    }

    std::ofstream outcomeSummary(outputPath + "/mac_qos_outcomes.csv");
    outcomeSummary << "scope,bss,direction,ac,arrived,acked,failed,unfinished,ack_fraction,failure_fraction,unfinished_fraction,tx_attempts,retransmissions,failed_enqueue,expired_lifetime,retry_limit,qos_old_packet,other_failure,arrived_bytes,acked_bytes,failed_bytes,unfinished_bytes,deadline_eligible,on_time_delivered,on_time_rate\n";
    auto writeOutcome = [&](const std::string& scope,
                            const std::string& bss,
                            const std::string& direction,
                            const std::string& ac,
                            const MacOutcome& outcome)
    {
        const double denominator = outcome.arrived ? static_cast<double>(outcome.arrived) : 1.0;
        outcomeSummary << scope << ',' << bss << ',' << direction << ',' << ac << ','
                       << outcome.arrived << ',' << outcome.acked << ',' << outcome.failed << ','
                       << outcome.unfinished << ',' << outcome.acked / denominator << ','
                       << outcome.failed / denominator << ',' << outcome.unfinished / denominator << ','
                       << outcome.txAttempts << ',' << outcome.retransmissions << ','
                       << outcome.failedEnqueue << ',' << outcome.expiredLifetime << ','
                       << outcome.retryLimit << ',' << outcome.qosOldPacket << ','
                       << outcome.otherFailure << ',' << outcome.arrivedBytes << ','
                       << outcome.ackedBytes << ',' << outcome.failedBytes << ','
                       << outcome.unfinishedBytes << ',' << outcome.deadlineEligible << ','
                       << outcome.onTimeDelivered << ','
                       << (outcome.deadlineEligible
                               ? static_cast<double>(outcome.onTimeDelivered) / outcome.deadlineEligible
                               : 0.0)
                       << '\n';
    };
    for (const auto& [key, outcome] : g_macOutcomes)
    {
        const auto& [bss, direction, ac] = key;
        writeOutcome("BSS", std::to_string(bss), direction, ac, outcome);
    }
    for (const auto& [key, outcome] : scopeOutcomes)
    {
        const auto& [scope, direction, ac] = key;
        writeOutcome(scope, scope == "TARGET" ? "0" : "", direction, ac, outcome);
    }

    std::ofstream delaySummary(outputPath + "/mac_service_delay_summary.csv");
    delaySummary << "scope,bss,direction,ac,successful_packets,mean_ms,p50_ms,p95_ms,p99_ms,max_ms\n";
    std::ofstream delayCcdf(outputPath + "/mac_service_delay_ccdf.csv");
    delayCcdf << "scope,bss,direction,ac,delay_ms,ccdf\n";
    auto writeDelay = [&](const std::string& scope,
                          const std::string& bss,
                          const std::string& direction,
                          const std::string& ac,
                          MacOutcome outcome)
    {
        auto& values = outcome.serviceDelayMs;
        std::sort(values.begin(), values.end());
        const double mean = values.empty() ? 0.0
                                           : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        delaySummary << scope << ',' << bss << ',' << direction << ',' << ac << ','
                     << values.size() << ',' << mean << ',' << Percentile(values, 0.50) << ','
                     << Percentile(values, 0.95) << ',' << Percentile(values, 0.99) << ','
                     << (values.empty() ? 0.0 : values.back()) << '\n';
        if (!values.empty())
        {
            const std::size_t stride = std::max<std::size_t>(1, (values.size() + 1999) / 2000);
            for (std::size_t index = 0; index < values.size(); index += stride)
            {
                delayCcdf << scope << ',' << bss << ',' << direction << ',' << ac << ','
                          << values[index] << ','
                          << static_cast<double>(values.size() - index) / values.size() << '\n';
            }
            if ((values.size() - 1) % stride != 0)
            {
                delayCcdf << scope << ',' << bss << ',' << direction << ',' << ac << ','
                          << values.back() << ',' << 1.0 / values.size() << '\n';
            }
        }
    };
    for (const auto& [key, outcome] : g_macOutcomes)
    {
        const auto& [bss, direction, ac] = key;
        writeDelay("BSS", std::to_string(bss), direction, ac, outcome);
    }
    for (const auto& [key, outcome] : scopeOutcomes)
    {
        const auto& [scope, direction, ac] = key;
        writeDelay(scope, scope == "TARGET" ? "0" : "", direction, ac, outcome);
    }

    std::ofstream tcpAckSummary(outputPath + "/tcp_ack_ac_summary.csv");
    tcpAckSummary << "bss,direction,ac,unique_tcp_ack_mpdus\n";
    for (const auto& [key, count] : g_tcpAckMpdus)
    {
        const auto& [bss, direction, ac] = key;
        tcpAckSummary << bss << ',' << direction << ',' << ac << ',' << count << '\n';
    }

    std::ofstream saturationSummary(outputPath + "/fullbuffer_sources.csv");
    saturationSummary << "source,ac,low_watermark,high_watermark,refill_us,injected_packets,injected_payload_bytes,refill_events\n";
    for (const auto& item : saturationRuntime)
    {
        saturationSummary << item.source << ',' << item.ac << ',' << saturationLowWatermark << ','
                          << saturationHighWatermark << ',' << saturationRefillUs << ','
                          << item.application->GetInjectedPackets() << ','
                          << item.application->GetInjectedPayloadBytes() << ','
                          << item.application->GetRefillEvents() << '\n';
    }
    std::ofstream saturationAudit(outputPath + "/fullbuffer_audit.csv");
    saturationAudit << "traffic_mode,ipv4_stack_installed,active_sources,duplicate_mac_admissions\n"
                    << trafficMode << ',' << (trafficMode == "applications") << ','
                    << (trafficMode == "mac-saturation" ? saturationRuntime.size() : g_fullBufferSources.size())
                    << ',' << g_duplicateAdmissions << '\n';
    std::ofstream arrivalFlows(outputPath + "/direct_arrival_flows.csv");
    arrivalFlows << "flow_id,bss,service,model,direction,transport,ac,source,sink,generated_events,generated_bytes,injected_packets,injected_bytes,pending_packets,pending_bytes,maximum_pending_bytes\n";
    for (const auto& item : macArrivalRuntime)
    {
        arrivalFlows << item.row.id << ',' << item.row.bss << ',' << item.row.service << ','
                     << item.row.model << ',' << item.row.direction << ",NONE," << item.row.ac
                     << ',' << item.row.source << ',' << item.row.sink << ','
                     << item.application->GetGeneratedEvents() << ','
                     << item.application->GetGeneratedBytes() << ','
                     << item.application->GetInjectedPackets() << ','
                     << item.application->GetInjectedBytes() << ','
                     << item.application->GetPendingPackets() << ','
                     << item.application->GetPendingBytes() << ','
                     << item.application->GetMaximumPendingBytes() << '\n';
    }
    std::ofstream arrivalAudit(outputPath + "/direct_arrival_audit.csv");
    uint64_t generatedBytes{}, injectedBytes{}, pendingBytes{};
    for (const auto& item : macArrivalRuntime)
    {
        generatedBytes += item.application->GetGeneratedBytes();
        injectedBytes += item.application->GetInjectedBytes();
        pendingBytes += item.application->GetPendingBytes();
    }
    arrivalAudit << "traffic_mode,ipv4_stack_installed,configured_flows,node_ac_sources,generated_bytes,injected_bytes,pending_bytes,byte_conservation,duplicate_mac_admissions\n"
                 << trafficMode << ',' << (trafficMode == "applications") << ','
                 << macArrivalRuntime.size() << ',' << g_fullBufferSources.size() << ','
                 << generatedBytes << ',' << injectedBytes << ',' << pendingBytes << ','
                 << (generatedBytes == injectedBytes + pendingBytes) << ',' << g_duplicateAdmissions << '\n';
    std::ofstream sourceOutcomes(outputPath + "/fullbuffer_source_outcomes.csv");
    sourceOutcomes << "source,ac,arrived,acked,failed,unfinished,acked_payload_bytes\n";
    for (const auto& source : g_fullBufferSources)
    {
        const auto& stats = g_fullBufferSourceStats[source];
        sourceOutcomes << source.first << ',' << source.second << ',' << stats.arrived << ','
                       << stats.acked << ',' << stats.failed << ',' << stats.unfinished << ','
                       << stats.ackedBytes << '\n';
    }
    std::ofstream sourceQueues(outputPath + "/fullbuffer_queue_stats.csv");
    sourceQueues << "source,ac,samples,nonempty_samples,nonempty_fraction,mean_packets,min_packets,max_packets\n";
    for (const auto& source : g_fullBufferSources)
    {
        const auto& stats = g_fullBufferQueueStats[source];
        sourceQueues << source.first << ',' << source.second << ',' << stats.samples << ','
                     << stats.nonempty << ','
                     << (stats.samples ? static_cast<double>(stats.nonempty) / stats.samples : 0.0) << ','
                     << (stats.samples ? static_cast<double>(stats.totalPackets) / stats.samples : 0.0) << ','
                     << (stats.minimum == std::numeric_limits<uint64_t>::max() ? 0 : stats.minimum) << ','
                     << stats.maximum << '\n';
    }
    std::ofstream ampduSummary(outputPath + "/ampdu_summary.csv");
    ampduSummary << "ac,psdus,ampdu_psdus,ampdu_fraction,total_mpdus,mean_mpdus_per_psdu,max_mpdus_per_psdu\n";
    for (const auto& ac : {std::string("VO"), std::string("VI"), std::string("BE"), std::string("BK")})
    {
        const auto& stats = g_ampduStats[ac];
        ampduSummary << ac << ',' << stats.psdus << ',' << stats.ampduPsdus << ','
                     << (stats.psdus ? static_cast<double>(stats.ampduPsdus) / stats.psdus : 0.0) << ','
                     << stats.mpdus << ','
                     << (stats.psdus ? static_cast<double>(stats.mpdus) / stats.psdus : 0.0) << ','
                     << stats.maximumMpdus << '\n';
    }
    std::ofstream cwSummary(outputPath + "/cw_summary.csv");
    cwSummary << "ac,observed_values,min_cw,max_cw,distinct_values\n";
    for (const auto& ac : {std::string("VO"), std::string("VI"), std::string("BE"), std::string("BK")})
    {
        const auto& values = g_cwValues[ac];
        std::ostringstream joined;
        bool first = true;
        for (const auto value : values)
        {
            if (!first) joined << ';';
            joined << value;
            first = false;
        }
        cwSummary << ac << ',' << joined.str() << ','
                  << (values.empty() ? 0 : *values.begin()) << ','
                  << (values.empty() ? 0 : *values.rbegin()) << ',' << values.size() << '\n';
    }

    std::ofstream associations(outputPath + "/associations.csv");
    associations << "node,bss,associated,actual_bssid,expected_bssid,match\n";
    for (const auto& [label, info] : nodes)
    {
        if (info.role != "STA") continue;
        const auto sta = DynamicCast<StaWifiMac>(info.device->GetMac());
        const auto expected = nodes.at(bssApLabel.at(info.bss)).device->GetMac()->GetAddress();
        associations << label << ',' << info.bss << ',' << sta->IsAssociated() << ','
                     << sta->GetBssid(0) << ',' << expected << ',' << (sta->GetBssid(0) == expected) << '\n';
    }

    Simulator::Destroy();
    return 0;
}
