// TL399

#include "atlas/grids/reduced_gg/Grids.h"

namespace atlas {
namespace grids {
namespace reduced_gg {

void N200::construct()
{
  int N=200;
  int lon[] = {
      18,
      25,
      36,
      40,
      45,
      50,
      60,
      64,
      72,
      72,
      75,
      81,
      90,
      96,
     100,
     108,
     120,
     125,
     128,
     135,
     144,
     150,
     160,
     160,
     180,
     180,
     180,
     192,
     192,
     200,
     216,
     216,
     225,
     225,
     240,
     240,
     243,
     250,
     256,
     270,
     270,
     288,
     288,
     288,
     300,
     300,
     320,
     320,
     320,
     320,
     360,
     360,
     360,
     360,
     360,
     360,
     375,
     375,
     375,
     384,
     400,
     400,
     400,
     400,
     432,
     432,
     432,
     432,
     432,
     450,
     450,
     450,
     480,
     480,
     480,
     480,
     480,
     480,
     486,
     500,
     500,
     500,
     512,
     512,
     512,
     540,
     540,
     540,
     540,
     540,
     576,
     576,
     576,
     576,
     576,
     576,
     576,
     576,
     600,
     600,
     600,
     600,
     600,
     640,
     640,
     640,
     640,
     640,
     640,
     640,
     640,
     640,
     640,
     648,
     648,
     675,
     675,
     675,
     675,
     675,
     675,
     675,
     720,
     720,
     720,
     720,
     720,
     720,
     720,
     720,
     720,
     720,
     720,
     720,
     720,
     720,
     729,
     729,
     729,
     750,
     750,
     750,
     750,
     750,
     750,
     750,
     750,
     768,
     768,
     768,
     768,
     768,
     768,
     768,
     768,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800,
     800
  };
  double lat[] = {
    89.6559642468698570,
    89.2102943916537043,
    88.7619926150752860,
    88.3130961752085710,
    87.8639741655226345,
    87.4147430020609875,
    86.9654508372179436,
    86.5161211563435444,
    86.0667667688679359,
    85.6173952520438633,
    85.1680113735615407,
    84.7186182827309864,
    84.2692181432142178,
    83.8198124905248676,
    83.3704024444936778,
    82.9209888409815079,
    82.4715723165065526,
    82.0221533643148888,
    81.5727323725375157,
    81.1233096507704943,
    80.6738854489796751,
    80.2244599711957420,
    79.7750333856012475,
    79.3256058320717869,
    78.8761774278903829,
    78.4267482721315616,
    77.9773184490627926,
    77.5278880308108143,
    77.0784570794720736,
    76.6290256487977217,
    76.1795937855497556,
    75.7301615306013787,
    75.2807289198356386,
    74.8312959848844912,
    74.3818627537401795,
    73.9324292512641392,
    73.4829954996128691,
    73.0335615185960449,
    72.5841273259792956,
    72.1346929377412351,
    71.6852583682926650,
    71.2358236306642283,
    70.7863887366680018,
    70.3369536970365061,
    69.8875185215433987,
    69.4380832191082646,
    68.9886477978879498,
    68.5392122653563405,
    68.0897766283744943,
    67.6403408932521728,
    67.1909050658021414,
    66.7414691513882303,
    66.2920331549678394,
    65.8425970811297532,
    65.3931609341278062,
    64.9437247179109391,
    64.4942884361500148,
    64.0448520922619196,
    63.5954156894312561,
    63.1459792306295924,
    62.6965427186332107,
    62.2471061560387255,
    61.7976695452775715,
    61.3482328886288499,
    60.8987961882311026,
    60.4493594460930836,
    59.9999226641034014,
    59.5504858440393576,
    59.1010489875751617,
    58.6516120962892700,
    58.2021751716710654,
    57.7527382151272164,
    57.3033012279872551,
    56.8538642115088706,
    56.4044271668826696,
    55.9549900952366244,
    55.5055529976401942,
    55.0561158751080271,
    54.6066787286034696,
    54.1572415590418288,
    53.7078043672932566,
    53.2583671541856134,
    52.8089299205069196,
    52.3594926670078706,
    51.9100553944038907,
    51.4606181033773566,
    51.0111807945793316,
    50.5617434686314908,
    50.1123061261277130,
    49.6628687676356506,
    49.2134313936981513,
    48.7639940048346929,
    48.3145566015425132,
    47.8651191842979529,
    47.4156817535573936,
    46.9662443097584941,
    46.5168068533209862,
    46.0673693846476837,
    45.6179319041253422,
    45.1684944121254688,
    44.7190569090049905,
    44.2696193951071706,
    43.8201818707620561,
    43.3707443362872453,
    42.9213067919884708,
    42.4718692381601528,
    42.0224316750859259,
    41.5729941030391430,
    41.1235565222833657,
    40.6741189330727977,
    40.2246813356527468,
    39.7752437302599446,
    39.3258061171230437,
    38.8763684964629164,
    38.4269308684929385,
    37.9774932334194588,
    37.5280555914419338,
    37.0786179427534677,
    36.6291802875407271,
    36.1797426259845949,
    35.7303049582601204,
    35.2808672845369671,
    34.8314296049794905,
    34.3819919197470512,
    33.9325542289941851,
    33.4831165328707669,
    33.0336788315222591,
    32.5842411250898678,
    32.1348034137107135,
    31.6853656975179412,
    31.2359279766409834,
    30.7864902512056204,
    30.3370525213341580,
    29.8876147871455231,
    29.4381770487554739,
    28.9887393062766527,
    28.5393015598186963,
    28.0898638094884596,
    27.6404260553899626,
    27.1909882976246067,
    26.7415505362912782,
    26.2921127714863623,
    25.8426750033038992,
    25.3932372318356840,
    24.9437994571712771,
    24.4943616793981853,
    24.0449238986018301,
    23.5954861148657571,
    23.1460483282715614,
    22.6966105388990727,
    22.2471727468263332,
    21.7977349521297477,
    21.3482971548841043,
    20.8988593551626280,
    20.4494215530370127,
    19.9999837485775522,
    19.5505459418531444,
    19.1011081329313370,
    18.6516703218783917,
    18.2022325087593515,
    17.7527946936380339,
    17.3033568765771513,
    16.8539190576382936,
    16.4044812368819741,
    15.9550434143677364,
    15.5056055901541026,
    15.0561677642986425,
    14.6067299368580965,
    14.1572921078882885,
    13.7078542774441789,
    13.2584164455800124,
    12.8089786123492146,
    12.3595407778045061,
    11.9101029419979287,
    11.4606651049807962,
    11.0112272668038411,
    10.5617894275171800,
    10.1123515871703233,
     9.6629137458122791,
     9.2134759034915206,
     8.7640380602559578,
     8.3146002161531456,
     7.8651623712301202,
     7.4157245255335607,
     6.9662866791096674,
     6.5168488320043112,
     6.0674109842630983,
     5.6179731359311544,
     5.1685352870535040,
     4.7190974376747370,
     4.2696595878392554,
     3.8202217375912566,
     3.3707838869747078,
     2.9213460360333610,
     2.4719081848109035,
     2.0224703333507437,
     1.5730324816963162,
     1.1235946298908883,
     0.6741567779776398,
     0.2247189259997370
  };
  setup_lat_hemisphere(N,lon,lat,DEG);
}

} // namespace reduced_gg
} // namespace grids
} // namespace atlas
