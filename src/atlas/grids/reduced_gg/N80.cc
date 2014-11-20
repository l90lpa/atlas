// TL159

#include "atlas/grids/reduced_gg/Grids.h"

namespace atlas {
namespace grids {
namespace reduced_gg {

void N80::construct()
{
  int N=80;
  int lon[] = {
     18,
     25,
     36,
     40,
     45,
     54,
     60,
     64,
     72,
     72,
     80,
     90,
     96,
    100,
    108,
    120,
    120,
    128,
    135,
    144,
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
    200,
    216,
    216,
    216,
    225,
    225,
    240,
    240,
    240,
    256,
    256,
    256,
    256,
    288,
    288,
    288,
    288,
    288,
    288,
    288,
    288,
    288,
    300,
    300,
    300,
    300,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320,
    320
  };
  double colat[] = {
    1.498331257266034E-002,
    3.439295439899631E-002,
    5.391722064224141E-002,
    7.346738541889221E-002,
    9.302737410765824E-002,
    0.112592116597086,
    0.132159515767881,
    0.151728548790336,
    0.171298657788255,
    0.190869512755201,
    0.210440906045689,
    0.230012700497262,
    0.249584801873994,
    0.269157143296792,
    0.288729675990173,
    0.308302363545950,
    0.327875178237581,
    0.347448098578188,
    0.367021107658770,
    0.386594191990574,
    0.406167340681772,
    0.425740544841005,
    0.445313797138077,
    0.464887091475519,
    0.484460422739714,
    0.504033786609954,
    0.523607179410301,
    0.543180597993457,
    0.562754039648841,
    0.582327502029213,
    0.601900983091582,
    0.621474481049300,
    0.641047994332901,
    0.660621521557921,
    0.680195061498252,
    0.699768613063976,
    0.719342175282811,
    0.738915747284504,
    0.758489328287636,
    0.778062917588421,
    0.797636514551148,
    0.817210118599993,
    0.836783729211987,
    0.856357345910933,
    0.875930968262149,
    0.895504595867887,
    0.915078228363351,
    0.934651865413195,
    0.954225506708470,
    0.973799151963919,
    0.993372800915602,
    1.01294645331879,
    1.03252010894608,
    1.05209376758575,
    1.07166742904028,
    1.09124109312498,
    1.11081475966685,
    1.13038842850349,
    1.14996209948213,
    1.16953577245875,
    1.18910944729731,
    1.20868312386904,
    1.22825680205175,
    1.24783048172927,
    1.26740416279088,
    1.28697784513084,
    1.30655152864788,
    1.32612521324481,
    1.34569889882814,
    1.36527258530766,
    1.38484627259614,
    1.40441996060899,
    1.42399364926395,
    1.44356733848083,
    1.46314102818122,
    1.48271471828820,
    1.50228840872616,
    1.52186209942049,
    1.54143579029739,
    1.56100948128359
  };
  setup_colat_hemisphere(N,lon,colat,RAD);
}

} // namespace reduced_gg
} // namespace grids
} // namespace atlas
