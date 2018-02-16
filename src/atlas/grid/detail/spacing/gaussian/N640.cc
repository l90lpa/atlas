// TL1279

#include "atlas/grid/detail/spacing/gaussian/N.h"

namespace atlas {
namespace grid {
namespace spacing {
namespace gaussian {

DEFINE_GAUSSIAN_LATITUDES(
    640, LIST( 89.892396445590, 89.753004943174, 89.612790258599, 89.472389582061, 89.331918354382, 89.191412986832,
               89.050888539966, 88.910352359260, 88.769808451100, 88.629259185412, 88.488706053376, 88.348150039999,
               88.207591822004, 88.067031879651, 87.926470563186, 87.785908134041, 87.645344791296, 87.504780689222,
               87.364215949215, 87.223650668104, 87.083084924071, 86.942518780929, 86.801952291278, 86.661385498868,
               86.520818440380, 86.380251146799, 86.239683644481, 86.099115955985, 85.958548100731, 85.817980095530,
               85.677411955006, 85.536843691943, 85.396275317563, 85.255706841758, 85.115138273282, 84.974569619910,
               84.834000888572, 84.693432085462, 84.552863216136, 84.412294285589, 84.271725298330, 84.131156258431,
               83.990587169587, 83.850018035154, 83.709448858186, 83.568879641474, 83.428310387568, 83.287741098803,
               83.147171777324, 83.006602425105, 82.866033043963, 82.725463635573, 82.584894201486, 82.444324743135,
               82.303755261850, 82.163185758865, 82.022616235328, 81.882046692304, 81.741477130791, 81.600907551716,
               81.460337955946, 81.319768344292, 81.179198717514, 81.038629076323, 80.898059421388, 80.757489753335,
               80.616920072753, 80.476350380198, 80.335780676193, 80.195210961228, 80.054641235771, 79.914071500258,
               79.773501755105, 79.632932000703, 79.492362237425, 79.351792465622, 79.211222685626, 79.070652897754,
               78.930083102307, 78.789513299568, 78.648943489809, 78.508373673288, 78.367803850250, 78.227234020928,
               78.086664185545, 77.946094344312, 77.805524497433, 77.664954645100, 77.524384787497, 77.383814924802,
               77.243245057181, 77.102675184796, 76.962105307802, 76.821535426346, 76.680965540569, 76.540395650606,
               76.399825756588, 76.259255858639, 76.118685956878, 75.978116051420, 75.837546142375, 75.696976229850,
               75.556406313944, 75.415836394757, 75.275266472383, 75.134696546911, 74.994126618429, 74.853556687021,
               74.712986752768, 74.572416815746, 74.431846876032, 74.291276933698, 74.150706988813, 74.010137041445,
               73.869567091658, 73.728997139516, 73.588427185079, 73.447857228405, 73.307287269551, 73.166717308572,
               73.026147345520, 72.885577380447, 72.745007413401, 72.604437444432, 72.463867473585, 72.323297500905,
               72.182727526435, 72.042157550217, 71.901587572293, 71.761017592701, 71.620447611481, 71.479877628669,
               71.339307644300, 71.198737658411, 71.058167671035, 70.917597682205, 70.777027691952, 70.636457700309,
               70.495887707304, 70.355317712967, 70.214747717328, 70.074177720412, 69.933607722247, 69.793037722860,
               69.652467722275, 69.511897720517, 69.371327717611, 69.230757713579, 69.090187708444, 68.949617702229,
               68.809047694955, 68.668477686643, 68.527907677314, 68.387337666986, 68.246767655681, 68.106197643416,
               67.965627630209, 67.825057616080, 67.684487601045, 67.543917585122, 67.403347568326, 67.262777550675,
               67.122207532184, 66.981637512868, 66.841067492743, 66.700497471823, 66.559927450122, 66.419357427655,
               66.278787404436, 66.138217380477, 65.997647355791, 65.857077330392, 65.716507304291, 65.575937277502,
               65.435367250035, 65.294797221902, 65.154227193115, 65.013657163685, 64.873087133622, 64.732517102938,
               64.591947071642, 64.451377039745, 64.310807007256, 64.170236974186, 64.029666940544, 63.889096906338,
               63.748526871579, 63.607956836274, 63.467386800434, 63.326816764065, 63.186246727177, 63.045676689778,
               62.905106651876, 62.764536613478, 62.623966574592, 62.483396535226, 62.342826495387, 62.202256455083,
               62.061686414319, 61.921116373105, 61.780546331445, 61.639976289347, 61.499406246817, 61.358836203862,
               61.218266160488, 61.077696116701, 60.937126072507, 60.796556027912, 60.655985982922, 60.515415937542,
               60.374845891778, 60.234275845637, 60.093705799122, 59.953135752239, 59.812565704993, 59.671995657391,
               59.531425609435, 59.390855561132, 59.250285512486, 59.109715463502, 58.969145414185, 58.828575364539,
               58.688005314568, 58.547435264277, 58.406865213671, 58.266295162752, 58.125725111527, 57.985155059998,
               57.844585008170, 57.704014956047, 57.563444903632, 57.422874850930, 57.282304797944, 57.141734744678,
               57.001164691135, 56.860594637319, 56.720024583233, 56.579454528882, 56.438884474268, 56.298314419394,
               56.157744364265, 56.017174308882, 55.876604253250, 55.736034197372, 55.595464141249, 55.454894084887,
               55.314324028286, 55.173753971452, 55.033183914385, 54.892613857089, 54.752043799568, 54.611473741823,
               54.470903683857, 54.330333625673, 54.189763567274, 54.049193508662, 53.908623449839, 53.768053390809,
               53.627483331573, 53.486913272134, 53.346343212494, 53.205773152657, 53.065203092623, 52.924633032395,
               52.784062971976, 52.643492911368, 52.502922850573, 52.362352789593, 52.221782728430, 52.081212667086,
               51.940642605563, 51.800072543864, 51.659502481990, 51.518932419943, 51.378362357725, 51.237792295338,
               51.097222232785, 50.956652170066, 50.816082107184, 50.675512044140, 50.534941980936, 50.394371917574,
               50.253801854056, 50.113231790383, 49.972661726557, 49.832091662580, 49.691521598453, 49.550951534178,
               49.410381469756, 49.269811405190, 49.129241340479, 48.988671275628, 48.848101210635, 48.707531145504,
               48.566961080235, 48.426391014830, 48.285820949291, 48.145250883618, 48.004680817814, 47.864110751879,
               47.723540685814, 47.582970619622, 47.442400553304, 47.301830486860, 47.161260420293, 47.020690353603,
               46.880120286791, 46.739550219859, 46.598980152808, 46.458410085640, 46.317840018355, 46.177269950954,
               46.036699883439, 45.896129815811, 45.755559748071, 45.614989680220, 45.474419612259, 45.333849544190,
               45.193279476013, 45.052709407729, 44.912139339340, 44.771569270846, 44.630999202249, 44.490429133549,
               44.349859064748, 44.209288995846, 44.068718926845, 43.928148857745, 43.787578788547, 43.647008719253,
               43.506438649863, 43.365868580378, 43.225298510799, 43.084728441127, 42.944158371362, 42.803588301507,
               42.663018231561, 42.522448161525, 42.381878091401, 42.241308021188, 42.100737950889, 41.960167880503,
               41.819597810032, 41.679027739476, 41.538457668836, 41.397887598112, 41.257317527307, 41.116747456420,
               40.976177385452, 40.835607314404, 40.695037243276, 40.554467172070, 40.413897100786, 40.273327029425,
               40.132756957987, 39.992186886473, 39.851616814884, 39.711046743221, 39.570476671484, 39.429906599674,
               39.289336527791, 39.148766455836, 39.008196383811, 38.867626311714, 38.727056239548, 38.586486167313,
               38.445916095009, 38.305346022636, 38.164775950197, 38.024205877690, 37.883635805117, 37.743065732479,
               37.602495659776, 37.461925587007, 37.321355514176, 37.180785441280, 37.040215368322, 36.899645295301,
               36.759075222219, 36.618505149076, 36.477935075872, 36.337365002607, 36.196794929284, 36.056224855901,
               35.915654782459, 35.775084708960, 35.634514635403, 35.493944561788, 35.353374488118, 35.212804414391,
               35.072234340608, 34.931664266771, 34.791094192879, 34.650524118932, 34.509954044932, 34.369383970879,
               34.228813896773, 34.088243822614, 33.947673748404, 33.807103674142, 33.666533599830, 33.525963525466,
               33.385393451053, 33.244823376590, 33.104253302077, 32.963683227516, 32.823113152906, 32.682543078249,
               32.541973003543, 32.401402928791, 32.260832853991, 32.120262779145, 31.979692704254, 31.839122629316,
               31.698552554333, 31.557982479306, 31.417412404234, 31.276842329118, 31.136272253958, 30.995702178755,
               30.855132103508, 30.714562028220, 30.573991952888, 30.433421877515, 30.292851802101, 30.152281726645,
               30.011711651148, 29.871141575611, 29.730571500034, 29.590001424417, 29.449431348760, 29.308861273064,
               29.168291197330, 29.027721121557, 28.887151045746, 28.746580969896, 28.606010894010, 28.465440818086,
               28.324870742125, 28.184300666128, 28.043730590094, 27.903160514025, 27.762590437920, 27.622020361779,
               27.481450285604, 27.340880209393, 27.200310133149, 27.059740056870, 26.919169980557, 26.778599904211,
               26.638029827831, 26.497459751418, 26.356889674973, 26.216319598495, 26.075749521985, 25.935179445443,
               25.794609368869, 25.654039292264, 25.513469215628, 25.372899138962, 25.232329062264, 25.091758985537,
               24.951188908779, 24.810618831992, 24.670048755175, 24.529478678328, 24.388908601453, 24.248338524549,
               24.107768447617, 23.967198370656, 23.826628293668, 23.686058216651, 23.545488139607, 23.404918062536,
               23.264347985438, 23.123777908313, 22.983207831162, 22.842637753984, 22.702067676780, 22.561497599550,
               22.420927522295, 22.280357445014, 22.139787367708, 21.999217290377, 21.858647213022, 21.718077135642,
               21.577507058237, 21.436936980809, 21.296366903357, 21.155796825881, 21.015226748382, 20.874656670859,
               20.734086593314, 20.593516515746, 20.452946438155, 20.312376360542, 20.171806282907, 20.031236205250,
               19.890666127571, 19.750096049871, 19.609525972149, 19.468955894407, 19.328385816643, 19.187815738859,
               19.047245661054, 18.906675583229, 18.766105505383, 18.625535427518, 18.484965349633, 18.344395271729,
               18.203825193805, 18.063255115862, 17.922685037900, 17.782114959919, 17.641544881920, 17.500974803902,
               17.360404725866, 17.219834647812, 17.079264569740, 16.938694491650, 16.798124413543, 16.657554335419,
               16.516984257277, 16.376414179119, 16.235844100943, 16.095274022751, 15.954703944543, 15.814133866318,
               15.673563788078, 15.532993709821, 15.392423631549, 15.251853553261, 15.111283474957, 14.970713396639,
               14.830143318305, 14.689573239957, 14.549003161593, 14.408433083215, 14.267863004823, 14.127292926417,
               13.986722847996, 13.846152769562, 13.705582691113, 13.565012612652, 13.424442534176, 13.283872455688,
               13.143302377186, 13.002732298672, 12.862162220144, 12.721592141604, 12.581022063052, 12.440451984487,
               12.299881905910, 12.159311827321, 12.018741748721, 11.878171670108, 11.737601591484, 11.597031512849,
               11.456461434202, 11.315891355544, 11.175321276875, 11.034751198196, 10.894181119506, 10.753611040805,
               10.613040962094, 10.472470883373, 10.331900804641, 10.191330725900, 10.050760647149, 9.910190568389,
               9.769620489619, 9.629050410840, 9.488480332051, 9.347910253253, 9.207340174447, 9.066770095632,
               8.926200016808, 8.785629937976, 8.645059859135, 8.504489780287, 8.363919701430, 8.223349622565,
               8.082779543693, 7.942209464813, 7.801639385925, 7.661069307030, 7.520499228128, 7.379929149219,
               7.239359070303, 7.098788991380, 6.958218912450, 6.817648833514, 6.677078754572, 6.536508675623,
               6.395938596669, 6.255368517708, 6.114798438741, 5.974228359769, 5.833658280791, 5.693088201808,
               5.552518122820, 5.411948043826, 5.271377964827, 5.130807885824, 4.990237806815, 4.849667727802,
               4.709097648785, 4.568527569763, 4.427957490737, 4.287387411707, 4.146817332673, 4.006247253635,
               3.865677174593, 3.725107095548, 3.584537016499, 3.443966937447, 3.303396858392, 3.162826779334,
               3.022256700273, 2.881686621209, 2.741116542142, 2.600546463073, 2.459976384002, 2.319406304928,
               2.178836225852, 2.038266146774, 1.897696067694, 1.757125988613, 1.616555909530, 1.475985830445,
               1.335415751359, 1.194845672272, 1.054275593184, 0.913705514095, 0.773135435005, 0.632565355914,
               0.491995276822, 0.351425197731, 0.210855118639, 0.070285039546 ) )

}  // namespace gaussian
}  // namespace spacing
}  // namespace grid
}  // namespace atlas
