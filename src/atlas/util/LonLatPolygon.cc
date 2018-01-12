/*
 * (C) Copyright 1996-2018 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include "atlas/util/LonLatPolygon.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include "atlas/util/CoordinateEnums.h"

namespace atlas {
namespace util {


//------------------------------------------------------------------------------------------------------


namespace {


double cross_product_analog(const PointLonLat& A, const PointLonLat& B, const PointLonLat& C) {
  return (A.lon() - C.lon()) * (B.lat() - C.lat())
       - (A.lat() - C.lat()) * (B.lon() - C.lon());
}


}  // (anonymous)


//------------------------------------------------------------------------------------------------------


LonLatPolygon::LonLatPolygon(
        const Polygon& poly,
        const atlas::Field& lonlat,
        bool includesNorthPole,
        bool includesSouthPole,
        bool removeAlignedPoints ) :
    PolygonCoordinates(poly, lonlat, includesNorthPole, includesSouthPole, removeAlignedPoints) {
}


LonLatPolygon::LonLatPolygon(
        const std::vector<PointLonLat>& points,
        bool includesNorthPole,
        bool includesSouthPole ) :
    PolygonCoordinates(points, includesNorthPole, includesSouthPole) {
}


bool LonLatPolygon::contains(const PointLonLat& P) const {
    ASSERT(coordinates_.size() >= 2);

    // check first bounding box
    if (coordinatesMin_.lon() <= P.lon() && P.lon() < coordinatesMax_.lon()
     && coordinatesMin_.lat() <= P.lat() && P.lat() < coordinatesMax_.lat()) {

        // winding number
        int wn = 0;

        // loop on polygon edges
        for (size_t i = 1; i < coordinates_.size(); ++i) {
            const PointLonLat& A = coordinates_[i-1];
            const PointLonLat& B = coordinates_[ i ];

            // check point-edge side and direction, using 2D-analog cross-product;
            // tests if P is left|on|right of a directed A-B infinite line, by intersecting either:
            // - "up" on upward crossing & P left of edge, or
            // - "down" on downward crossing & P right of edge
            const bool APB = (A.lat() <= P.lat() && P.lat() < B.lat());
            const bool BPA = (B.lat() <= P.lat() && P.lat() < A.lat());

            if (APB != BPA) {
                const double side = cross_product_analog(P, A, B);
                if (APB && side > 0) {
                    ++wn;
                } else if (BPA && side < 0) {
                    --wn;
                }
            }
        }

        // wn == 0 only when P is outside
        return wn != 0;
    }

    return ((includesNorthPole_ && P.lat() >= coordinatesMax_.lat())
         || (includesSouthPole_ && P.lat() <  coordinatesMin_.lat()));
}


//------------------------------------------------------------------------------------------------------


}  // util
}  // atlas

