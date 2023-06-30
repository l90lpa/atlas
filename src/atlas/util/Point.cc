/*
 * (C) Copyright 2013 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "atlas/util/Point.h"
#include "atlas/util/NormaliseLongitude.h"

namespace atlas {

void PointLonLat::normalise() {
    constexpr util::NormaliseLongitude normalize_from_zero;
    lon() = normalize_from_zero(lon());
}

void PointLonLat::normalise(double west) {
    util::NormaliseLongitude normalize_from_west(west);
    lon() = normalize_from_west(lon());
}

void PointLonLat::normalise(double west, double east) {
    util::NormaliseLongitude normalize_between_west_and_east(west, east);
    lon() = normalize_between_west_and_east(lon());
}

eckit::geometry::PointLonLat create_pointlonlat(const Point2 &p)
{
    return {p.X, p.Y};
}

atlas::Point2 from_pointlonlat(const PointLonLat &p)
{
    return {p.lon(), p.lat()};
}

}  // namespace atlas
