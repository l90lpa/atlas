/*
 * (C) Copyright 1996-2013 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include <cmath>

#include "eckit/log/Log.h"
#include "eckit/value/Value.h"
#include "eckit/geometry/RotateGrid.h"

#include "atlas/GridSpec.h"
#include "atlas/grids/RotatedGrid.h"

using namespace eckit;
using namespace eckit::geometry;
using namespace std;

namespace atlas {
namespace grids {

//------------------------------------------------------------------------------------------------------

RotatedGrid::RotatedGrid(Grid *grid,
                         double south_pole_latitude,
                         double south_pole_longitude,
                         double south_pole_rotation_angle):
    grid_(grid),
    south_pole_latitude_(south_pole_latitude),
    south_pole_longitude_(south_pole_longitude),
    south_pole_rotation_angle_(south_pole_rotation_angle) {

}


RotatedGrid::~RotatedGrid() {
}

std::string RotatedGrid::shortName() const {

    if ( shortName_.empty() ) {
        shortName_ = "rotated." + grid_->shortName();
    }
    return shortName_;
}

void RotatedGrid::hash(eckit::MD5& md5) const {

  md5.add("rotated.");

  grid_->hash(md5);

  md5.add(south_pole_longitude_);
  md5.add(south_pole_latitude_);
  md5.add(south_pole_rotation_angle_);
}

BoundBox RotatedGrid::bounding_box() const {
    NOTIMP;
}

size_t RotatedGrid::npts() const {
    return grid_->npts();
}


void RotatedGrid::lonlat( double pts[] ) const {
    std::vector<Grid::Point> gpts;
    gpts.resize( npts() );
    lonlat(gpts);

    int c(0);
    for (size_t i = 0; i < gpts.size(); i++) {
        pts[c++] = gpts[i].lon();
        pts[c++] = gpts[i].lat();
    }
}

void RotatedGrid::lonlat( std::vector<double> &v ) const {
    NOTIMP;
}

void RotatedGrid::lonlat( std::vector<Point> &pts) const {
    RotateGrid rotgrid(Grid::Point(south_pole_longitude_, south_pole_latitude_), south_pole_rotation_angle_);
    pts.resize( npts() );

    grid_->lonlat(pts);

    for (size_t i = 0; i < pts.size(); i++) {
        pts[i] = rotgrid.unrotate( Grid::Point( pts[i].lon(), pts[i].lat() ) );
    }
}

std::string RotatedGrid::grid_type() const {
    NOTIMP;
}

GridSpec RotatedGrid::spec() const {
    NOTIMP;
}


//------------------------------------------------------------------------------------------------------

} // namespace grids
} // namespace atlas