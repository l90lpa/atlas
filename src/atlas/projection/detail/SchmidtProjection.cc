/*
 * (C) Copyright 2013 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <cmath>

#include "atlas/projection/detail/SchmidtProjection.h"
#include "atlas/util/Constants.h"

namespace {
static double D2R( const double x ) {
    return atlas::util::Constants::degreesToRadians() * x;
}
static double R2D( const double x ) {
    return atlas::util::Constants::radiansToDegrees() * x;
}
}  // namespace

namespace atlas {
namespace projection {
namespace detail {

// constructor
template <typename Rotation>
SchmidtProjectionT<Rotation>::SchmidtProjectionT( const eckit::Parametrisation& params ) :
    ProjectionImpl(),
    rotation_( params ) {
    if ( !params.get( "stretching_factor", c_ ) )
        throw eckit::BadParameter( "stretching_factor missing in Params", Here() );
}

template <typename Rotation>
void SchmidtProjectionT<Rotation>::xy2lonlat( double crd[] ) const {
    // stretch
    crd[1] = R2D(
        std::asin( std::cos( 2. * std::atan( 1 / c_ * std::tan( std::acos( std::sin( D2R( crd[1] ) ) ) * 0.5 ) ) ) ) );

    // perform rotation
    rotation_.rotate( crd );
}

template <typename Rotation>
void SchmidtProjectionT<Rotation>::lonlat2xy( double crd[] ) const {
    // inverse rotation
    rotation_.unrotate( crd );

    // unstretch
    crd[1] =
        R2D( std::asin( std::cos( 2. * std::atan( c_ * std::tan( std::acos( std::sin( D2R( crd[1] ) ) ) * 0.5 ) ) ) ) );
}

// specification
template <typename Rotation>
typename SchmidtProjectionT<Rotation>::Spec SchmidtProjectionT<Rotation>::spec() const {
    Spec proj_spec;
    proj_spec.set( "type", static_type() );
    proj_spec.set( "stretching_factor", c_ );
    rotation_.spec( proj_spec );
    return proj_spec;
}

template <typename Rotation>
void SchmidtProjectionT<Rotation>::hash( eckit::Hash& hsh ) const {
    hsh.add( static_type() );
    rotation_.hash( hsh );
    hsh.add( c_ );
}

register_BuilderT1( ProjectionImpl, SchmidtProjection, SchmidtProjection::static_type() );
register_BuilderT1( ProjectionImpl, RotatedSchmidtProjection, RotatedSchmidtProjection::static_type() );

}  // namespace detail
}  // namespace projection
}  // namespace atlas
