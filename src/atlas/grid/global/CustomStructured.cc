/*
 * (C) Copyright 1996-2016 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include <typeinfo>
#include <string>
#include "eckit/memory/Builder.h"
#include "eckit/memory/Factory.h"
#include "atlas/grid/global/CustomStructured.h"

using eckit::Factory;
using eckit::MD5;
using eckit::BadParameter;

namespace atlas {
namespace grid {
namespace global {

//------------------------------------------------------------------------------

register_BuilderT1(Grid, CustomStructured, CustomStructured::grid_type_str());

std::string CustomStructured::className()
{
  return "atlas.global.CustomStructured";
}

CustomStructured::CustomStructured(const eckit::Parametrisation& params)
  : Structured()
{
  setup(params);

  //if( ! params.get("short_name",shortName_) ) throw BadParameter("short_name missing in Params",Here());
  //if( ! params.has("hash") ) throw BadParameter("hash missing in Params",Here());
}

void CustomStructured::setup(const eckit::Parametrisation& params)
{
  eckit::ValueList list;

  std::vector<long> pl;
  std::vector<double> latitudes;
  std::vector<double> lonmin;

  if( ! params.get("pl",pl) )
    throw BadParameter("pl missing in Params",Here());
  if( ! params.get("latitudes",latitudes) )
    throw BadParameter("latitudes missing in Params",Here());

  // Optionally specify N identifier
  params.get("N",Structured::N_);

  if( params.get("lon_min", lonmin) )
    Structured::setup(latitudes.size(),latitudes.data(),pl.data(),lonmin.data());
  else
    Structured::setup(latitudes.size(),latitudes.data(),pl.data());
}

CustomStructured::CustomStructured(
    size_t nlat,
    const double lats[],
    const long nlons[])
  : Structured()
{
  Structured::setup(nlat,lats,nlons);
}

CustomStructured::CustomStructured(
    size_t nlat,
    const double latitudes[],
    const long pl[],
    const double lonmin[] )
  : Structured()
{
  Structured::setup(nlat,latitudes,pl,lonmin);
}

eckit::Properties CustomStructured::spec() const
{
  eckit::Properties grid_spec;

  grid_spec.set("grid_type",gridType());
  grid_spec.set("short_name",shortName());

  grid_spec.set("nlat",nlat());

  grid_spec.set("latitudes", eckit::makeVectorValue(latitudes()));
  grid_spec.set("pl",        eckit::makeVectorValue(pl()));
  grid_spec.set("lon_min",   eckit::makeVectorValue(lonmin_));

  BoundBox bbox = boundingBox();
  grid_spec.set("bbox_s", bbox.min().lat());
  grid_spec.set("bbox_w", bbox.min().lon());
  grid_spec.set("bbox_n", bbox.max().lat());
  grid_spec.set("bbox_e", bbox.max().lon());

  if( N() != 0 )
    grid_spec.set("N", N() );

  return grid_spec;
}

extern "C"
{

Structured* atlas__grid__global__CustomStructured_int(size_t nlat, double lats[], int pl[])
{
  std::vector<long> pl_vector;
  pl_vector.assign(pl,pl+nlat);
  return new CustomStructured(nlat,lats,pl_vector.data());
}
Structured* atlas__grid__global__CustomStructured_long(size_t nlat, double lats[], long pl[])
{
  return new CustomStructured(nlat,lats,pl);
}

Structured* atlas__grid__global__CustomStructured_lonmin_int(size_t nlat, double lats[], int pl[], double lonmin[])
{
  std::vector<long> pl_vector;
  pl_vector.assign(pl,pl+nlat);
  return new CustomStructured(nlat,lats,pl_vector.data(),lonmin);
}

Structured* atlas__grid__global__CustomStructured_lonmin_long(size_t nlat, double lats[], long pl[], double lonmin[])
{
  return new CustomStructured(nlat,lats,pl,lonmin);
}


}


} // namespace global
} // namespace grid
} // namespace atlas
