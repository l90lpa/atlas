/*
 * (C) Copyright 1996-2014 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#ifndef ATLAS_MPI_MPI_h
#define ATLAS_MPI_MPI_h

#include <stdexcept>
#include <iostream>

#include "atlas/atlas_mpi.h"
#include "atlas/util/ArrayView.h"
#include "eckit/exception/Exceptions.h"

#define ATLAS_MPI_CHECK_RESULT( MPI_CALL )\
{ \
  int ierr = MPI_CALL; \
  if (ierr != MPI_SUCCESS) { \
    char errstr [MPI_MAX_ERROR_STRING]; \
    int errsize = 0; \
    MPI_Error_string(ierr,errstr,&errsize); \
    std::string mpistr( errstr, errsize ); \
    throw atlas::mpi::Error( std::string("MPI call: ") + \
      std::string(#MPI_CALL) + \
      std::string(" did not return MPI_SUCCESS:\n")+mpistr, Here() ); \
  } \
}

namespace atlas {
namespace mpi {


class Error : public eckit::Exception {
public:
  Error(const std::string& msg, const eckit::CodeLocation& loc)
  {
    eckit::StrStream s;
    s << "MPI Error: " << msg << " " << " in " << loc << " "  << eckit::StrStream::ends;
    reason(std::string(s));
  }
};


template<typename DATA_TYPE>
MPI_Datatype datatype();

template<typename DATA_TYPE>
MPI_Datatype datatype(DATA_TYPE&);


bool initialized();

bool finalized();

void init(int argc=0, char *argv[]=0);

void finalize();

class Comm
{
public:

  Comm();

  Comm( MPI_Comm );

  static Comm& instance();

  operator MPI_Comm() const { return comm_; }

  MPI_Fint fortran_handle();

  void set_with_fortran_handle( MPI_Fint );

  void set_with_C_handle( MPI_Comm );

  size_t size() const;

  size_t rank() const;

  void barrier() const;

private:
  MPI_Comm comm_;
};

class CommWorld : private Comm
{
public:
  CommWorld() : Comm(MPI_COMM_WORLD) {}
  static CommWorld& instance();
  operator MPI_Comm() const { return MPI_COMM_WORLD; }
};

size_t rank();

size_t size();

void barrier();

} // namespace mpi
} // namespace atlas

// ------------------------------------------------------------------
// C wrapper interfaces to C++ routines
extern "C"
{
  void atlas_mpi_comm_set_with_fortran_handle (int comm);
  int atlas_mpi_comm_fortran_handle ();
}
// ------------------------------------------------------------------

#endif // ATLAS_MPI_MPI_h
