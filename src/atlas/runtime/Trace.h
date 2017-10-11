/*
 * (C) Copyright 1996-2017 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#pragma once

#include "atlas/library/config.h"
#include "atlas/runtime/trace/TraceT.h"
#include "atlas/runtime/trace/Barriers.h"
#include "atlas/runtime/trace/Logging.h"

//-----------------------------------------------------------------------------------------------------------

/// Create scoped trace objects
///
/// Example:
///
///     void foo() {
///         ATLAS_TRACE();
///         // trace "foo" starts
///
///         /* interesting computations ... */
///
///         ATLAS_TRACE_SCOPE("bar") {
///             // trace "bar" starts
///
///             /* interesting computations ... */
///
///             // trace "bar" ends
///         }
///
///         // trace "foo" ends
///     }
///
/// Example 2:
///
///     void foo() {
///         ATLAS_TRACE("custom");
///         // trace "custom" starts
///
///         /* interesting computations ... */
///
///         // trace "custom" ends
///     }
///
#define ATLAS_TRACE(...)
#define ATLAS_TRACE_SCOPE(...)

//-----------------------------------------------------------------------------------------------------------

namespace atlas {

struct TraceTraits {
    using Barriers = runtime::trace::NoBarriers;
    using Tracing  = runtime::trace::Logging;
};

class Trace : public runtime::trace::TraceT< TraceTraits > {
    using Base = runtime::trace::TraceT< TraceTraits >;
public:
    using Base::Base;
};

} // namespace atlas


//-----------------------------------------------------------------------------------------------------------

#if ATLAS_HAVE_TRACE

#include "atlas/util/detail/BlackMagic.h"

#undef ATLAS_TRACE
#undef ATLAS_TRACE_SCOPE

#define ATLAS_TRACE(...) __ATLAS_SPLICE( __ATLAS_TRACE_, __ATLAS_ISEMPTY( __VA_ARGS__ ) ) (__VA_ARGS__)
#define __ATLAS_TRACE_1(...) __ATLAS_TYPE( ::atlas::Trace, Here() )
#define __ATLAS_TRACE_0(...) __ATLAS_TYPE( ::atlas::Trace, Here(), __VA_ARGS__ )

#define ATLAS_TRACE_SCOPE(...) __ATLAS_SPLICE( __ATLAS_TRACE_SCOPE_, __ATLAS_ISEMPTY( __VA_ARGS__ ) ) (__VA_ARGS__)
#define __ATLAS_TRACE_SCOPE_1(...) __ATLAS_TYPE_SCOPE( ::atlas::Trace, Here() )
#define __ATLAS_TRACE_SCOPE_0(...) __ATLAS_TYPE_SCOPE( ::atlas::Trace, Here(), __VA_ARGS__ )

#endif

//-----------------------------------------------------------------------------------------------------------
