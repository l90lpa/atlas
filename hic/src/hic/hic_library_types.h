/*
 * (C) Copyright 2024- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */
#pragma once

#if !defined(HIC_NAMESPACE)
  #define HIC_NAMESPACE
  #define HIC_NAMESPACE_BEGIN
  #define HIC_NAMESPACE_END
#else
  #define HIC_NAMESPACE_BEGIN namespace HIC_NAMESPACE {
  #define HIC_NAMESPACE_END }
#endif

#if HIC_BACKEND_CUDA
  #include <library_types.h>
#elif HIC_BACKEND_HIP
  #include <hip/library_types.h>
#else
  #error Unsupported hic backend. Please define HIC_BACKEND_CUDA or HIC_BACKEND_HIP or HIC_BACKEND_DUMMY
#endif

//------------------------------------------------
HIC_NAMESPACE_BEGIN
//------------------------------------------------

#if HIC_BACKEND_CUDA
  constexpr decltype(CUDA_R_32F) HIC_R_32F = CUDA_R_32F;
  constexpr decltype(CUDA_R_64F) HIC_R_64F = CUDA_R_64F;
#elif HIC_BACKEND_HIP
  constexpr decltype(HIP_R_32F) HIC_R_32F = HIP_R_32F;
  constexpr decltype(HIP_R_64F) HIC_R_64F = HIP_R_64F;
#else
  #error Unsupported hic backend. Please define HIC_BACKEND_CUDA or HIC_BACKEND_HIP or HIC_BACKEND_DUMMY
#endif

//------------------------------------------------
HIC_NAMESPACE_END
//------------------------------------------------