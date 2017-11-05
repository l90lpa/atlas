#pragma once

/*
 * (C) Copyright 1996-2017 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */
#include "atlas/array/SVector.h"
#include "atlas/array/ArrayView.h"


namespace atlas {
namespace parallel {

template<int ParallelDim, int Cnt, int CurrentDim>
struct halo_packer_impl {
    template<typename DATA_TYPE, int RANK, typename ... Idx>
    ATLAS_HOST_DEVICE
    static void apply(size_t& buf_idx, const size_t node_idx, const array::ArrayView<DATA_TYPE, RANK, array::Intent::ReadOnly>& field,
                      array::SVector<DATA_TYPE>& send_buffer, Idx... idxs) {
      for( size_t i=0; i< field.template shape<CurrentDim>(); ++i ) {
        halo_packer_impl<ParallelDim, Cnt-1, CurrentDim+1>::apply(buf_idx, node_idx, field, send_buffer, idxs..., i);
      }
    }
};

template<int ParallelDim, int Cnt>
struct halo_packer_impl<ParallelDim, Cnt, ParallelDim>
{
    template<typename DATA_TYPE, int RANK, typename ... Idx>
    ATLAS_HOST_DEVICE
    static void apply(size_t& buf_idx, const size_t node_idx, const array::ArrayView<DATA_TYPE, RANK, array::Intent::ReadOnly>& field,
                      array::SVector<DATA_TYPE>& send_buffer, Idx... idxs) {
      halo_packer_impl<ParallelDim, Cnt-1, ParallelDim+1>::apply(buf_idx, node_idx, field, send_buffer, idxs...,node_idx);
    }
};

template<int ParallelDim, int CurrentDim>
struct halo_packer_impl<ParallelDim, 0, CurrentDim> {
    template<typename DATA_TYPE, int RANK, typename ...Idx>
    ATLAS_HOST_DEVICE
    static void apply(size_t& buf_idx, size_t node_idx, const array::ArrayView<DATA_TYPE, RANK, array::Intent::ReadOnly>& field,
                     array::SVector<DATA_TYPE>& send_buffer, Idx...idxs)
    {
      send_buffer[buf_idx++] = field(idxs...);
    }
};

template<int ParallelDim, int Cnt, int CurrentDim>
struct halo_unpacker_impl {
    template<typename DATA_TYPE, int RANK, typename ... Idx>
    ATLAS_HOST_DEVICE
    static void apply(size_t& buf_idx, const size_t node_idx, array::SVector<DATA_TYPE> const & recv_buffer,
                      array::ArrayView<DATA_TYPE, RANK>& field, Idx... idxs) {
        for( size_t i=0; i< field.template shape<CurrentDim>(); ++i ) {
            halo_unpacker_impl<ParallelDim, Cnt-1, CurrentDim+1>::apply(buf_idx, node_idx, recv_buffer, field, idxs..., i);
        }
    }
};

template<int ParallelDim, int Cnt>
struct halo_unpacker_impl<ParallelDim, Cnt, ParallelDim> {
    template<typename DATA_TYPE, int RANK, typename ... Idx>
    ATLAS_HOST_DEVICE
    static void apply(size_t& buf_idx, const size_t node_idx, array::SVector<DATA_TYPE> const & recv_buffer,
                      array::ArrayView<DATA_TYPE, RANK>& field, Idx... idxs) {
        halo_unpacker_impl<ParallelDim, Cnt-1, ParallelDim+1>::apply(buf_idx, node_idx, recv_buffer, field, idxs...,node_idx);
    }
};

template<int ParallelDim, int CurrentDim>
struct halo_unpacker_impl<ParallelDim, 0, CurrentDim> {

    template<typename DATA_TYPE, int RANK, typename ...Idx>
    ATLAS_HOST_DEVICE
    static void apply(size_t& buf_idx, size_t node_idx, array::SVector<DATA_TYPE> const & recv_buffer,
                     array::ArrayView<DATA_TYPE, RANK>& field, Idx...idxs)
    {
      field(idxs...) = recv_buffer[buf_idx++];
    }
};


} //namespace parallel
} //namespace atlas

