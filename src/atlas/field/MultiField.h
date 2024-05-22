/*
 * (C) Copyright 2013 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

/// @author Willem Deconinck
/// @date June 2015

#pragma once

#include <map>
#include <vector>

#include "atlas/array/Array.h"
#include "atlas/field/Field.h"
#include "atlas/field/FieldSet.h"
#include "atlas/util/Config.h"
#include "atlas/util/Factory.h"
#include "atlas/util/Metadata.h"
#include "atlas/util/Object.h"
#include "atlas/util/ObjectHandle.h"

namespace eckit {
class Parametrisation;
}

namespace atlas {
namespace field {

/**
 * \brief MultiField class that owns a collection of fields that are co-allocated
 *
 * Fields can only be described by parametrisation during the construction.
 * Once setup, no additional fields can be added.
 *
 * Fields have to all be of same memory layout and data type
 */

class MultiFieldImpl : public util::Object {
public:  // methods
    //-- Constructors

    MultiFieldImpl() { }

    MultiFieldImpl(const array::ArraySpec& spec, const eckit::Parametrisation& config = util::NoConfig()) {
        array::ArraySpec s(spec);
        array_.reset(array::Array::create(std::move(s), config));
    }

    virtual ~MultiFieldImpl() {}


    //-- Accessors

    const Field& field(const std::string& name) const { return fieldset_.field(name); }
    Field& field(const std::string& name) { return fieldset_.field(name); }
    bool has(const std::string& name) const { return fieldset_.has(name); }
    std::vector<std::string> field_names() const { return fieldset_.field_names(); }

    const Field& field(const idx_t idx) const { return fieldset_[idx]; }
    Field& field(const idx_t idx) { return fieldset_[idx]; }
    idx_t size() const { return fieldset_.size(); }

    const Field& operator[](const idx_t idx) const { return fieldset_[idx]; }
    Field& operator[](const idx_t idx) { return fieldset_[idx]; }

    const Field& operator[](const std::string& name) const { return fieldset_.field(name); }
    Field& operator[](const std::string& name) { return fieldset_.field(name); }

    const util::Metadata& metadata() const { return metadata_; }
    util::Metadata& metadata() { return metadata_; }

    // -- Modifiers

    /// @brief Implicit conversion to Array
    operator const array::Array&() const { return array(); }
    operator array::Array&() { return array(); }

    operator const FieldSet&() const { return fieldset_; }

    operator FieldSet&() { return fieldset_; }

    /// @brief Access contained Array
    const array::Array& array() const {
        ATLAS_ASSERT(array_);
        return *array_;
    }
    array::Array& array() {
        ATLAS_ASSERT(array_);
        return *array_;
    }

    /// @brief Access contained FieldSet
    const FieldSet& fieldset() const { return fieldset_; }
    FieldSet& fieldset() { return fieldset_; }

    void add(Field& field);

public:  // temporary public for prototyping
    FieldSet fieldset_;
    std::shared_ptr<array::Array> array_;
    util::Metadata metadata_;
};


class MultiField : public util::ObjectHandle<MultiFieldImpl> {
public:  // methods
         //-- Constructors
    using Handle::Handle;

    MultiField(const eckit::Configuration&);

    //-- Accessors

    const Field& field(const std::string& name) const { return get()->field(name); }
    Field& field(const std::string& name) { return get()->field(name); }
    bool has(const std::string& name) const { return get()->has(name); }
    std::vector<std::string> field_names() const { return get()->field_names(); }

    const Field& field(const idx_t idx) const { return get()->field(idx); }
    Field& field(const idx_t idx) { return get()->field(idx); }
    idx_t size() const { return get()->size(); }

    const Field& operator[](const idx_t idx) const { return get()->field(idx); }
    Field& operator[](const idx_t idx) { return get()->field(idx); }

    const Field& operator[](const std::string& name) const { return get()->field(name); }
    Field& operator[](const std::string& name) { return get()->field(name); }

    const util::Metadata& metadata() const { return get()->metadata(); }
    util::Metadata& metadata() { return get()->metadata(); }

    // -- Modifiers

    /// @brief Implicit conversion to Array
    operator const array::Array&() const { return get()->array(); }
    operator array::Array&() { return get()->array(); }

    operator const FieldSet&() const { return get()->fieldset_; }
    operator FieldSet&() { return get()->fieldset_; }

    /// @brief Access contained Array
    const array::Array& array() const { return get()->array(); }
    array::Array& array() { return get()->array(); }
};


//------------------------------------------------------------------------------------------------------

class MultiFieldCreator : public util::Object {
public:
    MultiFieldCreator(const eckit::Configuration& = util::Config());

    virtual ~MultiFieldCreator();

    virtual MultiFieldImpl* create(const eckit::Configuration& = util::Config()) const = 0;
};

//------------------------------------------------------------------------------------------------------

class MultiFieldCreatorFactory : public util::Factory<MultiFieldCreatorFactory> {
public:
    static std::string className() { return "MultiFieldCreatorFactory"; }

    /*!
   * \brief build MultiFieldCreator with options specified in parametrisation
   * \return MutliField creator
   */
    static MultiFieldCreator* build(const std::string&, const eckit::Configuration& = util::NoConfig());

    using Factory::Factory;

private:
    virtual MultiFieldCreator* make(const eckit::Configuration&) = 0;
};

template <class T>
class MultiFieldCreatorBuilder : public MultiFieldCreatorFactory {
    virtual MultiFieldCreator* make(const eckit::Configuration& config) override { return new T(config); }

public:
    using MultiFieldCreatorFactory::MultiFieldCreatorFactory;
};

// ------------------------------------------------------------------------------------

}  // namespace field
}  // namespace atlas
