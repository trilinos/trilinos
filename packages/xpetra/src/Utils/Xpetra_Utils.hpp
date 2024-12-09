// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef XPETRA_LOOKUPSTATUS_HPP
#define XPETRA_LOOKUPSTATUS_HPP

#include "Xpetra_ConfigDefs.hpp"

#include "Tpetra_ConfigDefs.hpp"

#include "Xpetra_Map.hpp"  // definition of UnderlyingLib

namespace Xpetra {

//! Convert a Xpetra::UnderlyingLib to a std::string
std::string toString(UnderlyingLib lib);

//! Convert a Tpetra::LookupStatus to a Xpetra::LookupStatus.
Xpetra::LookupStatus toXpetra(Tpetra::LookupStatus);

//! Convert a Xpetra::OptimizeOption to a Tpetra::OptimizeOption.
Tpetra::OptimizeOption toTpetra(Xpetra::OptimizeOption);

//! Convert a Xpetra::CombineMode to a Tpetra::CombineMode.
Tpetra::CombineMode toTpetra(Xpetra::CombineMode CM);

//! Convert a Xpetra::LocalGlobal to a Tpetra::LocalGlobal.
Tpetra::LocalGlobal toTpetra(LocalGlobal lg);

}  // namespace Xpetra

#endif  // XPETRA_LOOKUPSTATUS_HPP
