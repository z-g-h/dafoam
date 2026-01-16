/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2017 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::aeroThermalCoupleFvPatchScalarField

Description
    This boundary condition applies a couple for aero and thermal 

    where:
    \vartable
        Tn  | neighbour temperature
        Cn  | neighbour Heat resistance parameter 
    \endvartable

    The thermal conductivity \c kappa is use polynomial read from solidProperties.

Usage
    \table
    Property    | Description                                       | Required | Default
    Tn          | neighbour temperature                             | yes |
    Cn          | neighbour Heat resistance parameter               | yes |
    discipline | owner solver type                                 | yes |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            aeroThermalCouple;

        Tn              uniform 300.0;
        Cn              uniform 1000000.0;
        discipline      aero;
        value           $internalField;
    }
    \endverbatim

See also
    Foam::mixedFvPatchScalarField

SourceFiles
    aeroThermalCoupleFvPatchScalarField.C

\*---------------------------------------------------------------------------*/

#ifndef aeroThermalCoupleFvPatchScalarField_H
#define aeroThermalCoupleFvPatchScalarField_H

#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class aeroThermalCoupleFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class aeroThermalCoupleFvPatchScalarField
    : public mixedFvPatchScalarField
{

private:
    // Private data

    //- neighbour temperature
    scalarField Tn_;

    //- neighbour Heat resistance parameter
    scalarField Cn_;

    //- tag for owner solver type : aero/thermal
    word discipline_;

    //- tag for distance type: default/ daCustom, this will be reset to DAFOAM options
    word distanceMode_;

public:
    //- Runtime type information
    TypeName("aeroThermalCouple");

    // Constructors

    //- Construct from patch and internal field
    aeroThermalCoupleFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct from patch, internal field and dictionary
    aeroThermalCoupleFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const dictionary&);

    //- Construct by mapping given
    //  externalWallHeatFluxTemperatureFvPatchScalarField
    //  onto a new patch
    aeroThermalCoupleFvPatchScalarField(
        const aeroThermalCoupleFvPatchScalarField&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    aeroThermalCoupleFvPatchScalarField(
        const aeroThermalCoupleFvPatchScalarField&);

    //- Construct and return a clone
    virtual tmp<fvPatchScalarField> clone() const
    {
        return tmp<fvPatchScalarField>(
            new aeroThermalCoupleFvPatchScalarField(*this));
    }

    //- Construct as copy setting internal field reference
    aeroThermalCoupleFvPatchScalarField(
        const aeroThermalCoupleFvPatchScalarField&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct and return a clone setting internal field reference
    virtual tmp<fvPatchScalarField> clone(
        const DimensionedField<scalar, volMesh>& iF) const
    {
        return tmp<fvPatchScalarField>(
            new aeroThermalCoupleFvPatchScalarField(*this, iF));
    }

    // Member functions

    // Access

    //- Allow manipulation of the boundary values
    virtual bool fixesValue() const
    {
        return false;
    }

    // Mapping functions

    //- Map (and resize as needed) from self given a mapping object
    virtual void autoMap(
        const fvPatchFieldMapper&);

    //- Reverse map the given fvPatchField onto this fvPatchField
    virtual void rmap(
        const fvPatchScalarField&,
        const labelList&);

    // Evaluation functions

    //- Update the coefficients associated with the patch field
    virtual void updateCoeffs();

    // Return defining fields
    virtual scalarField& refTn()
    {
        return Tn_;
    }

    virtual const scalarField& refTn() const
    {
        return Tn_;
    }

    virtual scalarField& refCn()
    {
        return Cn_;
    }

    virtual const scalarField& refCn() const
    {
        return Cn_;
    }

    virtual word& refDiscipline()
    {
        return discipline_;
    }

    virtual const word& refDiscipline() const
    {
        return discipline_;
    }

    virtual word& refDistanceMode()
    {
        return distanceMode_;
    }

    virtual const word& refDistanceMode() const
    {
        return distanceMode_;
    }

    // I-O

    //- Write
    void write(Ostream&) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
