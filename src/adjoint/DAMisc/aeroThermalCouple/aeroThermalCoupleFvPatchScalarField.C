/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2017 OpenFOAM Foundation
     \\/     M anipulation  | Copyright (C) 2015-2017 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "aeroThermalCoupleFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "DAModel.H"
#include "thermodynamicConstants.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::aeroThermalCoupleFvPatchScalarField::
    aeroThermalCoupleFvPatchScalarField(
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF)
    : mixedFvPatchScalarField(p, iF),
      Tn_(),
      Cn_(),
      discipline_("aero"),
      distanceMode_("default")
{
    refValue() = 0.0;
    refGrad() = 0.0;
    valueFraction() = 1.0;
}

Foam::aeroThermalCoupleFvPatchScalarField::
    aeroThermalCoupleFvPatchScalarField(
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const dictionary& dict)
    : mixedFvPatchScalarField(p, iF),
      Tn_(),
      Cn_(),
      discipline_(dict.lookupOrDefault<word>("discipline", "aero")),
      distanceMode_(dict.lookupOrDefault<word>("distanceMode", "default"))
{

    Tn_ = scalarField("Tn", dict, p.size());
    Cn_ = scalarField("Cn", dict, p.size());

    fvPatchScalarField::operator=(scalarField("value", dict, p.size()));

    if (dict.found("refValue"))
    {
        // Full restart
        refValue() = scalarField("refValue", dict, p.size());
        refGrad() = scalarField("refGradient", dict, p.size());
        valueFraction() = scalarField("valueFraction", dict, p.size());
    }
    else
    {
        // Start from user entered data. Assume fixedValue.
        refValue() = *this;
        refGrad() = 0.0;
        valueFraction() = 1.0;
    }
}

Foam::aeroThermalCoupleFvPatchScalarField::
    aeroThermalCoupleFvPatchScalarField(
        const aeroThermalCoupleFvPatchScalarField& ptf,
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const fvPatchFieldMapper& mapper)
    : mixedFvPatchScalarField(ptf, p, iF, mapper),
      Tn_(),
      Cn_(),
      discipline_(ptf.discipline_),
      distanceMode_(ptf.distanceMode_)
{

    Tn_.setSize(mapper.size());
    Tn_.map(ptf.Tn_, mapper);

    Cn_.setSize(mapper.size());
    Cn_.map(ptf.Cn_, mapper);
}

Foam::aeroThermalCoupleFvPatchScalarField::
    aeroThermalCoupleFvPatchScalarField(
        const aeroThermalCoupleFvPatchScalarField& ewhftpsf)
    : mixedFvPatchScalarField(ewhftpsf),
      Tn_(ewhftpsf.Tn_),
      Cn_(ewhftpsf.Cn_),
      discipline_(ewhftpsf.discipline_),
      distanceMode_(ewhftpsf.distanceMode_)
{
}

Foam::aeroThermalCoupleFvPatchScalarField::
    aeroThermalCoupleFvPatchScalarField(
        const aeroThermalCoupleFvPatchScalarField& ewhftpsf,
        const DimensionedField<scalar, volMesh>& iF)
    : mixedFvPatchScalarField(ewhftpsf, iF),
      Tn_(ewhftpsf.Tn_),
      Cn_(ewhftpsf.Cn_),
      discipline_(ewhftpsf.discipline_),
      distanceMode_(ewhftpsf.distanceMode_)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::aeroThermalCoupleFvPatchScalarField::autoMap(
    const fvPatchFieldMapper& m)
{
    mixedFvPatchScalarField::autoMap(m);

    Tn_.autoMap(m);
    Cn_.autoMap(m);
}

void Foam::aeroThermalCoupleFvPatchScalarField::rmap(
    const fvPatchScalarField& ptf,
    const labelList& addr)
{
    mixedFvPatchScalarField::rmap(ptf, addr);

    const aeroThermalCoupleFvPatchScalarField& ewhftpsf =
        refCast<const aeroThermalCoupleFvPatchScalarField>(ptf);

    Tn_.rmap(ewhftpsf.Tn_, addr);
    Cn_.rmap(ewhftpsf.Cn_, addr);
}

/// @brief  now only support solid boundary
/// TODO: add fluid
void Foam::aeroThermalCoupleFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // neighKDeltaCoeffs / ( neighKDeltaCoeffs + myKDeltaCoeffs)
    scalar deltaCoeffs = 0.0;
    if (discipline_ == "aero")
    {
        // for incompressible flow  Q = Cp * alphaEff * dT/dz, so kappa = Cp * alphaEff
        DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(db().lookupObject<DATurbulenceModel>("DATurbulenceModel"));
        word turbModelType = daTurb.getTurbModelType();

        if (turbModelType == "incompressible")
        {
            volScalarField alphaEff = daTurb.alphaEff();

            const IOdictionary& transportDict = db().lookupObject<IOdictionary>("transportProperties");
            scalar Cp = readScalar(transportDict.lookup("Cp"));

            forAll(this->refValue(), faceI)
            {
                if (distanceMode_ == "default")
                {
                    deltaCoeffs = patch().deltaCoeffs()[faceI];
                }
                else if (distanceMode_ == "daCustom")
                {
                    label nearWallCellIndex = this->patch().faceCells()[faceI];
                    vector c1 = this->patch().Cf()[faceI];
                    vector c2 = this->patch().boundaryMesh().mesh().C()[nearWallCellIndex];
                    scalar d = mag(c1 - c2);
                    deltaCoeffs = 1 / d;
                }

                scalar alphaEffBf = alphaEff.boundaryField()[patch().index()][faceI];
                scalar myKDeltaCoeffs = Cp * alphaEffBf * deltaCoeffs;
                refValue()[faceI] = Tn_[faceI];
                refGrad()[faceI] = 0.0;
                valueFraction()[faceI] = Cn_[faceI] / (Cn_[faceI] + myKDeltaCoeffs);
            }
        }
        else if (turbModelType == "compressible")
        {
            // for compressible flow Q = alphaEff * dHE/dz, so if enthalpy is used, kappa = Cp * alphaEff
            // if the internalEnergy is used, kappa = (Cp - R) * alphaEff
            volScalarField alphaEff = daTurb.alphaEff();
            // compressible flow, H = alphaEff * dHE/dz
            const fluidThermo& thermo = db().lookupObject<fluidThermo>("thermophysicalProperties");
            const volScalarField& he = thermo.he();

            const IOdictionary& thermoDict = db().lookupObject<IOdictionary>("thermophysicalProperties");
            dictionary mixSubDict = thermoDict.subDict("mixture");
            dictionary specieSubDict = mixSubDict.subDict("specie");
            scalar molWeight = specieSubDict.getScalar("molWeight");
            dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
            scalar Cp = thermodynamicsSubDict.getScalar("Cp");

            // 8314.4700665  gas constant in OpenFOAM
            // src/OpenFOAM/global/constants/thermodynamic/thermodynamicConstants.H
            scalar RR = Foam::constant::thermodynamic::RR;

            // R = RR/molWeight
            // Foam::specie::R() function in src/thermophysicalModels/specie/specie/specieI.H
            scalar R = RR / molWeight;

            scalar tmpVal = 0.0;
            // e = (Cp - R) * T, so Q = alphaEff * (Cp-R) * dT/dz
            if (he.name() == "e")
            {
                tmpVal = Cp - R;
            }
            // h = Cp * T, so Q = alphaEff * Cp * dT/dz
            else
            {
                tmpVal = Cp;
            }

            forAll(this->refValue(), faceI)
            {
                if (distanceMode_ == "default")
                {
                    deltaCoeffs = patch().deltaCoeffs()[faceI];
                }
                else if (distanceMode_ == "daCustom")
                {
                    label nearWallCellIndex = this->patch().faceCells()[faceI];
                    vector c1 = this->patch().Cf()[faceI];
                    vector c2 = this->patch().boundaryMesh().mesh().C()[nearWallCellIndex];
                    scalar d = mag(c1 - c2);
                    deltaCoeffs = 1 / d;
                }

                scalar alphaEffBf = alphaEff.boundaryField()[patch().index()][faceI];
                scalar myKDeltaCoeffs = tmpVal * alphaEffBf * deltaCoeffs;
                refValue()[faceI] = Tn_[faceI];
                refGrad()[faceI] = 0.0;
                valueFraction()[faceI] = Cn_[faceI] / (Cn_[faceI] + myKDeltaCoeffs);
            }
        }
    }
    else if (discipline_ == "thermal")
    {
        const volScalarField& kappa = db().lookupObject<volScalarField>("kappa");
        const fvPatchField<scalar>& kappaBF = kappa.boundaryField()[patch().index()];

        forAll(this->refValue(), faceI)
        {
            if (distanceMode_ == "default")
            {
                deltaCoeffs = patch().deltaCoeffs()[faceI];
            }
            else if (distanceMode_ == "daCustom")
            {
                label nearWallCellIndex = this->patch().faceCells()[faceI];
                vector c1 = this->patch().Cf()[faceI];
                vector c2 = this->patch().boundaryMesh().mesh().C()[nearWallCellIndex];
                scalar d = mag(c1 - c2);
                deltaCoeffs = 1 / d;
            }
            scalar myKDeltaCoeffs = kappaBF[faceI] * deltaCoeffs;
            refValue()[faceI] = Tn_[faceI];
            refGrad()[faceI] = 0.0;
            valueFraction()[faceI] = Cn_[faceI] / (Cn_[faceI] + myKDeltaCoeffs);
        }
    }
    else
    {
        FatalErrorIn("DAInputThermalCoupling::run") << " discipline not valid! "
                                                    << abort(FatalError);
    }

    mixedFvPatchScalarField::updateCoeffs();
}

void Foam::aeroThermalCoupleFvPatchScalarField::write(
    Ostream& os) const
{
    fvPatchScalarField::write(os);

    Tn_.writeEntry("Tn", os);

    Cn_.writeEntry("Cn", os);

    os.writeEntry("discipline", discipline_);
    os.writeEntry("distanceMode", distanceMode_);

    refValue().writeEntry("refValue", os);
    refGrad().writeEntry("refGradient", os);
    valueFraction().writeEntry("valueFraction", os);
    writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
makePatchTypeField(
    fvPatchScalarField,
    aeroThermalCoupleFvPatchScalarField);
}

// ************************************************************************* //
