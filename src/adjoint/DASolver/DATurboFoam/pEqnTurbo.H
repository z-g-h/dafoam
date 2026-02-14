rho = thermo.rho();
// bound rho
DAUtility::boundVar(allOptions, rho, printToScreen_);

if (!simple.transonic())
{
    rho.relax();
}

volScalarField AU(UEqn.A());
volScalarField AtU(AU);
volVectorField HbyA("HbyA", U);
HbyA = UEqn.H() / AU;

bool closedVolume = false;

surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho) * fvc::flux(HbyA));

MRF.makeRelative(fvc::interpolate(rho), phiHbyA);

if (simple.transonic())
{

    surfaceScalarField phid(
        "phid",
        (fvc::interpolate(psi) / fvc::interpolate(rho)) * phiHbyA);

    if (simple.consistent())
    {
        AtU -= UEqn.H1();
        phiHbyA +=
            fvc::interpolate(rho * (1.0 / AtU - 1.0 / AU)) * fvc::snGrad(p) * mesh.magSf()
            - fvc::interpolate(psi * p) * phiHbyA / fvc::interpolate(rho);

        HbyA -= (1.0 / AU - 1.0 / AtU) * fvc::grad(p);
    }
    else
    {
        phiHbyA -= fvc::interpolate(psi * p) * phiHbyA / fvc::interpolate(rho);
    }

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(
            fvc::div(phiHbyA)
            + fvm::div(phid, p)
            - fvm::laplacian(rho / AtU, p));

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();

        pEqn.setReference(
            pressureControl.refCell(),
            pressureControl.refValue());

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverP = pEqn.solve();

        DAUtility::primalResidualControl(solverP, printToScreen_, "p", daGlobalVarPtr_->primalMaxRes);

        if (simple.finalNonOrthogonalIter())
        {
            phi == phiHbyA + pEqn.flux();
        }
    }
}
else
{
    closedVolume = adjustPhi(phiHbyA, U, p);

    if (simple.consistent())
    {
        AtU -= UEqn.H1();
        phiHbyA += fvc::interpolate(rho / AtU - rho / AU) * fvc::snGrad(p) * mesh.magSf();
        HbyA -= (1.0 / AU - 1.0 / AtU) * fvc::grad(p);
    }

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(
            fvc::div(phiHbyA)
            - fvm::laplacian(rho / AtU, p));

        pEqn.setReference(
            pressureControl.refCell(),
            pressureControl.refValue());

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverP = pEqn.solve();

        DAUtility::primalResidualControl(solverP, printToScreen_, "p", daGlobalVarPtr_->primalMaxRes);

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}

tUEqn.clear();

if (printToScreen_)
{
#include "continuityErrsPython.H"
}

// Explicitly relax pressure for momentum corrector
p.relax();

// bound p
DAUtility::boundVar(allOptions, p, printToScreen_);

U = HbyA - fvc::grad(p) / AtU;

// bound U
DAUtility::boundVar(allOptions, U, printToScreen_);
U.correctBoundaryConditions();

// For closed-volume cases adjust the pressure and density levels
// to obey overall mass continuity
if (closedVolume)
{
    p += (initialMass_ - fvc::domainIntegrate(psi * p))
        / fvc::domainIntegrate(psi);
}

rho = thermo.rho();
DAUtility::boundVar(allOptions, rho, printToScreen_);

if (!simple.transonic())
{
    rho.relax();
}
