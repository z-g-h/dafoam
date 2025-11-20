Time& runTime = runTimePtr_();
// reset time to 0
runTime.setTime(0.0, 0);
fvMesh& mesh = meshPtr_();
volScalarField& T = TPtr_();
volScalarField& k = kPtr_();
forAll(kCoeffs,order)
{
    k += kCoeffs[order]*pow(T/dimensionedScalar("Tref", dimTemperature, 1.0), order) * dimensionedScalar("kUnit", dimPower / dimLength / dimTemperature, 1.0);
}
volScalarField& fvSource = fvSourcePtr_();
