Hi! This repository is an implementation of the EPED Saarelma-Connor Adapted Pedestal Evolution (ESCAPE) model.

This is a model for the tokamak pedestal and involves the Saarelma-Connor model implemented in https://doi.org/10.1088/1741-4326/ad4b3e
and paired with EPEDNN. 

This model physically targets ELMy H-mode plasmas but has been extended to ELM-free regimes.

This code was created by John Anthony Labbate (john.a.labbate@columbia.edu) and Andrew Oak Nelson. Enjoy!

Dependencies that need to be pip installed:
- OpenFusionToolkit
- omfit-classes
- Uncertainties
- scipy OR Firedrake 2026.4.1 (depending on choice of solver)
- HDF5
