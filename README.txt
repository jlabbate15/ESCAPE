Hi! This repository is an implementation of the EPED Saarelma-Connor Adapted Pedestal Evolution (ESCAPE) model.

This is a model for the tokamak pedestal and involves the Saarelma-Connor model implemented in https://doi.org/10.1088/1741-4326/ad4b3e
and paired with EPEDNN. 

This model physically targets ELMy H-mode plasmas but has been extended to ELM-free regimes.

This code was created by John Anthony Labbate (john.a.labbate@columbia.edu) and Andrew Oak Nelson. Enjoy!

Installation (Python >= 3.11)
------------------------------
    git clone --recurse-submodules https://github.com/jlabbate15/ESCAPE.git
    cd ESCAPE
    pip install -e .              # escape + numpy, scipy, matplotlib
    pip install -e ".[eped1]"     # also juliacall + juliapkg, for epednn_model='EPED1'

Only editable (-e) installs are supported: ESCAPE finds dependencies/EPEDNN.jl and
dependencies/epednn_mit relative to its source files. Then import with, e.g.:
    from escape.ESCAPE.ESCAPE_solve import ESCAPE_solve

Install separately when needed:
- Firedrake 2026.4.1             - 3D model, or implementation='firedrake'
- tensorflow                     - epednn_model='EPED_SPARC' or 'EPED_SCOPING' (dependencies/epednn_mit;
                                   the scoping model is on its 'scoping_model' branch)
- OpenFUSIONToolkit (needs HDF5) - g-file loading in escape.helpers
- omfit-classes, uncertainties   - profile loading in escape.helpers
