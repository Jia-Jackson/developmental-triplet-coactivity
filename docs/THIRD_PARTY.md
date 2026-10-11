# Third-party software

Maximum entropy toolbox for MATLAB by Ori Maoz: https://github.com/orimaoz/maxent_toolbox, source commit 82b4528675fed43beb1d90a1c06b43189afe023e. Its original MIT licence is retained in `third_party/maxent_toolbox/LICENSE.txt`. This external code is distinct from the authors' analysis code. Build the provided MEX source or obtain compatible binaries from the original project. MATLAB must be obtained separately.

The Python dependencies are listed in `requirements.txt` and are installed from their original distributions under their respective licences. `vendor_hotopology`, `vendor_null_models` and `vendor_kinetic_null` in the upgrade code are frozen copies of the existing project analysis implementation; the names do not imply newly imported external paper code.

Axion acquisition software and proprietary readers are not redistributed. The native SPK files are data, while the lossless CSV and MATLAB exports permit downstream analysis without that acquisition software.
