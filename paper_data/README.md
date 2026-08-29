# LKDock v4.0 — Paper Data

Supporting data for *"LKDock: An Integrated, All-in-One Platform for Molecular Docking, Virtual Screening and Molecular Dynamics Simulation"*.

## Contents

```
docking/          test scripts and raw outputs
                  - run_docking_test.py / run_extra_tests.py / run_param_scan.py / run_new_cases.py
                  - param_scan_results.json (exhaustiveness / box-size / repeatability / scoring)
                  - new_cases_results.json  (1REV HIV-1 RT, 4HJO EGFR)
                  - 测试结果汇总.md (consolidated test log, Chinese)
data_tables/      consolidated data tables (Excel, 16+ sheets; Markdown)
figures/          all manuscript figures (Fig 1-27, 600 dpi PNG)
test5_ppi/        LKlight PPI ranking data (rank_by_luciferin/scoring/rmsd.list)
```

## Systems & engines tested
- Docking accuracy: 3PTB (trypsin/benzamidine), 1HVR (HIV-1 protease/XK263),
  4YXO (carbonic anhydrase II, LKina metal mode), 6LU7 (Mpro), 1REV (HIV-1 RT/TIBO),
  4HJO (EGFR/erlotinib)
- Engines: AutoDock Vina 1.2.7, Uni-Dock v1.1.3, UniDock-Pro GPU (RTX 2080 / 3090),
  LKina (metal), LKlight (PPI), OpenMM 8.5.2, GROMACS 2025.4

## Citation
Please cite the manuscript (DOI TBD) and this dataset (DOI 10.5281/zenodo.XXXXXXX).

## License
CC BY 4.0 (data). Engine binaries retain their original open-source licenses.
