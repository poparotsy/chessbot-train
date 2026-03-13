# v6 Path Analysis

- images: 50
- board_pass: 50/50
- full_pass: 48/50
- median_sec: 4.6006
- p95_sec: 5.7781

## Selected Path Families
- full: 25
- contour: 13
- lattice: 7
- panel_split: 3
- axis_grid: 1
- gradient_projection: 1

## Risk Flags (Actionable)
- orientation_nonlabel_heuristic: 19
- candidate_cap_high_pressure: 8
- special_path_dependency: 8
- selection_inversion_low_delta: 7
- near_tie_low_conf: 4

## What This Means
- Board recognition is stable: 50/50.
- Full FEN misses are side-to-move only: 48/50.

## Critical Cases
- full_fen_failures: puzzle-00041.jpeg, puzzle-00045.jpeg
- special_dependency_cases: puzzle-00028.jpeg, puzzle-00031.jpeg, puzzle-00038.jpeg, puzzle-00041.jpeg, puzzle-00044.jpeg, puzzle-00045.jpeg, puzzle-00046.jpeg, puzzle-00049.jpeg
- candidate_cap_high_pressure_cases: puzzle-00003.jpeg, puzzle-00006.jpeg, puzzle-00012.jpeg, puzzle-00020.jpeg, puzzle-00028.jpeg, puzzle-00033.jpeg, puzzle-00040.jpeg, puzzle-00042.jpeg
- perf_outliers: puzzle-00003.jpeg

## Board Failures
- none

## High-Risk Per-Image
- puzzle-00028.jpeg: board=PASS tag=gradient_projection family=gradient_projection conf=0.9557 t=5.157s risks=near_tie_low_conf,candidate_cap_high_pressure,special_path_dependency
- puzzle-00031.jpeg: board=PASS tag=contour_robust_relaxed family=contour conf=0.9759 t=4.457s risks=near_tie_low_conf,orientation_nonlabel_heuristic,special_path_dependency
- puzzle-00046.jpeg: board=PASS tag=contour_robust_gfit_inset2 family=contour conf=1.0000 t=4.609s risks=orientation_nonlabel_heuristic,special_path_dependency,selection_inversion_low_delta
- puzzle-00006.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.233s risks=candidate_cap_high_pressure,selection_inversion_low_delta
- puzzle-00024.jpeg: board=PASS tag=lattice family=lattice conf=0.9823 t=4.425s risks=near_tie_low_conf,orientation_nonlabel_heuristic
- puzzle-00033.jpeg: board=PASS tag=full family=full conf=1.0000 t=5.322s risks=orientation_nonlabel_heuristic,candidate_cap_high_pressure
- puzzle-00044.jpeg: board=PASS tag=panel_split_top_trim8 family=panel_split conf=0.9794 t=4.343s risks=orientation_nonlabel_heuristic,special_path_dependency
- puzzle-00045.jpeg: board=PASS tag=panel_split_left_inset2 family=panel_split conf=0.9899 t=4.042s risks=orientation_nonlabel_heuristic,special_path_dependency
- puzzle-00049.jpeg: board=PASS tag=contour_legacy_gfit family=contour conf=0.9873 t=4.772s risks=near_tie_low_conf,special_path_dependency
- puzzle-00001.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.790s risks=selection_inversion_low_delta
- puzzle-00002.png: board=PASS tag=full family=full conf=0.9997 t=4.083s risks=orientation_nonlabel_heuristic
- puzzle-00003.jpeg: board=PASS tag=contour_robust_enhsrc family=contour conf=1.0000 t=10.549s risks=candidate_cap_high_pressure
- puzzle-00005.jpeg: board=PASS tag=full family=full conf=0.9993 t=4.413s risks=orientation_nonlabel_heuristic
- puzzle-00007.jpeg: board=PASS tag=full family=full conf=1.0000 t=3.790s risks=selection_inversion_low_delta
- puzzle-00008.jpeg: board=PASS tag=lattice family=lattice conf=1.0000 t=4.466s risks=orientation_nonlabel_heuristic
- puzzle-00010.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.413s risks=orientation_nonlabel_heuristic
- puzzle-00012.jpeg: board=PASS tag=contour_robust family=contour conf=1.0000 t=6.179s risks=candidate_cap_high_pressure
- puzzle-00017.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.898s risks=orientation_nonlabel_heuristic
- puzzle-00020.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.896s risks=candidate_cap_high_pressure
- puzzle-00021.jpeg: board=PASS tag=contour_legacy family=contour conf=1.0000 t=4.447s risks=selection_inversion_low_delta
- puzzle-00022.png: board=PASS tag=contour_legacy family=contour conf=1.0000 t=3.667s risks=selection_inversion_low_delta
- puzzle-00025.jpeg: board=PASS tag=full family=full conf=1.0000 t=6.913s risks=orientation_nonlabel_heuristic
- puzzle-00030.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.734s risks=orientation_nonlabel_heuristic
- puzzle-00032.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.353s risks=orientation_nonlabel_heuristic
- puzzle-00036.jpeg: board=PASS tag=contour_legacy family=contour conf=1.0000 t=5.231s risks=orientation_nonlabel_heuristic
- puzzle-00037.jpeg: board=PASS tag=contour_robust family=contour conf=1.0000 t=5.348s risks=orientation_nonlabel_heuristic
- puzzle-00038.jpeg: board=PASS tag=axis_grid_window family=axis_grid conf=1.0000 t=5.778s risks=special_path_dependency
- puzzle-00039.jpeg: board=PASS tag=full family=full conf=1.0000 t=3.834s risks=orientation_nonlabel_heuristic
- puzzle-00040.jpeg: board=PASS tag=lattice_enhsrc family=lattice conf=1.0000 t=5.623s risks=candidate_cap_high_pressure
- puzzle-00041.jpeg: board=PASS tag=panel_split_right family=panel_split conf=1.0000 t=4.820s risks=special_path_dependency
- puzzle-00042.jpeg: board=PASS tag=lattice family=lattice conf=1.0000 t=4.803s risks=candidate_cap_high_pressure
- puzzle-00043.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.416s risks=orientation_nonlabel_heuristic
- puzzle-00048.jpeg: board=PASS tag=contour_robust family=contour conf=1.0000 t=4.224s risks=selection_inversion_low_delta
- puzzle-00050.jpeg: board=PASS tag=full family=full conf=1.0000 t=4.586s risks=orientation_nonlabel_heuristic
