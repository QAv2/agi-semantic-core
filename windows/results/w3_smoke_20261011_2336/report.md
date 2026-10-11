# W3 analysis report

Smoke analysis; bootstrap 1000 image resamples, analysis seed 20261123. Gates: gate_on pass, gate_drop pass, gate_off pass, gdet pass.

| measure | result | verdict |
|---|---|---|
| P1 movement share (M-on, mid) | w 0.23 [0.18, 0.28], label share 0.77, κ 0.080 (upper 0.107); bands 20–250: 0.11 LABEL; 250–500: 0.37 SPLIT; 500–750: 0.23 UNCLEAR; 750–1000: 0.12 LABEL | **VARIES BY LEVEL** |
| P2 label sham (300-step class) | 25: S 0.21 (upper 0.28, full-length 1.02); 100: S 0.14 (upper 0.18, full-length 0.70); 300: S 0.09 (upper 0.12, full-length 0.52) | **SILENT** |
| S1 profile (M-on, all sites) | enc32 0.12 LABEL; enc16 0.13 LABEL; enc8 0.22 VARIES BY LEVEL; mid 0.23 VARIES BY LEVEL; dec8 0.21 VARIES BY LEVEL; dec16 0.08 UNREADABLE; dec32 0.07 LABEL | |
| S2 probe read (M-on) | enc32 -0.05 [-0.12, 0.01] UNREADABLE; enc16 0.14 [0.10, 0.17] LABEL; enc8 0.07 [0.05, 0.10] LABEL; mid 0.07 [0.05, 0.10] LABEL; dec8 0.07 [0.05, 0.10] LABEL; dec16 0.03 [0.00, 0.06] LABEL; dec32 0.03 [0.00, 0.06] LABEL | |
| S3 output (hand stats of ε̂) | w 1.46 [1.22, 1.69], κ 3.367 → UNREADABLE | |
| S4 directions (content share) | dir 1 0.20 [0.15, 0.24]; dir 2 0.27 [0.21, 0.33] | |
| S5 offsets (w by class) | 25: 0.04 [0.02, 0.06]; 100: 0.19 [0.15, 0.23]; 300: 0.26 [0.19, 0.32] | |
| S6 M-drop | w 0.55 [0.50, 0.60] SPLIT; M-drop − M-on 0.31 [0.27, 0.35]; sham S 0.18 (upper 0.24); no-label probe beats the hand floor at: dec8, dec16, dec32 | |
| S7 M-off against the floor | hand floor median 29 steps; best site mid 25 steps; beats hand at: enc8, mid, dec8, dec16, dec32; beats untrained twin at: enc32, enc16, enc8, mid, dec8, dec16, dec32 | |
| S8 untrained twin of M-on | enc32 0.21 κ 0.01 VARIES BY LEVEL; enc16 0.08 κ 0.01 LABEL; enc8 0.08 κ 0.02 LABEL; mid 0.07 κ 0.02 LABEL; dec8 0.07 κ 0.02 LABEL; dec16 0.07 κ 0.02 LABEL; dec32 0.12 κ 0.01 LABEL | |
| S9 seed spread (CV R², increment of mid over hand) | band 0 (t 91–109): skipped; band 1 (t 291–310): skipped; band 2 (t 492–508): skipped; band 3 (t 690–706): skipped | |
| S10 training record | files: gate_drop, gate_off, gate_on, gdet, train_drop, train_off, train_on | |

Verdict rules: registration §5 (docs/W3_LOOP_PROTOCOL.md). Full numbers in verdict.json.
