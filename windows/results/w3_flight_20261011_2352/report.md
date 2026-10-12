# W3 analysis report

Flight analysis; bootstrap 10000 image resamples, analysis seed 20261023. Gates: gate_on pass, gate_drop pass, gate_off pass, gdet pass.

| measure | result | verdict |
|---|---|---|
| P1 movement share (M-on, mid) | w 0.17 [0.15, 0.19], label share 0.83, κ 1.246 (upper 1.434); bands 20–250: 0.11 LABEL; 250–500: 0.27 SPLIT; 500–750: 0.20 UNCLEAR; 750–1000: 0.09 LABEL | **UNREADABLE** |
| P2 label sham (300-step class) | 25: S 0.10 (upper 0.12, full-length 0.35); 100: S 0.07 (upper 0.08, full-length 0.26); 300: S 0.07 (upper 0.07, full-length 0.30) | **not judged** |
| S1 profile (M-on, all sites) | enc32 0.16 VARIES BY LEVEL; enc16 -0.02 UNREADABLE; enc8 0.17 UNREADABLE; mid 0.17 UNREADABLE; dec8 0.24 UNREADABLE; dec16 0.14 UNREADABLE; dec32 -0.25 UNREADABLE | |
| S2 probe read (M-on) | enc32 -0.05 [-0.06, -0.04] UNREADABLE; enc16 0.01 [-0.00, 0.03] UNREADABLE; enc8 0.02 [0.02, 0.03] LABEL; mid -0.03 [-0.04, -0.02] LABEL; dec8 -0.07 [-0.08, -0.05] LABEL; dec16 0.04 [0.03, 0.06] UNREADABLE; dec32 -0.15 [-0.18, -0.12] UNREADABLE | |
| S3 output (hand stats of ε̂) | w 1.45 [1.31, 1.61], κ 17.735 → UNREADABLE | |
| S4 directions (content share) | dir 1 0.19 [0.17, 0.22]; dir 2 0.14 [0.12, 0.16] | |
| S5 offsets (w by class) | 25: 0.07 [0.04, 0.10]; 100: 0.11 [0.09, 0.14]; 300: 0.19 [0.17, 0.22] | |
| S6 M-drop | w 0.28 [0.26, 0.31] VARIES BY LEVEL; M-drop − M-on 0.12 [0.10, 0.13]; sham S 0.06 (upper 0.07); no-label probe beats the hand floor at: enc32, enc16, enc8, mid, dec8, dec16, dec32 | |
| S7 M-off against the floor | hand floor median 39 steps; best site enc8 16 steps; beats hand at: enc16, enc8, mid, dec8, dec16, dec32; beats untrained twin at: enc32, enc16, enc8, mid, dec8, dec16, dec32 | |
| S8 untrained twin of M-on | enc32 0.26 κ 0.01 VARIES BY LEVEL; enc16 0.12 κ 0.01 LABEL; enc8 0.10 κ 0.01 LABEL; mid 0.09 κ 0.02 LABEL; dec8 0.08 κ 0.02 LABEL; dec16 0.09 κ 0.01 LABEL; dec32 0.17 κ 0.01 VARIES BY LEVEL | |
| S9 seed spread (CV R², increment of mid over hand) | band 0 (t 90–110): hand 0.40, mid 0.75, both 0.82, increment +0.42; band 1 (t 290–310): hand 0.40, mid 0.84, both 0.83, increment +0.44; band 2 (t 490–510): hand 0.46, mid 0.68, both 0.65, increment +0.18; band 3 (t 690–710): hand 0.60, mid 0.46, both 0.44, increment -0.16 | |
| S10 training record | files: gate_drop, gate_off, gate_on, gdet, train_drop, train_off, train_on | |

Verdict rules: registration §5 (docs/W3_LOOP_PROTOCOL.md). Full numbers in verdict.json.
