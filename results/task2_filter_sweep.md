### Task 2: filter sweep

| filter | still jitter raw px | still jitter smoothed px | change % | fast moves: delay ms | fast moves: lag px | fast moves: lag p95 px |
| --- | --- | --- | --- | --- | --- | --- |
| EMA (tau 0.048 s) | 8.01 | 5.22 | -34.91 | 28.5 | 2.23 | 9.57 |
| One Euro, min cutoff 0.5 Hz, beta 0.003 | 8.01 | 1.59 | -80.21 | 61.0 | 5.75 | 20.47 |
| One Euro, min cutoff 0.5 Hz, beta 0.01 | 8.01 | 2.49 | -68.98 | 31.5 | 3.64 | 11.55 |
| One Euro, min cutoff 0.5 Hz, beta 0.03 | 8.01 | 3.82 | -52.29 | 15.0 | 2.25 | 6.37 |
| One Euro, min cutoff 1.0 Hz, beta 0.003 | 8.01 | 2.16 | -73.08 | 53.0 | 4.66 | 17.52 |
| One Euro, min cutoff 1.0 Hz, beta 0.01 | 8.01 | 2.9 | -63.81 | 29.0 | 3.14 | 10.5 |
| One Euro, min cutoff 1.0 Hz, beta 0.03 | 8.01 | 4.07 | -49.25 | 15.0 | 2.02 | 5.96 |
| One Euro, min cutoff 2.0 Hz, beta 0.003 | 8.01 | 3.06 | -61.81 | 41.5 | 3.38 | 13.44 |
| One Euro, min cutoff 2.0 Hz, beta 0.01 | 8.01 | 3.6 | -55.05 | 25.0 | 2.47 | 8.82 |
| One Euro, min cutoff 2.0 Hz, beta 0.03 | 8.01 | 4.51 | -43.68 | 13.0 | 1.7 | 5.29 |
