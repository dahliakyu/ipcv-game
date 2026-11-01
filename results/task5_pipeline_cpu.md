### Task 5: frame rate

| clip | frames | duration s | FPS mean | FPS 5% low | FPS 1% low | capture FPS | ms median | ms p95 | ms max | frames > 200 ms | capture gaps > 200 ms | failed frames | fallback | build s | startup s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo-cpu | 665 | 99.8 | 11.0 | 8.63 | 7.21 | 6.65 | 91.5 | 115.9 | 519.9 | 3 | 8 | 0 |  | 0.34 | 0.43 |
| session-cpu | 1908 | 110.0 | 11.2 | 8.40 | 7.42 | 17.3 | 85.5 | 119.0 | 640.3 | 3 | 2 | 0 |  | 1.40 | 1.53 |

### Task 5: stages

| clip | pose med | pose p95 | identity med | identity p95 | motion med | motion p95 | face med | face p95 | gestures med | gestures p95 | game med | game p95 | render med | render p95 | total med | total p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo-cpu | 48.8 | 63.4 | 22.9 | 30.6 | 0.12 | 0.14 | 14.7 | 16.2 | 0.11 | 0.13 | 0.13 | 0.16 | 10.3 | 14.5 | 91.5 | 115.9 |
| session-cpu | 52.0 | 70.6 | 0.66 | 26.8 | 0.11 | 0.15 | 15.5 | 17.7 | 0.11 | 0.14 | 0.11 | 0.15 | 13.2 | 17.2 | 85.5 | 119.0 |

### Task 5: resources

| clip | CPU % (1 core = 100) | CPU % max | RAM MB max | GPU util % | GPU mem MB max | torch GPU MB max |
| --- | --- | --- | --- | --- | --- | --- |
| demo-cpu | 627.4 | 781.5 | 1320.6 | 4.55 | 1510.9 | 0.00 |
| session-cpu | 594.0 | 784.4 | 1288.7 | 9.33 | 1539.6 | 0.00 |

### Task 5: fast motion

| clip | player | wrist samples | wrist p50 sw/s | wrist p95 sw/s | wrist p99 sw/s | wrist max sw/s |
| --- | --- | --- | --- | --- | --- | --- |
| demo-cpu | P1 | 1246 | 0.56 | 5.28 | 8.10 | 12.7 |
| demo-cpu | P2 | 1108 | 0.54 | 3.40 | 4.93 | 7.62 |
| session-cpu | P1 | 3554 | 0.41 | 5.74 | 10.8 | 54.4 |
| session-cpu | P2 | 3677 | 0.53 | 4.25 | 8.00 | 20.5 |
