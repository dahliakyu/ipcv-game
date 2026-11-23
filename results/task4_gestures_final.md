### Task 4: events

| clip | player | active min | punch | arms_crossed | hands_up | events /min | bolts | shields | beams | no orb | not facing | event→screen ms | with other ±s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shoot_test-final | P1 | 0.34 | 4 | 0 | 0 | 11.9 | 4 | 0 | 0 | 0 | 0 | 32.7 | 2 |
| Shoot_test-final | P2 | 0.34 | 5 | 0 | 0 | 14.9 | 2 | 0 | 0 | 0 | 3 | 26.1 | 2 |
| shield_test-final | P1 | 0.46 | 1 | 11 | 0 | 25.9 | 1 | 11 | 0 | 0 | 0 | 26.6 | 3 |
| shield_test-final | P2 | 0.46 | 0 | 7 | 0 | 15.1 | 0 | 6 | 0 | 0 | 1 | 30.6 | 3 |
| Laser_test-final | P1 | 0.58 | 2 | 0 | 5 | 12.0 | 2 | 0 | 1 | 4 | 0 | 25.6 | 5 |
| Laser_test-final | P2 | 0.58 | 2 | 1 | 5 | 13.8 | 2 | 1 | 0 | 5 | 0 | 26.2 | 5 |
| session-final | P1 | 1.83 | 8 | 3 | 2 | 7.10 | 8 | 3 | 1 | 1 | 0 | 28.1 | 1 |
| session-final | P2 | 1.77 | 8 | 2 | 1 | 6.22 | 8 | 2 | 0 | 1 | 0 | 28.1 | 1 |
| session-final-light0.3-noise8 | P1 | 1.81 | 8 | 6 | 2 | 8.82 | 6 | 4 | 1 | 1 | 4 | 26.0 | 0 |
| session-final-light0.3-noise8 | P2 | 1.83 | 8 | 8 | 1 | 9.29 | 7 | 7 | 0 | 1 | 2 | 29.4 | 0 |

### Task 4: recognition

| clip | gesture | intended | recognised | missed % | false triggers | of which other gesture | precision % | latency ms | latency p95 ms | to screen ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shoot_test-final | punch | 11 | 9 | 18.2 | 0 | 0 | 100.0 | 159.3 | 324.6 | 180.1 |
| Shoot_test-final | arms_crossed | 0 | 0 | – | 0 | 0 | – | – | – | – |
| Shoot_test-final | hands_up | 0 | 0 | – | 0 | 0 | – | – | – | – |
| shield_test-final | punch | 0 | 0 | – | 1 | 1 | 0.00 | – | – | – |
| shield_test-final | arms_crossed | 15 | 12 | 20.0 | 6 | 0 | 66.7 | 217.0 | 619.3 | 245.6 |
| shield_test-final | hands_up | 0 | 0 | – | 0 | 0 | – | – | – | – |
| Laser_test-final | punch | 0 | 0 | – | 4 | 4 | 0.00 | – | – | – |
| Laser_test-final | arms_crossed | 0 | 0 | – | 1 | 0 | 0.00 | – | – | – |
| Laser_test-final | hands_up | 14 | 10 | 28.6 | 0 | 0 | 100.0 | 176.3 | 307.9 | 203.2 |

### Task 4: both players

| clip | two-player trials | both recognised % |
| --- | --- | --- |
| Shoot_test-final | 2 | 100.0 |
| shield_test-final | 3 | 66.7 |
| Laser_test-final | 5 | 60.0 |
