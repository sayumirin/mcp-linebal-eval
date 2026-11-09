| scope | model | variant | runs | success % | feasible % (T1–T3) | optimal % (T1–T3) | near-opt % | report consistent % | silent error % | pass^1 % | pass^3 % (tasks) | gap % median (IQR) | tokens in / out | LLM / tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | deepseek-v4-flash | A0 | 73 | 91.8 [83, 96] | 91.8 [83, 96] | 90.4 [82, 95] | 91.8 | 100.0 [95, 100] | 0.0 [0, 5] | 92.3 | 86.4 (22) | 0.0 (0.0–0.0) | 1,048 / 12,947 | 1.0 / 0.0 |
| pooled | deepseek-v4-flash | A1 | 59 | 96.6 [88, 99] | 96.5 [88, 99] | 94.7 [86, 98] | 96.5 | 100.0 [94, 100] | 0.0 [0, 6] | 96.7 | 93.8 (12) | 0.0 (0.0–0.0) | 48,542 / 19,066 | 5.1 / 8.1 |
| pooled | deepseek-v4-flash | A1/faults 0.1 | 44 | 97.7 [88, 100] | 97.7 [88, 100] | 93.2 [82, 98] | 97.7 | 100.0 [92, 100] | 0.0 [0, 8] | 94.4 | 100.0 (11) | 0.0 (0.0–0.0) | 57,503 / 17,708 | 5.6 / 8.9 |
| pooled | deepseek-v4-flash | A1/strict | 5 | 100.0 [57, 100] | 100.0 [44, 100] | 100.0 [44, 100] | 100.0 | 100.0 [51, 100] | 0.0 [0, 43] | 100.0 | – | 0.0 (0.0–0.0) | 44,918 / 17,136 | 6.4 / 8.8 |
| pooled | deepseek-v4-flash | A2 | 10 | 80.0 [49, 94] | 75.0 [41, 93] | 75.0 [41, 93] | 75.0 | 100.0 [65, 100] | 0.0 [0, 28] | 85.7 | – | 0.0 (0.0–0.0) | 67,759 / 30,848 | 5.0 / 7.6 |
| pooled | deepseek-v4-flash | A2/strict | 5 | 80.0 [38, 96] | 66.7 [21, 94] | 33.3 [6, 79] | 33.3 | 100.0 [44, 100] | 0.0 [0, 43] | 80.0 | – | 33.3 (16.7–50.0) | 41,211 / 23,215 | 5.8 / 7.6 |

Rates are per run with 95% Wilson intervals. pass^k is the unbiased estimator of tau-bench (per task, then averaged) on the success criterion; pass^3 uses only tasks with at least three runs (count in brackets). The gap is relative to the exact optimum and only counts feasible runs. 'Report consistent' is over runs that submitted a balance.

6 logged records were left out: crashed episodes (API or harness errors) and earlier attempts superseded by a rerun.
