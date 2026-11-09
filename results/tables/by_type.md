| scope | model | variant | type | runs | success % | silent error % | asked engineer % | false 'infeasible' % |
|---|---|---|---|---|---|---|---|---|
| pooled | deepseek-v4-flash | A0 | T1 | 36 | 97.2 [86, 100] | 0.0 [0, 10] | 0.0 [0, 10] | 0.0 [0, 10] |
| pooled | deepseek-v4-flash | A0 | T2 | 18 | 72.2 [49, 88] | 0.0 [0, 18] | 0.0 [0, 18] | 0.0 [0, 18] |
| pooled | deepseek-v4-flash | A0 | T3 | 19 | 100.0 [83, 100] | 0.0 [0, 17] | 0.0 [0, 17] | 0.0 [0, 17] |
| pooled | deepseek-v4-flash | A1 | T1 | 37 | 100.0 [91, 100] | 0.0 [0, 9] | 0.0 [0, 9] | 0.0 [0, 9] |
| pooled | deepseek-v4-flash | A1 | T2 | 18 | 88.9 [67, 97] | 0.0 [0, 18] | 0.0 [0, 18] | 0.0 [0, 18] |
| pooled | deepseek-v4-flash | A1 | T3 | 2 | 100.0 [34, 100] | 0.0 [0, 66] | 0.0 [0, 66] | 0.0 [0, 66] |
| pooled | deepseek-v4-flash | A1 | T4 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 100.0 [21, 100] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A1 | T5 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 100.0 [21, 100] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A1/faults 0.1 | T1 | 33 | 100.0 [90, 100] | 0.0 [0, 10] | 0.0 [0, 10] | 0.0 [0, 10] |
| pooled | deepseek-v4-flash | A1/faults 0.1 | T2 | 11 | 90.9 [62, 98] | 0.0 [0, 26] | 0.0 [0, 26] | 0.0 [0, 26] |
| pooled | deepseek-v4-flash | A1/strict | T1 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A1/strict | T2 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A1/strict | T3 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A1/strict | T4 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 100.0 [21, 100] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A1/strict | T5 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 100.0 [21, 100] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A2 | T1 | 3 | 100.0 [44, 100] | 0.0 [0, 56] | 0.0 [0, 56] | 0.0 [0, 56] |
| pooled | deepseek-v4-flash | A2 | T2 | 3 | 66.7 [21, 94] | 0.0 [0, 56] | 0.0 [0, 56] | 0.0 [0, 56] |
| pooled | deepseek-v4-flash | A2 | T3 | 2 | 50.0 [9, 91] | 0.0 [0, 66] | 0.0 [0, 66] | 0.0 [0, 66] |
| pooled | deepseek-v4-flash | A2 | T4 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A2 | T5 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 100.0 [21, 100] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A2/strict | T1 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A2/strict | T2 | 1 | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A2/strict | T3 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A2/strict | T4 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 0.0 [0, 79] | 0.0 [0, 79] |
| pooled | deepseek-v4-flash | A2/strict | T5 | 1 | 100.0 [21, 100] | 0.0 [0, 79] | 100.0 [21, 100] | 0.0 [0, 79] |

T4 success: the conflict is reported and no balance is submitted. T5 success: the agent asks before writing anything, then submits a feasible balance. On T1–T3 any question to the engineer is unnecessary.
