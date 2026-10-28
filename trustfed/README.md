# trustfed

Core Python library. Every module needs `pytest` tests in `tests/`.

| Package | Content | Step |
|---|---|---|
| `data/` | loaders, cleaning, splitting, partitioning, scaling | 01 |
| `models/` | DNN, GRU, LSTM with a shared input interface | 02 |
| `metrics/` | detection, robustness, privacy, efficiency metrics | 02 |
| `fl/` | `Client`, `Server`, `Aggregator` interface, FedAvg | 03 |
| `attacks/` | label flipping, backdoor, update scaling | 04 |
| `aggregation/` | trimmed mean, clustering, static-weight, secure aggregation | 04, 06 |
| `privacy/` | clipping, Gaussian noise, accountant, adaptive budget | 05 |
| `verify/` | quantization, masking, commitments, signatures, verification | 06 |
