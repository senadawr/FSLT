# Model training

`ml.models.build_recurrent_classifier` builds equivalent GRU and LSTM isolated
sign baselines. Training code should use reproducible seeds, signer-aware
splits, early stopping, best-checkpoint saving, class weights where justified,
and report macro F1 plus per-class metrics. TensorFlow is intentionally not a
runtime dependency of the API until a model is configured.

The improved Dataset A training command is:

```powershell
.\.venv\Scripts\python.exe ml\prepare_fsl105.py
.\.venv\Scripts\python.exe ml\train_fsl105.py --architecture gru --epochs 30 --batch-size 32
.\.venv\Scripts\python.exe ml\train_fsl105.py --architecture lstm --epochs 30 --batch-size 32
```
