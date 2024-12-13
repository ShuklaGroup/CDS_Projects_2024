## Baseline Model Modules

- **`./plot_loss.ipynb`**:  
  Reads MAE data from `try_guess.log` and `try_FCNN_123.log` to generate a learning curve plot.

- **`./run_try_FCNN.sh` & `./run_try_guess.sh`**:  
  Bash scripts for training BaseM2 (FCNN) and BaseM1 (guess) models.

- **`./try_FCNN.py` & `./try_guess.py`**:  
  Python scripts for implementing training code.

- **`./run2.py`**:  
  A revised version of `../dig/threedgraph/method/run.py`.  
  Updates the `run.val()` function to output formatted loss and MAE results.
