# OptimizationMethodsForPathPlanning

ECE 509 (Spring 2026) term project. 2D point robot path planning with potential fields.

Team: Adam D'Souza, Nihal Abdul Muneer, Vineal Sunkara (POC)

## Layout
- `src/` map env, potential field, optimizers
- `maps/` test environments (JSON)
- `scripts/` runnable tests
- `results/` output figures
- `docs/` proposal + instructions

## Run
```
pip install -r requirements.txt
python scripts/test_path_planning.py
```

## Methods (planned)
gradient descent, GD + momentum, Newton, quasi-Newton (BFGS), augmented Lagrangian
