# Model-based Control Strategies for PEM Electrolyzers

Code for the MSc thesis "Model-based Control Strategies for PEM Electrolyzers: A Multi-Objective 
Approach to Operational Efficiency" (DTU Wind & Energy Systems, Aug 2026).

**Requires:** Python 3 (numpy, pandas, matplotlib, scipy, pyomo) + a Gurobi license.

**Structure:**
- `PEMWE_Parameters.py`, `Electrochemical_Model.py`, `Resistance_Calc.py`, `Thermal_Model.py`, `Mass_balance_model.py`, `Lookup_Table_for_OPT.py` — stack physics model
- `ANHOLT_Analysis.py`, `Scenario_search.py` — wind/price data prep and scenario selection
- `mpc_attempt_*.py`, `Single_shot.py`, `Normalization.py` — dispatch optimization formulations
- `read_mpc.py`, `Pareto_front.py` — results and plots

Anholt SCADA data (Ørsted) is confidential and not included.
