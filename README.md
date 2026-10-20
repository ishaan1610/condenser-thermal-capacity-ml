# Condenser performance and thermal capacity assessment

Python analysis and machine learning extension of Heat and Mass Transfer Mini Project 7 using ASHRAE RP-1043 data.

## Setup

Use Python 3.12. Download this repository and place your ASHRAE RP-1043.zip archive in its root directory.

```text
python -m pip install -r requirements.txt
python scripts/run_analysis.py
python scripts/check_results.py
```

Alternatively, open Condenser_Study.ipynb and run the cells in order. The original dataset is an external input and is not redistributed in this code repository. The ZIP must have the Chiller Data workbook layout used by the supplied project archive.

## Outputs

The pipeline reads 13 Steady States worksheets covering benchmarks, condenser-flow reductions and tube blockage. It produces cleaned SI data, a quality audit, six scientific figures, thermal capacity and minimum-flow calculations, grouped model evaluation, factor ranking, ten what-if scenarios and engineering recommendations under output/. It saves the final Random Forest separately from the train-only evaluation model.

The target is condenser approach temperature in kelvin. Inputs are water mass flow in kg/s, entering condenser-water temperature in C, evaporator cooling load in kW, leaving evaporator-water temperature in C, and known blocked-tube fraction.

```text
python scripts/predict.py --flow 16.9 --inlet 24 --load 191.1 --evap-out 6.74 --blockage 0
```

## Validation and interpretation

The reference run retains 350 of 351 records after physical checks. Whole nominal operating conditions are held out. The Random Forest achieves approximately 0.169 K test MAE and 0.920 test R squared. Ridge gives similar performance. No generated observations enter training or testing.

The calculated thermal bound is not an observed failure threshold. The 0.85 effectiveness screen is illustrative rather than a validated control trigger. Feature importance and what-if predictions are predictive associations, not causal effects. Pump affinity-law estimates apply to pump power alone. The supplied lab report is treated separately and its heater power and assumed water flow need energy-balance verification.

## Sources

- ASHRAE RP-1043 source archive and test notes, provided separately.
- Jin et al. 2019, variable definitions: https://arxiv.org/abs/1902.06366
- ASHRAE Handbook, Condensers: https://handbook.ashrae.org/Handbooks/S20/IP/S20_Ch39/S20_ch39_ip.aspx
- ASHRAE Handbook, Centrifugal Pumps: https://handbook.ashrae.org/Handbooks/S20/SI/S20_Ch44/S20_ch44_si.aspx
