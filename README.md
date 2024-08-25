# PAVR: Preference Aware Vehicle Routing Solutions  using Graph Neural Networks

Run the main script with the following command and options
```
usage: python main.py [-h] [--eta_w ETA_W] [--eta_t ETA_T] [--lookback LOOKBACK] [--d1 D1] [--d2 D2] [--d3 D3] [--attn_heads ATTN_HEADS] [--beta BETA] [--epochs_w EPOCHS_W] [--epochs_t EPOCHS_T] [--warmup WARMUP]
```
The default values for the command follow the given specifications in the thesis for the parameters. For different experiments, the command could be as following for example.

```
usage: python main.py --beta 1 --d1 16 --d2 16 --d3 4 --warmup False
```
