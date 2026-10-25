# Model Architecture — ClimateSense MAF-Net (selected configuration)

Selected run: `final-mafnet-4ac9637255` (seed 42, chosen by validation macro-F1 among five seeds). Checkpoint: `models/checkpoints/final-mafnet-4ac9637255.pt` (copy in `models/best/`).

## Hyper-parameters

| Parameter | Value |
|---|---|
| lookback | 14 |
| horizon | 1 |
| encoder | transformer |
| air_encoder | gru |
| dry_encoder | gru |
| hidden | 32 |
| d_model | 32 |
| layers | 1 |
| heads | 2 |
| dropout | 0.3 |
| modality_dropout | 0.0 |
| token_self_attention | True |
| fusion | attention |
| use_context | True |
| loss | focal |
| optimizer | adamw |
| lr | 0.001 |
| weight_decay | 0.0 |
| batch_size | 256 |
| epochs | 40 |
| patience | 6 |
| seed | 42 |
| best epoch | 10 of 16 run |

**Total trainable parameters: 33,268**

## Parameters per module

| Module | Parameters |
|---|---|
| `encoders` | 13,920 |
| `proj` | 2,336 |
| `ctx_enc` | 752 |
| `self_attn` | 8,544 |
| `cross` | 4,224 |
| `gate` | 2,080 |
| `head` | 1,252 |

## Tensor shapes (batch size B)

| Stage | Shape |
|---|---|
| Input window (all model features) | B × 14 × 38 |
| climate branch input → encoder output | B × 14 × 18 → B × 32 |
| air branch input → encoder output | B × 14 × 10 → B × 16 |
| dryness branch input → encoder output | B × 14 × 10 → B × 16 |
| context input → MLP output | B × 8 → B × 32 |
| modality tokens (after projection + embedding) | B × 4 × 32 |
| attention weights over modalities | B × 4 |
| fused vector → logits → probabilities | B × 32 → B × 4 → B × 4 |

## Features per branch

* **climate** (18): `temp_c`, `feels_like_c`, `rh`, `dewpoint_c`, `heat_index_c`, `wet_bulb_c`, `wind_kph`, `gust_kph`, `pressure_hpa`, `cloud`, `visibility_km`, `uv_index`, `temp_anom_bk`, `obs_hour`, `doy_sin`, `doy_cos`, `heat_score`, `climate_missing`
* **air** (10): `log_co`, `log_o3`, `log_no2`, `log_so2`, `log_pm25`, `log_pm10`, `us_epa_index`, `gb_defra_index`, `air_score`, `air_missing`
* **dryness** (10): `vpd_kpa`, `log_precip`, `wet_flag`, `precip_sum_7d`, `precip_sum_30d`, `wet_frac_7d`, `wet_frac_30d`, `days_since_wet`, `dry_score`, `dry_missing`
* **context** (8): `lat_scaled`, `abs_lat_scaled`, `lon_sin`, `lon_cos`, `log_population`, `bk_normal_annual`, `bk_seasonal_amplitude`, `bk_warming_recent_minus_normal`

## Module listing

```
MAFNet(
  (encoders): ModuleDict(
    (climate): TemporalEncoder(
      (inp): Linear(in_features=18, out_features=32, bias=True)
      (pos): PositionalEncoding()
      (net): TransformerEncoder(
        (layers): ModuleList(
          (0): TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
            )
            (linear1): Linear(in_features=32, out_features=64, bias=True)
            (dropout): Dropout(p=0.3, inplace=False)
            (linear2): Linear(in_features=64, out_features=32, bias=True)
            (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
            (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
            (dropout1): Dropout(p=0.3, inplace=False)
            (dropout2): Dropout(p=0.3, inplace=False)
          )
        )
      )
      (proj): Linear(in_features=64, out_features=32, bias=True)
    )
    (air): TemporalEncoder(
      (net): GRU(10, 16, batch_first=True)
    )
    (dryness): TemporalEncoder(
      (net): GRU(10, 16, batch_first=True)
    )
  )
  (proj): ModuleDict(
    (climate): Sequential(
      (0): Linear(in_features=32, out_features=32, bias=True)
      (1): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (air): Sequential(
      (0): Linear(in_features=16, out_features=32, bias=True)
      (1): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (dryness): Sequential(
      (0): Linear(in_features=16, out_features=32, bias=True)
      (1): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
    )
  )
  (ctx_enc): Sequential(
    (0): Linear(in_features=8, out_features=16, bias=True)
    (1): GELU(approximate='none')
    (2): Linear(in_features=16, out_features=32, bias=True)
    (3): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
  )
  (self_attn): TransformerEncoderLayer(
    (self_attn): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
    )
    (linear1): Linear(in_features=32, out_features=64, bias=True)
    (dropout): Dropout(p=0.3, inplace=False)
    (linear2): Linear(in_features=64, out_features=32, bias=True)
    (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
    (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
    (dropout1): Dropout(p=0.3, inplace=False)
    (dropout2): Dropout(p=0.3, inplace=False)
  )
  (cross): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
  )
  (gate): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): Sigmoid()
  )
  (head): Sequential(
    (0): LayerNorm((32,), eps=1e-05, elementwise_affine=True, bias=True)
    (1): Linear(in_features=32, out_features=32, bias=True)
    (2): GELU(approximate='none')
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=32, out_features=4, bias=True)
  )
)
```

Diagram: `results/figures/architecture_mafnet.png`; equations: IEEE report §VI and notebook 07.