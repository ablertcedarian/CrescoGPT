#%%
from collections import OrderedDict

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 100)

import wandb
WANDB_API_KEY = '16f6c877029e38df96355a125195daf0152a3942'

#%%
def flops(run_params):
    vocab_size = 50257
    n_layer, n_embd, n_head, block_size = run_params['n_layer'], run_params['n_embd'], run_params['n_head'], run_params['block_size']
    # we only count Weight FLOPs, all other layers (LayerNorm, Softmax, etc) are effectively irrelevant
    # we count actual FLOPs, not MACs. Hence 2* all over the place
    # basically for any matrix multiply A (BxC) @ B (CxD) -> (BxD) flops are 2*B*C*D

    out = OrderedDict()
    head_size = n_embd // n_head

    # attention blocks
    # 1) the projection to key, query, values
    out['attention/kqv'] = 2 * block_size * (n_embd * 3*n_embd)
    # 2) calculating the attention scores
    out['attention/scores'] = 2 * block_size * block_size * n_embd
    # 3) the reduction of the values (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    out['attention/reduce'] = 2 * n_head * (block_size * block_size * head_size)
    # 4) the final linear projection
    out['attention/proj'] = 2 * block_size * (n_embd * n_embd)
    out['attention'] = sum(out['attention/'+k] for k in ['kqv', 'scores', 'reduce', 'proj'])

    # MLP blocks
    ffw_size = 4*n_embd # feed forward size
    out['mlp/ffw1'] = 2 * block_size * (n_embd * ffw_size)
    out['mlp/ffw2'] = 2 * block_size * (ffw_size * n_embd)
    out['mlp'] = out['mlp/ffw1'] + out['mlp/ffw2']

    # the transformer and the rest of it
    out['block'] = out['attention'] + out['mlp']
    out['transformer'] = n_layer * out['block']
    out['dense'] = 2 * block_size * (n_embd * vocab_size)

    # forward,backward,total
    out['forward_total'] = out['transformer'] + out['dense']
    out['backward_total'] = 2 * out['forward_total'] # use common estimate of bwd = 2*fwd
    out['total'] = (out['forward_total'] + out['backward_total'])
    # out['total'] = np.log((out['forward_total'] + out['backward_total'])) + np.log(run_params['batch_size'] * block_size)
    # * run_params['batch_size'])

    return out

#%%
api = wandb.Api()

run_baseline_tiny = api.run("cresco/runs/5rzutfzr") # baseline-tiny-3-192
run_baseline_mini = api.run("cresco/runs/b8i59ank") # baseline-mini-6-384
run_baseline_small = api.run("cresco/runs/i65xmv6e") # baseline-small-12-768
run_baseline_med = api.run("cresco/runs/c43zi3go") # baseline-med-24-1024
run_baseline_large = api.run("cresco/runs/k5ky18rf") # baseline-large-36-1280
run_baseline_xl = api.run("cresco/runs/ttyi4ny5") # baseline-xl-48-1600

runs = {
    'baseline_tiny': dict(run=run_baseline_tiny, n_layer=3, n_head=3, n_embd=192, block_size=128, vocab_size=50257, batch_size=64),
    'baseline_mini': dict(run=run_baseline_mini, n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=50257, batch_size=64),
    'baseline_small': dict(run=run_baseline_small, n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50257, batch_size=12),
    'baseline_med': dict(run=run_baseline_med, n_layer=24, n_head=16, n_embd=1024, block_size=1024, vocab_size=50257, batch_size=8),
    'baseline_large': dict(run=run_baseline_large, n_layer=36, n_head=20, n_embd=1280, block_size=1024, vocab_size=50257, batch_size=8),
    'baseline_xl': dict(run=run_baseline_xl, n_layer=48, n_head=25, n_embd=1600, block_size=1024, vocab_size=50257, batch_size=8),
}

#%%
for run_key, run in runs.items():
    run_history = run['run'].history()
    # run_history['flops'] = np.log(run_history['iter']) + (flops(run)['total']) # This is log actual flops used (don't use log scale in plots)
    run_history['flops'] = run_history['iter'] * flops(run)['total']
    if run_key == 'baseline_med':
        run_history = run_history.drop([x for x in range(427, 500)])
    elif run_key == 'baseline_small':
        run_history = run_history.drop([x for x in range(74, 86)])
    run_history = run_history.drop([0])
    runs[run_key]['run_flops'] = run_history

# runs['baseline_med']['run_flops'].tail()
# #%%
# runs['baseline_xl']['run_flops'].head(10)
# %%
plt.figure(figsize=(8,6))
for run_key, run in runs.items():
    plt.plot(run['run_flops']['flops'], run['run_flops']['val/loss'], label=run_key)
plt.xscale('log')
plt.legend(loc='upper right')
plt.xlabel('FLOPs')
plt.ylabel('Pre-training Validation Error')
plt.show()
# %%
fig = go.Figure()
for run_key, run in runs.items():
    fig.add_trace(go.Line(
        x=run['run_flops']['flops'],
        y=run['run_flops']['val/loss'],
        name=run_key,
        # log_x=True
    ))

fig.update_layout(xaxis_type='log')
fig.show()
# plt.legend(loc='upper right')
# plt.xlabel('FLOPs')
# plt.ylabel('Pre-training Validation Error')
# plt.show()
# %%
med = run_baseline_med.history()
med['date'] = pd.to_datetime(med['_timestamp'], unit='s')
med.tail(100)
# %%
small = run_baseline_small.history()
small['date'] = pd.to_datetime(small['_timestamp'], unit='s')
# small.tail(65)
small.shape
# %%
large = run_baseline_large.history()
large['date'] = pd.to_datetime(large['_timestamp'], unit='s')
large.head(400)
# %%
