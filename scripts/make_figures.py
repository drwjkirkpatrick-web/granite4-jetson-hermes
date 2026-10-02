#!/usr/bin/env python3
"""Generate the 5 paper figures from paper_data.json (real results only)."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = os.path.expanduser('~/projects/granite4-jetson-hermes/results')
figdir = os.path.join(base, 'figures')
os.makedirs(figdir, exist_ok=True)
d = json.load(open(os.path.join(base, 'paper_data.json')))

plt.rcParams.update({'font.size': 9, 'figure.dpi': 200, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.axisbelow': True})

# ── Fig 1: BFCL v2 overall accuracy (17 models) ──────────────────────
bfcl = {k: v for k, v in sorted(d['bfcl'].items(), key=lambda kv: -kv[1]['pct'])}
names = list(bfcl.keys())
pcts = [bfcl[n]['pct'] for n in names]
chained = [int(bfcl[n]['chained'].split('/')[0]) for n in names]

fig, ax = plt.subplots(figsize=(7.0, 3.6))
colors = ['#2e7d32' if p == max(pcts) else ('#90a4ae' if p < 60 else '#1976d2') for p in pcts]
bars = ax.bar(range(len(names)), pcts, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('BFCL v2 overall accuracy (%)')
ax.set_title('Function-calling accuracy, 17 small models on-device (52 tests)', fontsize=8.5)
for i, (b, p) in enumerate(zip(bars, pcts)):
    ax.text(b.get_x()+b.get_width()/2, p+1, f'{p:.0f}', ha='center', fontsize=6.5)
for i, c in enumerate(chained):
    ax.text(i, 4, f'C{c}', ha='center', fontsize=5.5, color='white', fontweight='bold', rotation=90)
fig.tight_layout()
fig.savefig(f'{figdir}/fig1_bfcl.png')
plt.close()

# ── Fig 2: speed frontier scatter ────────────────────────────────────
speed = d['speed']
arch_colors = {'hybrid conv+GQA': '#e65100', 'hybrid Mamba2': '#2e7d32',
               'dense': '#1565c0', 'looped dense': '#6a1b9a'}
fig, ax = plt.subplots(figsize=(6.0, 3.8))
for name, v in speed.items():
    c = arch_colors.get(v['arch'], '#616161')
    star = '*' if 'DSpark' in name else 'o'
    sz = max(40, v['file_gb']*45)
    ax.scatter(v['gen'], v['pp'], s=sz, c=c, marker=star, alpha=0.85,
               edgecolors='black', linewidths=0.6, zorder=3)
    short = (name.replace(' + DSpark', '+DS').replace('Granite 4.2-8B IQ3_M', 'G4.2-8B')
             .replace('Ornith-1.0-9B IQ3_M', 'Ornith-9B'))
    ax.annotate(short, (v['gen'], v['pp']), textcoords='offset points', xytext=(6, 4), fontsize=6.8)
ax.set_xlabel('Generation speed (tok/s, short prompts)')
ax.set_ylabel('Prompt processing (tok/s)')
ax.set_title('Speed frontier on Jetson Orin Nano 8GB: decode vs prefill', fontsize=9)
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=7, label=a)
           for a, c in arch_colors.items()]
ax.legend(handles=handles, fontsize=6.5, loc='lower right')
fig.tight_layout()
fig.savefig(f'{figdir}/fig2_speed_frontier.png')
plt.close()

# ── Fig 3: DSpark gain ────────────────────────────────────────────────
ds = d['dspark']
fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.bar(['target\nalone', 'with DSpark\n(n_max=3)'], [ds['target_alone'], ds['with_dspark']],
       color=['#90a4ae', '#e65100'], width=0.55, edgecolor='white')
ax.set_ylabel('tok/s')
ax.set_title('Speculative decoding gain\n(LFM2.5-VL-3B, 32K ctx)', fontsize=8)
for x, v in enumerate([ds['target_alone'], ds['with_dspark']]):
    ax.text(x, v+0.5, f'{v:.1f}', ha='center', fontsize=8)
ax.set_ylim(0, 42)
fig.tight_layout()
fig.savefig(f'{figdir}/fig3_dspark_gain.png')
plt.close()

# ── Fig 4: Granite profiles ───────────────────────────────────────────
gp = d['granite_profiles']
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
prof_names = list(gp.keys())
rss = [gp[p]['rss'] for p in prof_names]
free = [gp[p]['free'] for p in prof_names]
kv_types = [gp[p]['kv'] for p in prof_names]
colors_kv = {'f16': '#0d47a1', 'q8_0': '#1565c0', 'q4_0': '#42a5f5'}
axes[0].bar(prof_names, rss, color=[colors_kv[k] for k in kv_types], edgecolor='white')
axes[0].bar(prof_names, free, bottom=rss, color='#c8e6c9', edgecolor='white')
axes[0].axhline(7.4, color='#c62828', ls='--', lw=1, label='physical RAM')
axes[0].set_ylabel('GB'); axes[0].set_title('Memory by profile (RSS + free)', fontsize=8)
axes[0].legend(fontsize=6, loc='upper left')
axes[0].text(0.02, 7.45, '7.4 GB total', fontsize=6, color='#c62828')
for i, (r, f) in enumerate(zip(rss, free)):
    axes[0].text(i, r+f+0.15, f'{r+f:.1f}', ha='center', fontsize=6.5)
axes[1].bar(prof_names, [gp[p]['gen'] for p in prof_names], color='#2e7d32', edgecolor='white')
axes[1].set_ylabel('tok/s'); axes[1].set_title('Gen speed by profile', fontsize=8)
for i, g in enumerate([gp[p]['gen'] for p in prof_names]):
    axes[1].text(i, g+0.3, f'{g}', ha='center', fontsize=6.5)
fig.tight_layout()
fig.savefig(f'{figdir}/fig4_granite_profiles.png')
plt.close()

# ── Fig 5: KV-quant context ceilings ──────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.0))
kv_data = [
    ('f16', 32, 'ok'), ('q8_0', 49, 'ok under load'),
    ('q8_0', 64, 'standalone only'), ('q8_0', 82, 'dead'),
    ('q8_0', 96, 'dead'), ('q8_0', 128, 'dead'), ('q4_0', 128, 'ok'),
]
cmap = {'ok': '#2e7d32', 'ok under load': '#558b2f', 'standalone only': '#f9a825', 'dead': '#c62828'}
for i, (kv, ctx, status) in enumerate(kv_data):
    ax.bar(i, ctx, color=cmap[status], edgecolor='white')
    ax.text(i, ctx+2, f'{ctx}K', ha='center', fontsize=7)
    ax.text(i, 3, kv, ha='center', fontsize=6.5, rotation=90, color='white', fontweight='bold')
ax.set_xticks(range(len(kv_data)))
ax.set_xticklabels(['f16/32K', 'q8/49K', 'q8/64K', 'q8/82K', 'q8/96K', 'q8/128K', 'q4/128K'], fontsize=7)
ax.set_ylabel('Context window (K tokens)')
ax.set_title('Granite 4.0-3B KV-quant context ceilings (measured)', fontsize=9)
fig.tight_layout()
fig.savefig(f'{figdir}/fig5_kv_ceilings.png')
plt.close()

print("All 5 figures:", sorted(os.listdir(figdir)))