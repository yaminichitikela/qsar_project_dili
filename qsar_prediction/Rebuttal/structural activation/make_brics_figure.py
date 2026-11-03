import os, io, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

FIGS = '/Users/yamini/Desktop/projects/qsar casepase/qsar_prediction/Rebuttal/figures/'

FAMILIES = [
 ('Phenolic family', [
   ('frag_phenol',             '[16*]c1ccc(O)cc1',        'Para-hydroxyphenyl\n(phenol)'),
   ('frag_hydroxyphenyl',      '[16*]c1ccc(O)c([16*])c1', 'Disubstituted\nhydroxyphenyl'),
   ('frag_methylhydroxyphenyl','[16*]c1cc([16*])c(O)cc1C','Methyl-hydroxyphenyl\n(cresol-like)'),
 ]),
 ('Lipophilic chain family', [
   ('frag_dodecyl',     '[8*]CCCCCCCCCCCC', 'Dodecyl chain\n(linear C12)'),
   ('frag_tertbutyl',   '[8*]C(C)(C)C',     'tert-Butyl\n(branched C4)'),
   ('frag_ethylene',    '[7*]CC[7*]',       'Ethylene bridge\n(−CH₂CH₂−)'),
   ('frag_dodecyl_alt', '[4*]CCCCCCCCCCCC', 'Dodecyl chain alt.\n(linear C12)'),
   ('frag_methylene',   '[7*]C[8*]',        'Methylene bridge\n(−CH₂−)'),
 ]),
 ('Aromatic ring systems', [
   ('frag_phenylene', '[16*]c1ccc([16*])cc1', '1,4-Phenylene\n(para-disubstituted benzene)'),
 ]),
 ('Other enriched fragments', [
   ('frag_organotin',   '[3*][O][Sn]([CH2]CCC)([CH2]CCC)[CH2]CCC', 'Trialkyltin\n(tributylstannyloxy)'),
   ('frag_imidazolium', '[9*]n1cc[n+](C)c1',  'N-Methylimidazolium\n(cationic)'),
   ('frag_ammonium',    '[5*][N+]([5*])(C)C', 'Dimethyl quaternary\nammonium (cationic)'),
 ]),
]

NCOL, NAVY = 3, '#1b365d'
FIG_W      = 13.6
PW         = 2600          # panel width in px
HMAX       = 1850          # cap on molecule height in px
FILL       = 0.97
LABEL_PX   = 430           # reserved strip under each molecule for its name
STROKE     = 7.0
WHITE      = (1.0, 1.0, 1.0, 1.0)
REF        = 70.0

def _draw(frag, bond, canvas):
    mol = Chem.MolFromSmiles(re.sub(r'\[\d+\*\]', '[*]', frag))
    d = rdMolDraw2D.MolDraw2DCairo(*canvas)
    o = d.drawOptions()
    o.setBackgroundColour(WHITE); o.bondLineWidth = STROKE; o.fixedBondLength = bond
    # clamp label size so a low-bond fragment (dodecyl) keeps a legible '*'
    o.minFontSize = int(np.clip(bond*0.62, 120, 190))
    o.maxFontSize = int(np.clip(bond*0.80, 150, 240))
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol); d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB')

def _crop(im, p=6):
    a = np.array(im); ys, xs = np.where(a.astype(int).sum(2) < 740)
    if len(xs) == 0: return im
    return im.crop((max(xs.min()-p,0), max(ys.min()-p,0),
                    min(xs.max()+p,a.shape[1]), min(ys.max()+p,a.shape[0])))

# ---- pass 1: measure every fragment at a common reference bond length ----
meas = {}
for fam, frags in FAMILIES:
    for stem, frag, _ in frags:
        meas[stem] = _crop(_draw(frag, REF, (3000, 2400))).size   # (w, h)

# ---- per-fragment zoom factor, then per-row panel height ----------------
scale = {s: min(PW*FILL/w, HMAX/h) for s, (w, h) in meas.items()}

rows = []   # ('header', name) | ('mols', frags, row_height_px)
for fam, frags in FAMILIES:
    rows.append(('header', fam, 0))
    for i in range(0, len(frags), NCOL):
        chunk = frags[i:i+NCOL]
        rh = max(meas[s][1]*scale[s] for s, _, _ in chunk)
        rows.append(('mols', chunk, rh))

# ---- pass 2: redraw each fragment at its own zoom, pad to panel ----------
def panel(stem, frag, rh):
    s = scale[stem]
    im = _crop(_draw(frag, float(np.clip(REF*s, 18, 400)),
                     (int(PW*1.5), int((rh+200)*1.6))))
    if im.size[0] > PW or im.size[1] > rh:
        k = min(PW/im.size[0], rh/im.size[1])
        im = im.resize((max(1,int(im.size[0]*k)), max(1,int(im.size[1]*k))), Image.LANCZOS)
    box = Image.new('RGB', (PW, int(rh)+LABEL_PX), 'white')
    box.paste(im, ((PW-im.size[0])//2, (int(rh)-im.size[1])//2))
    return np.array(box)

colW_in = FIG_W*0.976/NCOL
px_in   = PW/colW_in
H_HDR   = 0.58
heights = [H_HDR if r[0]=='header' else (r[2]+LABEL_PX)/px_in for r in rows]

fig = plt.figure(figsize=(FIG_W, sum(heights)+0.3), facecolor='white')
gs  = gridspec.GridSpec(len(rows), NCOL, figure=fig, height_ratios=heights,
                        hspace=0.10, wspace=0.03,
                        left=0.012, right=0.988, top=0.99, bottom=0.008)

for ri, r in enumerate(rows):
    if r[0] == 'header':
        ax = fig.add_subplot(gs[ri, :]); ax.axis('off')
        ax.text(0.0, 0.12, r[1], fontsize=21, fontweight='bold', style='italic',
                color=NAVY, ha='left', va='bottom', transform=ax.transAxes)
    else:
        chunk, rh = r[1], r[2]
        for ci in range(NCOL):
            ax = fig.add_subplot(gs[ri, ci]); ax.axis('off')
            if ci < len(chunk):
                stem, frag, name = chunk[ci]
                ax.imshow(panel(stem, frag, rh), aspect='equal', interpolation='lanczos')
                ax.text(PW/2, rh + LABEL_PX*0.42, name, fontsize=16.5, ha='center',
                        va='center', multialignment='center', linespacing=1.32)

out = FIGS + 'fig_brics_enriched_fragments'
fig.savefig(out + '.png', dpi=250, bbox_inches='tight', facecolor='white')
fig.savefig(out + '.pdf',            bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved {out}.png / .pdf  ({FIG_W} x {sum(heights)+0.3:.1f} in)')
