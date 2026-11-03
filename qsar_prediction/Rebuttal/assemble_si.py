"""
Assembles the full acs-si-2.tex: keeps the old S1/S2 (Caspase summary) and
old BRICS tables (renumbered), and inserts the new dataset-summary table,
all 19 generated performance/comparison tables (grouped into S4-S10 with
"Part N of M" + \\setcounter{table} the same way the existing BRICS S4 does),
and the new balancing-comparison table.

Usage: python assemble_si.py
Reads: acs-si-2.tex (current working copy, already has preamble + old S1/S2
       intact up to the first \\clearpage after Figure S2), rebuttal_results_tables.tex
Writes: acs-si-2.tex (overwritten, assembled)
"""
import re

BASE = "/Users/yamini/Desktop/projects/qsar casepase/qsar_prediction/Rebuttal/"

# ---- 1. Load the 19 generated table blocks, keyed by label ----
content = open(BASE + 'rebuttal_results_tables.tex').read()
pattern = re.compile(r'(\\begin\{table\*?\}.*?\\end\{table\*?\})', re.DOTALL)
blocks = pattern.findall(content)
by_label = {}
for b in blocks:
    m = re.search(r'label\{([^}]+)\}', b)
    by_label[m.group(1)] = b

# ---- 2. Load the current acs-si-2.tex up through old S1/S2 (stop before \clearpage at end) ----
si = open(BASE + 'acs-si-2.tex').read()
cut_marker = r'\clearpage'
idx = si.rindex(cut_marker)  # the \clearpage right after Figure S2
head = si[:idx + len(cut_marker)]

# ---- helper: set caption to include "(Part N of M)" and force table number ----
def set_part(block, part, total, set_counter_to=None):
    cap_match = re.search(r'\\caption\{', block)
    insert_at = cap_match.end()
    if total > 1:
        block = block[:insert_at] + f'\\textit{{(Part {part} of {total})}} ' + block[insert_at:]
    prefix = f'\\setcounter{{table}}{{{set_counter_to}}}\n' if set_counter_to is not None else ''
    return prefix + block


out = [head, '\n\n']

# ---- Table S3: dataset summary (new) ----
out.append(r"""
\begin{table*}[h]
\centering
\small
\caption{Dataset summary for the three qHTS summary assays used in this study. Active, inactive and inconclusive counts are from the InChIKey-deduplicated, score-averaged cleaned data; rate is the active fraction of the total.}
\label{tab:assay_summary}
\begin{tabular}{lcrrrrr}
\toprule
\textbf{Cell Line} & \textbf{Assay} & \textbf{Active} & \textbf{Inactive} & \textbf{Inconclusive} & \textbf{Total} & \textbf{Rate (\%)} \\
\midrule
HepG2  & Caspase-3/7 (AID 654+1347034) & 282 & 7{,}003 & 431   & 7{,}716 & 3.65 \\
CHO-K1 & Caspase-3/7 (AID 1347037)     & 171 & 7{,}325 & 175   & 7{,}671 & 2.23 \\
HepG2  & MMP summary assay (AID 720637) & 857 & 5{,}080 & 2{,}162 & 8{,}099 & 10.58 \\
\bottomrule
\end{tabular}
\end{table*}

""")

# ---- Table S4: MMP, 3 parts ----
mmp_labels = ['tab:mmp_random', 'tab:mmp_loo', 'tab:mmp_scaffold']
for i, lab in enumerate(mmp_labels):
    blk = set_part(by_label[lab], i + 1, len(mmp_labels), set_counter_to=3 if i > 0 else None)
    out.append(blk + '\n\n')

# ---- Table S5: HepG2 Caspase, 5 parts ----
hepg2_labels = ['tab:hepg2_random', 'tab:hepg2_loo', 'tab:hepg2_scaffold', 'tab:hepg2_up_random', 'tab:hepg2_up_loo']
for i, lab in enumerate(hepg2_labels):
    blk = set_part(by_label[lab], i + 1, len(hepg2_labels), set_counter_to=4 if i > 0 else None)
    out.append(blk + '\n\n')

# ---- Table S6: CHO-K1 Caspase, 5 parts ----
chok1_labels = ['tab:chok1_random', 'tab:chok1_loo', 'tab:chok1_scaffold', 'tab:chok1_up_random', 'tab:chok1_up_loo']
for i, lab in enumerate(chok1_labels):
    blk = set_part(by_label[lab], i + 1, len(chok1_labels), set_counter_to=5 if i > 0 else None)
    out.append(blk + '\n\n')

# ---- Table S7: balancing comparison (new, classical-model mean, CPT=0.5) ----
out.append(r"""
\begin{table*}[h]
\centering
\small
\caption{Comparison of six class-balancing strategies on the two caspase-3/7 datasets, mean of the three classical models (Random Forest, SVM, XGBoost) at CPT = 0.5, single 80/10/10 split. Similarity- and diversity-based selection did not improve on random downsampling; SMOTE could only be run on these fingerprint-based classical models, not the seven graph-based models. Random up-sampling, applicable to all ten models identically, was adopted for the final pipeline.}
\label{tab:balancing_comparison}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{Strategy} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{MCC} & \textbf{F1} \\
\midrule
\multirow{6}{*}{Caspase HepG2 (654+1347034)}
 & 1:1 downsampling   & 0.856 & 0.403 & 0.271 & 0.230 \\
 & 1:5 downsampling   & 0.882 & 0.446 & 0.444 & 0.449 \\
 & Similarity-based   & 0.684 & 0.119 & 0.092 & 0.106 \\
 & Diversity-based    & 0.789 & 0.156 & 0.174 & 0.138 \\
 & SMOTE              & 0.878 & 0.454 & 0.391 & 0.343 \\
 & Up-sampling        & \textbf{0.886} & \textbf{0.465} & \textbf{0.489} & \textbf{0.495} \\
\midrule
\multirow{6}{*}{Caspase CHO-K1 (1347037)}
 & 1:1 downsampling   & 0.717 & 0.074 & 0.118 & 0.096 \\
 & 1:5 downsampling   & \textbf{0.745} & 0.140 & \textbf{0.233} & \textbf{0.242} \\
 & Similarity-based   & 0.504 & 0.031 & -0.011 & 0.042 \\
 & Diversity-based    & 0.654 & 0.074 & 0.069 & 0.062 \\
 & SMOTE              & 0.727 & 0.212 & 0.252 & 0.224 \\
 & Up-sampling        & 0.740 & \textbf{0.234} & 0.159 & 0.144 \\
\bottomrule
\end{tabular}
\end{table*}

""")

# ---- Table S8: DILI, 4 parts ----
dili_labels = ['tab:dili_single_1', 'tab:dili_single_2', 'tab:dili_cv_1', 'tab:dili_cv_2']
for i, lab in enumerate(dili_labels):
    blk = set_part(by_label[lab], i + 1, len(dili_labels), set_counter_to=7 if i > 0 else None)
    out.append(blk + '\n\n')

# ---- Table S9: MV-GNN comparison ----
out.append(by_label['tab:mvgnn_comparison'] + '\n\n')

# ---- Table S10: Xu et al. comparison ----
out.append(by_label['tab:xu_comparison'] + '\n\n')

out.append('\n\\clearpage\n')

# ---- Old BRICS tables, renumbered S11-S13 (same bodies, bump \\setcounter) ----
out.append(r"""
% ============================================================


% The following tables report the full enrichment statistics for all BRICS fragment analyses described in the main text.
% Table~\ref{tab:brics_cellline_enrichment} lists fragments enriched in cell-line-specific active compound sets.
% Table~\ref{tab:brics_strict_enrichment} reports fragments enriched in dual-active (caspase $+$ mitochondrial) versus dual-inactive compounds.
% Table~\ref{tab:brics_t7_enrichment} reports fragments enriched in compounds active in both the mitochondrial membrane potential and cell viability assays versus compounds inactive in both.
% For all tables, odds ratios (OR) and 95\,\% confidence intervals are conditional maximum-likelihood estimates from Fisher's exact test; OR~$= \infty$ denotes fragments absent from the background set; adjusted $p$-values were computed using the Benjamini--Hochberg procedure.


\begin{table*}[h!]
\centering
\small
\caption{%
    Significantly enriched BRICS fragments per cell line among the 323
    Caspase-active compounds (Fisher's exact test, BH-FDR
    $\alpha = 0.10$, minimum foreground count $= 2$). OR~$= \infty$ denotes
    fragments absent from the background set.
    $n_a/N_a$ = count in foreground / foreground total;
    $n_b/N_b$ = count in background / background total.%
}
\label{tab:brics_cellline_enrichment}
\begin{tabular}{lp{2.0cm}lccp{1.8cm}p{1.8cm}}
\hline
\textbf{Fragment name} &
\textbf{Structure} &
\textbf{Cell line} &
\textbf{OR} &
\textbf{$p_{\text{adj}}$} &
$n_a / N_a$ (\%) & $n_b / N_b$ (\%) \\
\hline
\multicolumn{7}{l}{\textit{SK-N-SH only ($N_a=19$, $N_b=304$)}} \\[2pt]
Epoxide &
\includegraphics[width=1.4cm,keepaspectratio]{figures/c_frag_epoxide.png} &
SK-N-SH & 35.6 & 0.05 & 10.5 & 0.3 \\[6pt]
\hline
\multicolumn{7}{l}{\textit{H-4-II-E only ($N_a=19$, $N_b=304$)}} \\[2pt]
1,2,3,3-Tetramethylcyclohexene &
\includegraphics[width=1.4cm,keepaspectratio]{figures/c_frag_tmcyclohex.png} &
H-4-II-E & $\infty$ & 0.001 & 15.8 & 0 \\[6pt]
Acetaldehyde &
\includegraphics[width=1.4cm,keepaspectratio]{figures/c_frag_acetaldehyde.png} &
H-4-II-E & 35.6 & 0.032 & 10.5 & 0.3 \\[6pt]
\hline
\multicolumn{7}{l}{\textit{HEK293 only ($N_a=17$, $N_b=306$)}} \\[2pt]
1,1-Dichloroethane &
\includegraphics[width=1.4cm,keepaspectratio]{figures/c_frag_dichloroethane.png} &
HEK293 & $\infty$ & 0.01 & 11.8 & 0 \\[6pt]
Chlorobenzene &
\includegraphics[width=1.4cm,keepaspectratio]{figures/c_frag_chlorobenzene.png} &
HEK293 & 10.7 & 0.022 & 17.6 & 2.0 \\[6pt]
\hline
\end{tabular}
\end{table*}



%% ══════════════════════════════════════════════════════════════════════════════
%% Analysis A — Table 7: Mito + Viability Both Active vs Both Inactive (17 frags)
%% ══════════════════════════════════════════════════════════════════════════════

\begin{table*}[h!]
\centering
\small
\caption{%
    All 17 BRICS fragments significantly enriched in dual-active
    (mitochondrial membrane potential $+$ cell viability, HepG2)
    versus dual-inactive compounds
    ($n_{\text{active}} = 292$, $n_{\text{inactive}} = 3{,}196$;
    min.\ prevalence $\geq 3$ in active set, OR~$\geq 2$,
    Benjamini--Hochberg FDR $\alpha = 0.01$).
    OR and 95\,\% CI are conditional maximum-likelihood estimates
    (Fisher exact); OR~$= \infty$ denotes
    fragments absent from the dual-inactive set. \% active and \% inactive denote percentages of dual-active and dual-inactive compounds with a fragment respectively.
    \textit{(Part 1 of 2)}%
}
\label{tab:brics_t7_enrichment}
\begin{tabular}{lp{2.2cm}p{3.0cm}cc}
\hline
\textbf{Fragment name} &
\textbf{Structure} &
\textbf{OR (95\,\% CI)} &
\textbf{$p_{\text{adj}}$} &
\textbf{ \% active vs \% inactive} \\
\hline
\multicolumn{5}{l}{\textit{Phenolic family}} \\[2pt]
Para-hydroxyphenyl (phenol) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_phenol.png} &
5.9 (3.1--10.9) & $6.0\times10^{-6}$ & 6.16\% vs 1.10\% \\[8pt]
4-Chlorophenol &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_chlorophenol.png} &
$\infty$ (7.3, $\infty$) & $4.7\times10^{-4}$ & 1.37\% vs 0\% \\[8pt]
Pyrogallol (benzene-1,2,3-triol) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_pyrogallol.png} &
$\infty$ (4.5, $\infty$) & $2.9\times10^{-3}$ & 1.03\% vs 0\% \\[8pt]
\textit{m}-Cresol (3-methylphenol) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_mcresol.png} &
$\infty$ (4.5, $\infty$) & $2.9\times10^{-3}$ & 1.03\% vs 0\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Lipophilic chain family}} \\[2pt]
\textit{tert}-Butyl (branched C4) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_tbutyl.png} &
7.2 (3.1--15.7) & $4.6\times10^{-5}$ & 4.11\% vs 0.59\% \\[8pt]
2,2,4-Trimethylpentyl (branched C8) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_isooctyl.png} &
44.3 (4.4--2186) & $1.7\times10^{-3}$ & 1.37\% vs 0.03\% \\[8pt]
Methine carbon ($-$CH$<$) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_methine.png} &
9.2 (2.2--36.6) & $5.1\times10^{-3}$ & 1.71\% vs 0.19\% \\[8pt]
Octyl chain (linear C8) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_octyl.png} &
7.9 (2.0--29.2) & $7.6\times10^{-3}$ & 1.71\% vs 0.22\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Nitroaromatic family}} \\[2pt]
1,3-Dinitrophenyl &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_dinitrobenzene.png} &
$\infty$ (4.5, $\infty$) & $2.9\times10^{-3}$ & 1.03\% vs 0\% \\[8pt]
Ortho-disubstituted nitrophenyl &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_nitrobenzene_ortho.png} &
33.1 (2.6--1742) & $7.6\times10^{-3}$ & 1.03\% vs 0.03\% \\[8pt]
\hline
\end{tabular}
\end{table*}

\setcounter{table}{11}
\begin{table*}[h!]
\centering
\small
\caption{\textit{(Part 2 of 2)}}
\begin{tabular}{lp{2.2cm}p{3.0cm}cc}
\hline
\textbf{Fragment name} &
\textbf{Structure} &
\textbf{OR (95\,\% CI)} &
\textbf{$p_{\text{adj}}$} &
\textbf{\% active vs \% inactive} \\
\hline
\multicolumn{5}{l}{\textit{Chlorinated aromatic family}} \\[2pt]
Para-chlorophenyl &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_chlorobenzene.png} &
4.2 (1.8--9.0) & $3.0\times10^{-3}$ & 3.42\% vs 0.85\% \\[8pt]
2-Methyl-1,4-disubstituted phenyl &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_tolyl_disub.png} &
33.1 (2.6--1742) & $7.6\times10^{-3}$ & 1.03\% vs 0.03\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Halogenated}} \\[2pt]
Trifluoromethyl &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_cf3.png} &
6.4 (3.0--12.9) & $2.7\times10^{-5}$ & 4.79\% vs 0.78\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Nitrogen-containing}} \\[2pt]
Dimethylamine (tertiary amine) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_dimethylamine.png} &
16.9 (5.3--58.0) & $1.7\times10^{-5}$ & 3.08\% vs 0.19\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Organophosphate family}} \\[2pt]
O,O-Dimethyl thiophosphate &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_dimethylthiophosphate.png} &
55.5 (6.2--2633) & $2.6\times10^{-4}$ & 1.71\% vs 0.03\% \\[8pt]
Phosphate triester &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_phosphate.png} &
18.5 (3.6--120) & $1.5\times10^{-3}$ & 1.71\% vs 0.09\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Organometallic}} \\[2pt]
Tributyltin alkoxide$^\dagger$ &
\includegraphics[width=0.9cm,keepaspectratio]{figures/t7_frag_tributyltin.png} &
$\infty$ (4.5, $\infty$) & $2.9\times10^{-3}$ & 1.03\% vs 0\% \\[8pt]
\hline
\end{tabular}
\end{table*}


\begin{table*}[h!]
\centering
\small
\caption{%
    Top twelve BRICS fragments significantly enriched in
    mitochondria-active, viability-inactive compounds versus
    dual-inactive compounds
    ($n_{\text{mito-only}} = 680$, $n_{\text{inactive}} = 3{,}196$;
    min.\ prevalence $\geq 3$ in foreground,
    Benjamini--Hochberg FDR $\alpha = 0.05$).
    OR and 95\,\% CI are conditional maximum-likelihood estimates
    (Fisher exact); OR~$= \infty$ denotes
    fragments absent from the dual-inactive set. \% mito-only and \% inactive denote percentages of mitochondria-active, viability-inactive and dual-inactive compounds with a fragment respectively.
}
\label{tab:brics_mito_only_enrichment}
\begin{tabular}{lp{2.2cm}p{3.2cm}cc}
\hline
\textbf{Fragment name} &
\textbf{Structure} &
\textbf{OR (95\,\% CI)} &
\textbf{$p_{\text{adj}}$} &
\textbf{\% mito-only vs \% inactive} \\
\hline
\multicolumn{5}{l}{\textit{Phenolic family}} \\[2pt]
Para-hydroxyphenyl (phenol) &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_phenol.png} &
4.0\ (2.4--6.8) & $2.4\times10^{-5}$ & 4.26\% vs 1.10\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Complex ring / sugar family}} \\[2pt]
2-Methyltetrahydropyran-4-ol &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_methylthp.png} &
$\infty$\ (6.8, $\infty$) & $1.5\times10^{-4}$ & 1.03\% vs 0\% \\[8pt]
4-Amino-2-methyloxan-3-ol &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_aminosugar.png} &
$\infty$\ (4.3, $\infty$) & $1.7\times10^{-3}$ & 0.74\% vs 0\% \\[8pt]
Anthracyclinone &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_anthracycline.png} &
$\infty$\ (4.3, $\infty$) & $1.7\times10^{-3}$ & 0.74\% vs 0\% \\[8pt]
Methyldecalin &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_methyldecalin.png} &
28.4\ (3.4, $\infty$) & $1.7\times10^{-3}$ & 0.88\% vs 0.03\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Lipophilic chain family}} \\[2pt]
Ethane &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_ethane.png} &
7.2\ (3.0--18.0) & $1.4\times10^{-4}$ & 2.21\% vs 0.31\% \\[8pt]
Isobutane &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_isobutane.png} &
4.8\ (2.4--9.6) & $1.5\times10^{-4}$ & 2.79\% vs 0.59\% \\[8pt]
Decane &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_decane.png} &
$\infty$\ (4.3, $\infty$) & $1.7\times10^{-3}$ & 0.74\% vs 0\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Carbonyl family}} \\[2pt]
Formaldehyde &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_formaldehyde.png} &
5.5\ (2.5--12.2) & $2.1\times10^{-4}$ & 2.35\% vs 0.44\% \\[8pt]
Methyl ethyl ketone &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_mek.png} &
$\infty$\ (4.3, $\infty$) & $1.7\times10^{-3}$ & 0.74\% vs 0\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Halogenated family}} \\[2pt]
Chloroacetaldehyde &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_chloroacetaldehyde.png} &
28.4\ (3.4, $\infty$) & $1.7\times10^{-3}$ & 0.88\% vs 0.03\% \\[8pt]
\hline
\multicolumn{5}{l}{\textit{Heteroaromatic family}} \\[2pt]
Benzimidazole &
\includegraphics[width=0.9cm,keepaspectratio]{figures/frag_benzimidazole.png} &
28.4\ (3.4, $\infty$) & $1.7\times10^{-3}$ & 0.88\% vs 0.03\% \\[8pt]
\hline
\end{tabular}
\end{table*}

""")

out.append(r"""

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using classical BibTeX rather than biblatex,
%% remove the \printbibliography and uncomment the \bibliograpy one
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\printbibliography
% \bibliography{reference.bib}


\end{document}
""")

final = ''.join(out)
with open(BASE + 'acs-si-2.tex', 'w') as f:
    f.write(final)

print(f'Assembled {len(final)} chars -> acs-si-2.tex')
