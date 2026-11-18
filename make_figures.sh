#!/usr/bin/env bash
# Regenerate all figures of the paper; each PDF is written next to its script.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}

(cd fig1_schematic && $PY generate_tGQCE_schematic.py && rsvg-convert -f pdf -o Fig1.pdf tGQCE_schematic.svg)
$PY fig2_B1_ibmq/plot_fig2.py
$PY fig3_B357_avqds/plot_fig3.py
$PY fig4_measurement/plot_fig4.py
echo "Written: fig1_schematic/Fig1.pdf fig2_B1_ibmq/Fig2.pdf fig3_B357_avqds/Fig3.pdf fig4_measurement/Fig4.pdf"
