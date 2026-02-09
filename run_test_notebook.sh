#!/bin/bash
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║          Starting Jupyter Notebook for Local Testing             ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📓 Opening Test_Local_Notebook.ipynb..."
echo ""
echo "Instructions:"
echo "  1. Jupyter will open in your browser"
echo "  2. Click on 'Test_Local_Notebook.ipynb'"
echo "  3. Click 'Cell → Run All' or use Shift+Enter for each cell"
echo "  4. Wait ~5-10 minutes for completion"
echo ""
echo "Press Ctrl+C in this terminal to stop Jupyter when done."
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

cd /Users/mkanaka/git/rsai-a1
./venv/bin/jupyter notebook UAP_Full_Experiment.ipynb
