"""Ancienne commande conservée : exécute le script Avis de films."""
from pathlib import Path
import runpy
import sys

racine = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(racine))
runpy.run_path(str(racine / 'entrainer_avis.py'), run_name='__main__')
