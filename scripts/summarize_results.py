#!/usr/bin/env python3
"""Rebuild the seed-level report from an immutable experiment plan."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from roma.evaluation.statistics import report
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
plan=json.loads((args.output/'plan.json').read_text())
if len(plan['seeds'])<2:
    p.error('At least two seeds are required for mean ± sample standard deviation')
report(plan,args.output)
