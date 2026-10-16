"""Create a working Colab badge once the destination repository is known."""
import argparse
import re
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--repository', required=True, help='owner/repository')
parser.add_argument('--ref', default='main')
args = parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository):
    parser.error('Expected owner/repository')
if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.ref):
    parser.error('Use a branch, tag, or commit without slashes')
url = f'https://colab.research.google.com/github/{args.repository}/blob/{args.ref}/notebooks/benchmark_TCGA_reproducible.ipynb'
root = Path(__file__).resolve().parents[1]
(root/'COLAB.md').write_text(f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n\nRun the first cell to clone the public repository into `/content/imputation-reproducibility`. No ZIP upload is required.\n')
print('Created COLAB.md')
