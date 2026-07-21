# Note: Porter Stemming has been upgraded to POS-aware WordNet Lemmatization in 05_lemmatization.py
import os
import subprocess
import sys

print("Executing 05_lemmatization.py (Stemming upgraded to Lemmatization)...")
current_dir = os.path.dirname(os.path.abspath(__file__))
lemmatize_script = os.path.join(current_dir, "05_lemmatization.py")
subprocess.run([sys.executable, lemmatize_script], check=True)
