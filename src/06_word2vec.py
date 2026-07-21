# Note: Word2Vec training has been moved to 07_word2vec.py after 06_train_test_split.py to prevent data leakage.
import os
import subprocess
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
split_script = os.path.join(current_dir, "06_train_test_split.py")
w2v_script = os.path.join(current_dir, "07_word2vec.py")

print("Executing Train-Test Split (06_train_test_split.py)...")
subprocess.run([sys.executable, split_script], check=True)

print("Executing Word2Vec Training on Train Set ONLY (07_word2vec.py)...")
subprocess.run([sys.executable, w2v_script], check=True)