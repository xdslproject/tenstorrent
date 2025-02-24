import sys
import subprocess


def run_from_examples(file_name):
    """
    Runs a filecheck test case from the examples directory using the same
    interpreter as the caller.
    """
    file_name = file_name.replace("tests/filecheck", "examples")
    subprocess.run([sys.executable, file_name], check=True)
