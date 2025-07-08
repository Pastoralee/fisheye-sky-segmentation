import os
import subprocess
import sys
import platform

REQUIRED_PYTHON = (3, 12)
MAX_SUPPORTED_PYTHON = (3, 13)

def check_python_version():
    current = sys.version_info
    if current < REQUIRED_PYTHON:
        print(f"Warning: Python <= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} is not officially supported/tested. You are using {platform.python_version()}.")
    if current >= MAX_SUPPORTED_PYTHON:
        print(f"Warning: Python >= {MAX_SUPPORTED_PYTHON[0]}.{MAX_SUPPORTED_PYTHON[1]} is not officially supported/tested. You are using {platform.python_version()}.")

def create_virtualenv(env_name="venv"):
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", env_name])
    print(f"Virtual environment created in ./{env_name}")

def install_requirements(env_name="venv"):
    pip_executable = (
        os.path.join(env_name, "Scripts", "pip.exe") if os.name == "nt"
        else os.path.join(env_name, "bin", "pip")
    )

    if not os.path.exists("requirements.txt"):
        sys.exit("requirements.txt not found!")

    print("Installing dependencies...")

    try:
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to upgrade pip. Continuing anyway...")
    
    try:
        subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])
        print("All dependencies installed.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Failed to install dependencies: {e}")

def setup_jupyter(env_name="venv"):
    python_executable = (
        os.path.join(env_name, "Scripts", "python.exe") if os.name == "nt"
        else os.path.join(env_name, "bin", "python")
    )

    print("Registering virtual environment as Jupyter kernel...")
    try:
        subprocess.check_call([
            python_executable, "-m", "ipykernel", "install",
            "--user", "--name", env_name, "--display-name", f"Python ({env_name})"
        ])
        print("Jupyter kernel registered!")
        print("To launch Jupyter Notebook, activate the environment and run:")
        print(f"    jupyter-notebook")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install or register Jupyter: {e}")

def main():
    print("Python environment setup starting...")
    check_python_version()
    create_virtualenv()
    install_requirements()
    setup_jupyter()
    print("\nSetup complete! Activate your environment using:")
    if os.name == "nt":
        print("    ./venv/Scripts/activate")
    else:
        print("    source venv/bin/activate")

if __name__ == "__main__":
    main()