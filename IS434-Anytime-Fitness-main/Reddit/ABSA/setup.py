"""
Setup Script for Aspect-Based Sentiment Analysis
Automates installation and verification of dependencies
"""

import subprocess
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"➜ {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {description} failed")
        print(f"  Error message: {e.stderr}")
        return False

def check_python_version():
    """Verify Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Error: Python 3.8 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True

def install_requirements():
    """Install Python packages from requirements.txt"""
    print_header("Installing Python Packages")
    
    if not os.path.exists('requirements.txt'):
        print("✗ Error: requirements.txt not found")
        return False
    
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing packages from requirements.txt"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def download_spacy_model():
    """Download spaCy language model"""
    print_header("Downloading spaCy Model")
    
    if not run_command(
        "python -m spacy download en_core_web_sm",
        "Downloading en_core_web_sm model"
    ):
        return False
    
    return True

def verify_installation():
    """Verify all packages are installed correctly"""
    print_header("Verifying Installation")
    
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn'),
        ('spacy', 'spacy'),
        ('vaderSentiment', 'vaderSentiment.vaderSentiment'),
    ]
    
    all_ok = True
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} is installed correctly")
        except ImportError:
            print(f"✗ {package_name} failed to import")
            all_ok = False
    
    # Special check for spaCy model
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        print(f"✓ spaCy model 'en_core_web_sm' loaded successfully")
    except:
        print(f"✗ spaCy model 'en_core_web_sm' not available")
        all_ok = False
    
    return all_ok

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    directories = ['data', 'output']
    
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ Created directory: {dir_name}/")
        else:
            print(f"  Directory already exists: {dir_name}/")
    
    return True

def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("  ASPECT-BASED SENTIMENT ANALYSIS - SETUP SCRIPT")
    print("  For Anytime Fitness Singapore Reddit Analysis")
    print("="*70)
    
    steps = [
        (check_python_version, "Python Version Check"),
        (install_requirements, "Package Installation"),
        (download_spacy_model, "spaCy Model Download"),
        (create_directories, "Directory Setup"),
        (verify_installation, "Installation Verification"),
    ]
    
    failed_steps = []
    
    for step_func, step_name in steps:
        if not step_func():
            failed_steps.append(step_name)
    
    print_header("Setup Summary")
    
    if not failed_steps:
        print("✓ Setup completed successfully!")
        print("\nYou can now run the analysis:")
        print("  python aspect_sentiment_spacy_vader.py")
        print("\nMake sure to place your data file in the same directory:")
        print("  reddit_singapore_anytimefitness_comments.csv")
    else:
        print("✗ Setup completed with errors:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease fix the errors and run setup again.")
        return 1
    
    print("\n" + "="*70 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
