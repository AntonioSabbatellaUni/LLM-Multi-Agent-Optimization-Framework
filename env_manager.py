"""
Environment Configuration Manager for LLM Multi-Agent Optimization Framework

This module handles loading and validation of environment variables required for
different architectures, especially for GAIA real benchmark evaluation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Try to import python-dotenv, install if not available
try:
    from dotenv import load_dotenv
except ImportError:
    print("⚠️  python-dotenv not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv


class EnvironmentManager:
    """Manages environment variables for the optimization framework."""
    
    # Required environment variables for different architectures
    REQUIRED_VARS = {
        'gaia_smolagents': {
            'critical': ['HF_TOKEN'],  # Must have for basic functionality
            'recommended': ['SERPAPI_API_KEY'],  # Needed for search functionality
            'optional': ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
        },
        'simulated': {
            'critical': [],  # No API keys needed for simulated
            'recommended': [],
            'optional': []
        }
    }
    
    def __init__(self, framework_root: Optional[Path] = None):
        """
        Initialize environment manager.
        
        Args:
            framework_root: Root directory of the framework (auto-detected if None)
        """
        if framework_root is None:
            # Auto-detect framework root (look for config.yaml)
            current = Path(__file__).parent
            while current != current.parent:
                if (current / 'config.yaml').exists():
                    framework_root = current
                    break
                current = current.parent
            else:
                framework_root = Path.cwd()
        
        self.framework_root = framework_root
        self.env_file = self.framework_root / '.env'
        self.env_template = self.framework_root / '.env.template'
        
        # Load environment variables
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables from .env file if it exists."""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            print(f"✅ Environment variables loaded from: {self.env_file}")
        else:
            print(f"⚠️  No .env file found at: {self.env_file}")
            if self.env_template.exists():
                print(f"💡 Template available at: {self.env_template}")
                print(f"   Copy and rename to .env, then fill in your API keys")
    
    def validate_architecture(self, architecture: str) -> Dict[str, List[str]]:
        """
        Validate environment variables for a specific architecture.
        
        Args:
            architecture: Architecture name ('gaia_smolagents', 'simulated')
            
        Returns:
            Dictionary with 'missing_critical', 'missing_recommended', 'available'
        """
        if architecture not in self.REQUIRED_VARS:
            return {
                'missing_critical': [],
                'missing_recommended': [],
                'available': [],
                'warnings': [f'Unknown architecture: {architecture}']
            }
        
        arch_vars = self.REQUIRED_VARS[architecture]
        result = {
            'missing_critical': [],
            'missing_recommended': [],
            'available': [],
            'warnings': []
        }
        
        # Check critical variables
        for var in arch_vars['critical']:
            if os.getenv(var):
                result['available'].append(var)
            else:
                result['missing_critical'].append(var)
        
        # Check recommended variables
        for var in arch_vars['recommended']:
            if os.getenv(var):
                result['available'].append(var)
            else:
                result['missing_recommended'].append(var)
        
        # Check optional variables
        for var in arch_vars['optional']:
            if os.getenv(var):
                result['available'].append(var)
        
        return result
    
    def print_validation_report(self, architecture: str):
        """Print a detailed validation report for an architecture."""
        validation = self.validate_architecture(architecture)
        
        print(f"\n🔐 ENVIRONMENT VALIDATION for '{architecture.upper()}' architecture")
        print("=" * 60)
        
        # Critical variables
        if validation['missing_critical']:
            print("❌ MISSING CRITICAL VARIABLES:")
            for var in validation['missing_critical']:
                print(f"   • {var} - REQUIRED for basic functionality")
        
        # Recommended variables
        if validation['missing_recommended']:
            print("⚠️  MISSING RECOMMENDED VARIABLES:")
            for var in validation['missing_recommended']:
                print(f"   • {var} - Needed for full functionality")
        
        # Available variables
        if validation['available']:
            print("✅ AVAILABLE VARIABLES:")
            for var in validation['available']:
                value = os.getenv(var, '')
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                print(f"   • {var} = {masked_value}")
        
        # Warnings
        if validation['warnings']:
            print("⚠️  WARNINGS:")
            for warning in validation['warnings']:
                print(f"   • {warning}")
        
        # Summary
        critical_ok = len(validation['missing_critical']) == 0
        if critical_ok:
            print("✅ Architecture can proceed with current environment")
        else:
            print("❌ Architecture requires additional environment setup")
            
        return critical_ok
    
    def setup_environment_for_architecture(self, architecture: str) -> bool:
        """
        Ensure environment is properly set up for an architecture.
        
        Args:
            architecture: Architecture name
            
        Returns:
            True if environment is ready, False if setup needed
        """
        validation = self.validate_architecture(architecture)
        
        if validation['missing_critical']:
            print(f"\n🔧 SETTING UP ENVIRONMENT for {architecture.upper()}")
            print("-" * 50)
            
            if not self.env_file.exists():
                print("📝 Creating .env file from template...")
                if self.env_template.exists():
                    import shutil
                    shutil.copy2(self.env_template, self.env_file)
                    print(f"✅ Created: {self.env_file}")
                else:
                    print(f"❌ Template not found: {self.env_template}")
                    return False
            
            print("\n🔑 Please edit .env file and add your API keys:")
            for var in validation['missing_critical']:
                print(f"   • {var}")
            
            print(f"\n📝 Edit file: {self.env_file}")
            print("Then restart your optimization process.")
            return False
        
        return True
    
    def is_ready_for_architecture(self, architecture: str) -> bool:
        """Quick check if environment is ready for an architecture."""
        validation = self.validate_architecture(architecture)
        return len(validation['missing_critical']) == 0


# Global environment manager instance
env_manager = EnvironmentManager()


def validate_environment_for_architecture(architecture: str) -> bool:
    """
    Convenience function to validate environment for an architecture.
    
    Args:
        architecture: Architecture name
        
    Returns:
        True if environment is ready
    """
    return env_manager.print_validation_report(architecture)


def ensure_environment_ready(architecture: str) -> bool:
    """
    Convenience function to ensure environment is ready for an architecture.
    
    Args:
        architecture: Architecture name
        
    Returns:
        True if ready to proceed, False if user action required
    """
    return env_manager.setup_environment_for_architecture(architecture)


if __name__ == "__main__":
    # Test the environment manager
    print("🧪 Testing Environment Manager")
    
    # Test both architectures
    for arch in ['simulated', 'gaia_smolagents']:
        env_manager.print_validation_report(arch)
        print()
