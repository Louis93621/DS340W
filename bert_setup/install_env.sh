#!/bin/bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
echo ">>> Installing Python 3.10.12... This may take a while."
pyenv install 3.10.12
pyenv global 3.10.12
echo ">>> Python environment setup complete! Current version is:"
python --version