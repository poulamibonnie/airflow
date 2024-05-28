#!/bin/bash

# deprecated we will move with pytorch 
if [[ "$(uname)" == "Darwin" ]]; then
  pip install tensorflow-macos tensorflow-metal
  python -c "import tensorflow; print(tensorflow.__version__)"
  pip install keras
else
  echo "This script is intended to run on macOS only."
fi
