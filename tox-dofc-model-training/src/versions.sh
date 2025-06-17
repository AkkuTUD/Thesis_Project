#!/bin/bash
echo "DVC $(dvc -V)"
echo "jisap-cli $(jisap-cli version)"
python3 -V
python3 -c "import torch; print(\"PyTorch\", torch.__version__)"
python3 -c "import torchvision; print(\"TorchVision\", torchvision.__version__)"
# python3 -c "import jisap.tracking; print(\"JISAP Tracking\", jisap.tracking.__version__)"
# python3 -c "import jisap.objectdetection; print(\"JISAP Object-Detection\", jisap.objectdetection.__version__)"

