#!/bin/bash
set -x
mkdir -p ~/.config/Ultralytics
if ! test -f ~/.config/Ultralytics/Arial.ttf
then
	mv Arial.ttf ~/.config/Ultralytics/Arial.ttf
fi
python3 -m dvc pull -f -R -j 4 || (echo "DVC pull did not finish without error:" && jisap-cli logs dvc && false)
export WANDB_MODE="disabled"
dvc repro -v
