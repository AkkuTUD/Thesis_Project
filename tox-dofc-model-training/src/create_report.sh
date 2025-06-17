(
echo "# Versions overview"
echo
echo "\`\`\`"
src/versions.sh
echo "\`\`\`"
echo
echo "# Changed params"
echo
dvc params diff --md HEAD^  || echo "Parameter diff report could not be created"

echo "# Metrics"
echo
dvc metrics diff --md --all || echo "Metrics diff report could not be created"
echo
echo "# Plots"
echo

for f in $(find runs/train/exp/ | grep jpg)
do
    echo
    echo "## $f"
    echo
    jisap-cli publish --content-type "image/jpeg" "$f"
    echo
done

for f in $(find runs/train/exp/ | grep png)
do
    echo
    echo "## $f"
    echo
    jisap-cli publish --content-type "image/png" "$f"
    echo
done

echo "### Trained model"
echo
echo "[runs/train/exp/weights/last.pt]($(cat runs/train/exp/weights/last.pt | jisap-cli publish --url))"
echo
echo "[runs/train/exp/weights/best.pt]($(cat runs/train/exp/weights/best.pt | jisap-cli publish --url))"
echo
echo "[runs/train/model.version]($(cat runs/train/model.version | jisap-cli publish --url))"
echo
) | tee README.md > report.md
