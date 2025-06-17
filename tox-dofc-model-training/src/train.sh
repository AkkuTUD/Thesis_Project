#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
rm -rf src/runs
python3 src/launch.py yolov5/train.py --img "$(./p train.img)" --batch "$(./p train.batch)" --epochs "$(./p train.epochs)" --workers "$(./p train.workers)" --data "$(./p train.data)" --weights "$(./p train.weights)" --hyp "$(./p train.hyp)" 
rm -rf ./runs
mv src/runs ./runs
cp runs/train/exp/results.csv runs/train/exp_results.csv
src/res2yaml.sh


version="$CI_PIPELINE_IID.$CI_BUILD_ID-$CI_COMMIT_SHORT_SHA"
pipelineUrl="$CI_PROJECT_URL/-/pipelines/$CI_PIPELINE_ID"
echo "1.$version;$pipelineUrl" > runs/train/model.version
