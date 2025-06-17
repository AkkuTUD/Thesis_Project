#!/bin/bash
set -e
cf="yolo_results_columns.txt"
inp="runs/train/exp_results.csv"
outp="runs/train/exp_results_last.yaml"

cat $cf | awk '{print "  " $0}' > tmp_ids.txt
cat $inp | tail -n 1 | tr ',' '\n' | sed '/^$/d' | cut -d'/' -f1 > tmp_values.txt

echo "train:" > $outp
paste -d ":" tmp_ids.txt tmp_values.txt | sed 's/:/: /g' | grep -v gpu_mem >> $outp

rm tmp_ids.txt tmp_values.txt
