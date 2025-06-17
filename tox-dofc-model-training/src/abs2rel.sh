#!/bin/bash
set -e # halt on error
rm -rf training_data/labels
mkdir -p training_data/labels/training training_data/labels/validation
cd tiles_data/labels
for f in *.txt
do
    export f
    awk '{ print $1 " " ($2+$4)/2/1280 " " ($3+$5)/2/1280 " " ($4-$2)/1280 " " ($5-$3)/1280 > "../../training_data/labels/" ENVIRON["f"]}' "$f"
    # cp "../../training_data/labels/$f" "../../training_data/labels/training/$f"
    # cp "../../training_data/labels/$f" "../../training_data/labels/validation/$f"
    # rm "../../training_data/labels/$f"
done

cd ../..

dir_train="training_data/images/training"
dir_validate="training_data/images/validation"

tmpdir_train_files=$(ls "$dir_train")
animal_ids_train=$(echo "$tmpdir_train_files" | awk -F'-' '{print $2}' | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo $animal_ids_train

tmpdir_val_files=$(ls "$dir_validate")
animal_ids_val=$(echo "$tmpdir_val_files" | awk -F'-' '{print $2}' | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo $animal_ids_val

for animal_id in $animal_ids_train; do
    cp training_data/labels/*$animal_id* training_data/labels/training
done

for animal_id in $animal_ids_val; do
    cp training_data/labels/*$animal_id* training_data/labels/validation
done

rm training_data/labels/*.txt
cd ../..
echo "Done.."