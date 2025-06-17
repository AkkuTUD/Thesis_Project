#!/bin/bash
set -e # exit on error

#########################
## Convert tiff to png ##
#########################
tmpdir="training_data/all"
rm -rf $tmpdir
mkdir -p $tmpdir
cd tiles_data/images

parallel convert "{}" -combine -type truecolor "../../$tmpdir/{.}.png" ::: *.tiff
#for f in *.tiff
#do
#    tgt="$(basename "$f" .tiff).png"
#    convert "$f" -combine -type truecolor "../../training_data/all/$tgt"
#done
cd ../..

####################################
## Split into training/validation ##
####################################

mkdir -p training_data/images/validation training_data/images/training

val_percent=$(./p split.val_percent)
# RANDOM=$(./p split.random_seed)

# ls "$tmpdir" | while read file
# do
#     if (( RANDOM % 100 < $val_percent ))
#     then
#         mv "$tmpdir/$file" training_data/images/validation
#     else
#         mv "$tmpdir/$file" training_data/images/training
#     fi
# done

image_files=$(ls "$tmpdir")
animal_ids=$(echo "$image_files" | awk -F'-' '{print $2}' | tr ' ' '\n' | sort -u | tr '\n' ' ')
ids_list=$(echo "$animal_ids" | tr '\n' ' ')
shuffled_ids=$(echo "$ids_list" | tr ' ' '\n' | shuf | tr '\n' ' ')
total_values=$(echo "$ids_list"| wc -w)

#split the train validate set in 80:20 ratio
train_percent=$((total_values * val_percent / 100))
echo "total_ids=$total_values , train_percent=$train_percent"
train_animal_ids=$(echo "$ids_list" | awk -v n="$train_percent" '{for (i=1; i<=n; i++) print $i}')
echo "selected train ids $train_animal_ids"
validate_animal_ids=$(echo "$ids_list" | awk -v n="$train_percent" -v total="$total_values" '{for (i=n+1; i<=total; i++) print $i}')
echo "selected validate ids $validate_animal_ids"

for animal_id in $train_animal_ids; do
  cp $tmpdir/*$animal_id* training_data/images/training
done

for animal_id in $validate_animal_ids; do
  cp $tmpdir/*$animal_id* training_data/images/validation
done

rm -r $tmpdir