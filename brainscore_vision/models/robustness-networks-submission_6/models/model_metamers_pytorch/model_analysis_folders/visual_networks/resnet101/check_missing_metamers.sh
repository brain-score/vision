#!/bin/bash

# export CHECK_FOLDER='adversarial_examples/manifolds_dataset_jsinv3_word_audio_subsetREPTYPEcoch_RS0_l2_E1.000_I32_S0.100/'
# export CHECK_FOLDER='adversarial_examples/manifolds_dataset_jsinv3_word_audio_subsetREPTYPEcoch_RS0_l2_E0.300_I32_S0.100/'
export CHECK_FOLDER='metamers/400_16_class_imagenet_val_inversion_loss_layer_RS0_I2000_N8/'

# Redirect output, per answer
# exec > file.txt
my_array=()
for ((i=0 ; i<=400; i++)) ; do
   # Convert to 4 digit zero padded
   printf -v id '%d' $i
   if [ ! -d "${CHECK_FOLDER}${id}_SOUND"* ] ; then
       my_array+=( $id )
   fi
   ls "${CHECK_FOLDER}${id}_SOUND"* | wc
#    echo {$ls ${CHECK_FOLDER}${id}_SOUND"*"/EXAMPLE_"*}
done
printf -v joined '%s,' "${my_array[@]}"
echo "${joined%,}"
