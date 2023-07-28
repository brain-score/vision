SUBMISSION_IDS=(6710 6680 2425 6757 6660 6698 6758 6706 6768 6727 6708 2430 6670 6763 6687 6674 6724 6762 6826 6764 3303 6693 6691 6869 6766 3295 6797 2446 6759 6684 6723 3300 6778 3349 6728 6811 2436 6828 6731 6682 6805 6767 6668 6729 6795 6737 6798 6707 6711 6715 6697 6827 6721 6760 6688 1888 6692 6681 3331 3297 6815 6796 6777 6725 6652 6806 3360 6694 6690 6712 6821 6695 6765 6657 6686 6822 1568 2437 3294 6761 3298 6699 6810 6689 6730 6678)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MODELS_DIR=$(realpath "$SCRIPT_DIR"/../brainscore_vision/models/)
TEMPORARY_UNZIP_DIRECTORY=$(realpath "$MODELS_DIR"/../models_unzip_temp)
SUBMISSIONS_DIRECTORY=/braintree/data2/active/common/brainscore_submissions

submission_names=()
for submission_id in "${SUBMISSION_IDS[@]}"; do
  # unzip
  submission_zip="$SUBMISSIONS_DIRECTORY"/submission_"$submission_id".zip
  rm -rf --- "${TEMPORARY_UNZIP_DIRECTORY:?}"/*                      # clear temporary directory
  unzip "$submission_zip" -d "$TEMPORARY_UNZIP_DIRECTORY" >/dev/null # unzip, do not print stdout
  chmod -R ugo+w "$TEMPORARY_UNZIP_DIRECTORY"
  rm -rf "${TEMPORARY_UNZIP_DIRECTORY:?}"/__MACOSX # clear Apple leftovers

  # find name of zip file contents
  zip_contents=$(find "$TEMPORARY_UNZIP_DIRECTORY"/* -maxdepth 0 -type d)
  submission_name=$(basename "$zip_contents")
  echo "Submission $submission_id: $submission_name"
  submission_names+=("$submission_name")
done

echo "Submissions: ${submission_names[*]}"
# -> effnet_b0_tiny_cmc2_0_pretrain effnet_b0_lpplinearinput rs_robustness_submission alexnet_tiny_base64 effnet_b0_lpplinear vgg16_lpppre_pretrain alexnet_tiny_base64 effnet_b0_tiny_cmc2_0_pretrain effnet_b0_lppprev3_pretrain_l2 effnet_b0_imagenet effnet_b0_tiny_cmc2_all_pretrain madry_robustness_submission_l2_3 effnet_b0_lpplinearconv alexnet_tiny_base512_grayscale effnet_b0_lpppre_pretrain effnet_b0_lpplinearconv effnet_b0_lpppre_pretrain_noise effnet_b0_lppcircle resnet50_tiny_lppprev3_dog alexnet_tiny_base256_dropout vgg11_baselines_submission_cifar effnet_b0_tiny_cmc0_pretrain effnet_b0_tiny_cmc0_pretrain_grayscale imagenet64_resnet50_lppprev3_local_dog_onoff effnet_b0_lppprev3_pretrain vgg11_baselines_submission w23_cv_rw_reccurrent rs_robustness_submission_sigma0p5 alexnet_tiny_base256 effnet_b0_lpppre effnet_b0_wd0_pretrain_ vgg11_baselines_submission copy effnet_b0_colorblock hipernet_brain_score_sub effnet_b0_random_initialization w23_coarse madry_robustness_submission_linf_4 imagenet_resnet50_lppprev3_local effnet_b0_wd_pretrain_no_interp effnet_b0_lpplinearconv w23_cv_polar effnet_b0_lppprev31_pretrain effnet_b0_lpplinear effnet_b0_imagenet imagenet_resnet50_lppprev3 effnet_b0_rw_pretrain_2 w23_cv_rw effnet_b0_tiny_cmc2_v1_pretrain effnet_b0_tiny_cmc2_v1_pretrain effnet_b0_lpplinearconv_pretrain vgg16_lpppre_pretrain resnet50_tiny_lppprev3_dog_onoff effnet_b0_wd0_pretrain alexnet_tiny_base512 effnet_b0_tiny_cmc0 my_alexnet_subission effnet_b0_tiny_cmc0_pretrain_v2 effnet_b0_lpplinearinput_ml2 vgg11_baselines_submission_imagenet vgg11_baselines_submission resnet50_tiny_lppprev3_local imagenet_effnetb0_lppprev3 alexnet_tiny_64_rf25 effnet_b0_lpppre_pretrain_noise_alpha02 my_alexnet_submission w23_baseline hipernet_brain_score_sub_1 effnet_b0_tiny_cmc0_pretrain_v3 effnet_b0_lpppre_pretrain_grayscale effnet_b0_tiny_cmc2_all_pretrain resnet50_tiny_lppprev3_deform effnet_b0_cmc_block1_pretrain effnet_b0_lppsquare effnet_b0_tiny resnet50_lpppre resnet50_tiny_pretrain my_FrankRobWob_submission madry_robustness_submission_linf_8 alexnet_test_submission alexnet_tiny_base512_0 vgg11_baselines_submission copy effnet_b0_tiny_pretrain w23_cortical effnet_b0_tiny_cmc0_pretrain effnet_b0_rw_pretrain effnet_b0_lpplinearinput
