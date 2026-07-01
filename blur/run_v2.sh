#!/bin/bash
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

export CUDA_VISIBLE_DEVICES=3

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate CUDA122_Grad-TTS  

HYDRA_FULL_ERROR=1 \
bash -c 'exec -a blur python train.py -m --config-name=config_swp_frozen +data=data_swp model.masking.b=10,40'


HYDRA_FULL_ERROR=1 \
bash -c 'exec -a blur python train.py -m --config-name=config_swp_free +data=data_swp model.masking.b=10,40'

conda deactivate
conda activate CUDA122_Grad-TTS_eval
gt=/exp/exp4/acp23xt/DTDM_Unet/blur/blur_inf/model.masking.b:10/test/
gen=/exp/exp4/acp23xt/DTDM_Unet/blur/blur_inf/model.masking.b:10/test/converted/Epoch_590
out=/exp/exp4/acp23xt/DTDM_Unet/blur/blur_inf/model.masking.b:10/test/metrics

python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python "tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"

gt=/exp/exp4/acp23xt/DTDM_Unet/blur/blur_inf/model.masking.b:40/test/
gen=/exp/exp4/acp23xt/DTDM_Unet/blur/blur_inf/model.masking.b:40/test/converted/Epoch_590
out=/exp/exp4/acp23xt/DTDM_Unet/blur/blur_inf/model.masking.b:40/test/metrics

python "tools/F0/F0.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"
python "tools/MCD/MCD.py" --gt_wavdir_or_wavscp "$gt" --gen_wavdir_or_wavscp  "$gen" --outdir "$out"


