export PATH="$PATH:/kaggle/working/sample"
export PYTHONPATH="$PYTHONPATH:/kaggle/working/sample"

export MUSTC_ROOT="speech_data/mustc"
export WMT_ROOT="wmt_data"
export SAVE_ROOT="checkpoints"
export target=de

export ST_SAVE_DIR="$SAVE_ROOT/st"
export MT_SAVE_DIR="$SAVE_ROOT/mt"
export SAVE_DIR=$MT_SAVE_DIR
export WAVE2VEC_DIR="$SAVE_ROOT/wave2vec"
pretrained_ckpt=wav2vec_small.pt
mkdir -p $ST_SAVE_DIR $MT_SAVE_DIR $WAVE2VEC_DIR $MUSTC_ROOT $WMT_ROOT

# downloading wav2vec2 ckpt
bash chimera/tools/download_wav2vec2.sh $pretrained_ckpt $WAVE2VEC_DIR

# WMT-MUSTC joint data and spm
TEXT=$WMT_ROOT/${dataset}_en_$target
spm_model=$TEXT/spm/spm_unigram10000_wave_joint.model
spm_dict=$TEXT/spm/spm_unigram10000_wave_joint.txt

python3 /kaggle/working/sample/fairseq_cli/preprocess.py \
    --source-lang en --target-lang $target \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --testpref $TEXT/test,$TEXT/mustc-tst-COMMON \
    --destdir $WMT_ROOT/bin --thresholdtgt 0 --thresholdsrc 0 \
    --srcdict $spm_dict --tgtdict $spm_dict \
    --workers 100
