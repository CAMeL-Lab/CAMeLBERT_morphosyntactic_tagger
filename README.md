# CAMeLBERT_morphosyntactic_tagger
Codebase for "[Morphosyntactic Tagging with Pre-trained Language Models for Arabic and its Dialects](https://aclanthology.org/2022.findings-acl.135/)". Findings of ACL, 2022.

Some of the models are already part of the newer version of [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools). Please check out the repository if you want to try out our tagger!

## Data
- We release our preprocessed data through [Muddler](https://github.com/CAMeL-Lab/muddler), a tool for sharing derived data. To install the tool, run `pip install muddler`. 
- The encyrpted pre-processed files are available in our GitHub release.
- To unpack these files, you will first need to obtain the following resources.
  - Morphological Analyzer:
    - [LDC2010L01](https://catalog.ldc.upenn.edu/LDC2010L01): `LDC2010L01.tgz`
  - MSA 
    - [LDC2010T13](https://catalog.ldc.upenn.edu/LDC2010T13): `atb1_v4_1_LDC2010T13.tgz`
    - [LDC2011T09](https://catalog.ldc.upenn.edu/LDC2011T09): `atb_2_3.1_LDC2011T09.tgz`
    - [LDC2010T08](https://catalog.ldc.upenn.edu/LDC2010T08): `atb3_v3_2_LDC2010T08.tgz`
  - EGY
    - [LDC2018T23](https://catalog.ldc.upenn.edu/LDC2018T23): `bolt_arz-df_LDC2018T23.tgz`
  - GLF
    - [The Annotated Gumar Corpus](https://camel.abudhabi.nyu.edu/annotated-gumar-corpus/): `annotated-gumar-corpus.zip`
  - LEV
    - [Curras](https://portal.sina.birzeit.edu/curras/download.html): `CurrasAnnotations_Full.csv`
- Once you obtain these resources along with the encrypted files, put all the downloaded files in some directory. (In this example, we name the directory `corpora/`)
  ```bash
  # unmuddle the files
  muddler unmuddle -s corpora -m corpora/MSA.tar.gz.muddle MSA.tar.gz
  muddler unmuddle -s corpora -m corpora/EGY.tar.gz.muddle EGY.tar.gz
  muddler unmuddle -s corpora -m corpora/GLF.tar.gz.muddle GLF.tar.gz
  muddler unmuddle -s corpora -m corpora/LEV.tar.gz.muddle LEV.tar.gz
  ```

- If you want to use other datasets, the input format should look like this:
  - `word label` per line, where the delimiter is a space character.
  - Empty line at the end of a sentence.
  - Example:
      ```
      هند noun_prop
      : punc
      هي pron
      صح adj
      بس conj
      ابي verb
      اعرف verb
      اذا conj_sub
      هو pron
      بيخطبني verb
      ؟ punc

      منى noun_prop
      : punc
      اكيد adj

      ```

## Fine-tuning Experiment
### Pre-trained Models
Pre-trained language models are avaialble on Hugging Face model hub. To download our models as described in the model hub, you would need `transformers>=3.5.0`. Otherwise, you could download the models manually.
  - CAMeLBERT-MSA: [`CAMeL-Lab/bert-base-arabic-camelbert-msa`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa)
  - CAMeLBERT-Mix: [`CAMeL-Lab/bert-base-arabic-camelbert-mix`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix)

### Requirements
```bash
git clone https://github.com/CAMeL-Lab/CAMeLBERT_morphosyntactic_tagger.git
cd CAMeLBERT_morphosyntactic_tagger

conda create -n CAMeLBERT_morphosyntactic_tagger python=3.7
conda activate CAMeLBERT_morphosyntactic_tagger

pip install -U git+https://github.com/go-inoue/camel_tools.git@bert-disambig
pip install -r requirements.txt
```

### Training a morphosyntactic tagger
```bash
# set variables
export SEED=12345
export BERT_MODEL=/path/to/pretrained_model
export DATA_DIR=/path/to/data
export OUTPUT_DIR=./train_MSA_full_unfactored
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=500
export MAX_LENGTH=512

# run the training script
python scripts/run_token_classification.py \
  --data_dir $DATA_DIR \
  --task_type pos \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed $SEED \
  --overwrite_output_dir \
  --overwrite_cache \
  --do_train \
  --do_eval

# move files
for CP in checkpoint-*/; do
  cp tokenizer_config.json $CP/
  cp special_tokens_map.json $CP/
  cp vocab.txt $CP/
done
mkdir $OUTPUT_DIR/checkpoint-last
mv $OUTPUT_DIR/*.* $OUTPUT_DIR/checkpoint-last/
```

### Picking the best checkpoint
```bash
# set variables
export SEED=12345
export DATA_DIR=/path/to/data
export OUTPUT_DIR=./train_MSA_full_unfactored
export BATCH_SIZE=32
export NUM_EPOCHS=10
export MAX_LENGTH=512

# eval on all the checkpoints
for CHECKPOINT in $OUTPUT_DIR/checkpoint-*/; do
python scripts/run_token_classification.py \
  --data_dir $DATA_DIR \
  --task_type pos \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $CHECKPOINT \
  --output_dir $CHECKPOINT \
  --max_seq_length  $MAX_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --seed $SEED \
  --overwrite_cache \
  --do_eval
done

# copy the output of the command, and run it
python scripts/pick_best_checkpoint.py $OUTPUT_DIR
```

### Inference
```bash
# set variables
export SEED=12345
export DATA_DIR=/path/to/data
export MODEL_DIR=./train_MSA_full_unfactored/checkpoint-best
export BATCH_SIZE=32
export NUM_EPOCHS=10
export MAX_LENGTH=512

# predict on dev test
python scripts/run_token_classification.py \
  --data_dir $DATA_DIR \
  --task_type pos \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL_DIR \
  --output_dir $MODEL_DIR \
  --max_seq_length  $MAX_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --seed $SEED \
  --overwrite_cache \
  --do_pred

# predict on blind test
python scripts/run_token_classification.py \
  --data_dir $DATA_DIR \
  --task_type pos \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL_DIR \
  --output_dir $MODEL_DIR \
  --max_seq_length  $MAX_LENGTH \
  --per_device_train_batch_size $BATCH_SIZE \
  --seed $SEED \
  --overwrite_cache \
  --do_pred \
  --blind_test
```

## Citation

```bibtex
@inproceedings{inoue-etal-2022-morphosyntactic,
    title = "Morphosyntactic Tagging with Pre-trained Language Models for Arabic and its Dialects",
    author = "Inoue, Go  and
      Khalifa, Salam  and
      Habash, Nizar",
    booktitle = "Proceedings of the Findings of the Association for Computational Linguistics: ACL2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    abstract = "We present state-of-the-art results on morphosyntactic tagging across different varieties of Arabic using fine-tuned pre-trained transformer language models. Our models consistently outperform existing systems in Modern Standard Arabic and all the Arabic dialects we study, achieving 2.6% absolute improvement over the previous state-of-the-art in Modern Standard Arabic, 2.8% in Gulf, 1.6% in Egyptian, and 8.3% in Levantine. We explore different training setups for fine-tuning pre-trained transformer language models, including training data size, the use of external linguistic resources, and the use of annotated data from other dialects in a low-resource scenario. Our results show that strategic fine-tuning using datasets from other high-resource dialects is beneficial for a low-resource dialect Additionally, we show that high-quality morphological analyzers as external linguistic resources are beneficial especially in low-resource settings."
}
```