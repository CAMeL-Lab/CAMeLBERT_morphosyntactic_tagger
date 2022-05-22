## Fine-tuning Experiment
### Pre-trained Models
Pre-trained language models are avaialble on Hugging Face model hub. To download our models as described in the model hub, you would need `transformers>=3.5.0`. Otherwise, you could download the models manually.
  - CAMeLBERT-MSA: [`CAMeL-Lab/bert-base-arabic-camelbert-msa`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa)
  - CAMeLBERT-Mix: [`CAMeL-Lab/bert-base-arabic-camelbert-mix`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix)


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