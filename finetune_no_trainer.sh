accelerate launch finetune_no_trainer.py \
 --model_path "bert-base-uncased" \
 --output_dire "output/baseline_lr5e-5" \
 --train_file "data/preprocessed/train.csv" \
 --validation_file "data/preprocessed/dev.csv" \
 --test_file "data/preprocessed/test.csv" \
 --per_device_batch_size 8 \
 --num_train_epochs 10 \
 --learning_rate 5e-5 \
 --num_warmup_steps 98 \
 --seed 42 \
 --experiment_name sentiment_analysis \
 --scenario_name "batch_size 8, lr 5e-5, baseline"