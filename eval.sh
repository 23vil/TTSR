### evaluation
python main.py --save_dir ./eval/IMM/TTSR \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 4 \
               --dataset IMM \
               --dataset_dir ./dataset/IMM/ \
               --model_path ./pretrainedModels/model_00050.pt