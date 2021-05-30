# Dialog Generation Model




## Installation
```bash
export INSTALL_DIR=$PWD

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install oscar
cd coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install requirements
pip install -r requirements.txt

unset INSTALL_DIR
```
## Training

Script to train dialog generation model:

```bash
python oscar/run_dialog_generate.py \
    --model_name_or_path unilm_checkpoints \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --add_od_labels \
    --learning_rate 0.00003 \
    --per_gpu_train_batch_size 64 \
    --num_train_epochs 30 \
    --save_steps 5000 \
    --output_dir output/
```

## Inference

Script to inference on Reddit with beam search:

```bash
python oscar/run_dialog_generate.py \
    --do_test \
    --do_eval \
    --test_yaml test.yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --eval_model_dir your_model_for_evaluation 
```


Script to inference on Reddit with top_k/top_p sampling:
```bash
python oscar/run_dialog_generate.py \
    --do_test \
    --do_sample \
    --do_eval \
    --test_yaml test.yaml \
    --per_gpu_eval_batch_size 64 \
    --top_k k_val \
    --top_p p_val \
    --max_gen_length 20 \
    --eval_model_dir your_model_for_evaluation 
```



## Reference

The code is implemented as a fork from [OSCAR](https://github.com/microsoft/OSCAR), more details please refer to their repo, many thanks for their work.

