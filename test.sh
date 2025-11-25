export CUDA_VISIBLE_DEVICES=6
export MUJOCO_GL=osmesa
export PYTHONPATH="/data_all/gzr1/3dcavla:${PYTHONPATH}"
export ROBOSUITE_LOG_DIR="/data_all/gzr1/3dcavla/logs/robosuite"
mkdir -p "$ROBOSUITE_LOG_DIR"
# Run 3D CAVLA evaluations on one of the LIBERO datasets
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /data_all/gzr1/3dcavla/runs/openvla-7b+libero_spatial_cotdep+b8+lr-5e-05+lora-r32+dropout-0.0--image_aug--libero-spatial-cotdep-3dcavla--130000_chkpt \
  --task_suite_name libero_spatial_cotdep \
  --use_depth True \
  --num_trials_per_task 50 

# Run OpenVLA-OFT evaluations on one of the LIBERO datasets
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
#   --task_suite_name libero_spatial_cotdep \
#   --use_depth False \
#   --num_trials_per_task 50 