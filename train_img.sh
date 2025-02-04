python rerunTraining.py \
    --resume_from outputs/027-MMDiT-S-8/checkpoint_0300000.pt \
    --model MMDiT-B/2 \
    --image_size 256 \
    --batch_size 32 \
    --epochs 1400 \
    --ckpt_every 10000 \
