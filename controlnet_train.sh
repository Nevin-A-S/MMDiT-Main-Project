python ControlNetTrain.py \
    --model MMDiT-B/2 \
    --pretrained_path outputs/009-MMDiT-B-2/checkpoint_1100000.pt\
    --resume_from outputs/028-MMDiT-B-2/checkpoint_0180000.pt \
    --image_size 256 \
    --batch_size 16 \
    --epochs 1600 \
    --ckpt_every 10000 \
