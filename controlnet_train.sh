python ControlNetTrain.py \
    --model MMDiT-B/2 \
    --pretrained_path /media/sct/projects/MMDit/Project/MMDiT-Main-Project/outputs/009-MMDiT-B-2/checkpoint_0390000.pt \
    --image_size 256 \
    --batch_size 32 \
    --epochs 1600 \
    --ckpt_every 10000 \
