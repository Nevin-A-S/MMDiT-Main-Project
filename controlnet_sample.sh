python SampleMMditControlnet.py \
    --model MMDiT-B/2 \
    --pretrained_path outputs/009-MMDiT-B-2/checkpoint_1100000.pt \
    --image_size 256 \
    --batch_size 16 \
    --ckpt_every 10000 \
    --output_dir outputs/sample_control_net \
    --ckpt \
    --cfg 0 