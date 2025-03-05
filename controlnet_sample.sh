python SampleMMditControlnet.py \
    --model MMDiT-B/2 \
    --pretrained_path outputs/checkpoint_1100000.pt \
    --image_size 256 \
    --batch_size 64 \
    --output_dir outputs/sample_control_net \
    --ckpt outputs/checkpoint_control_net_0160000.pt \
    --cfg 0