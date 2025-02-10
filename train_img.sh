echo 'Hellooo'
sleep 1h
echo 'Rerun'

python rerunTraining.py \
    --resume_from outputs/030-MMDiT-B-2/checkpoint_0720000.pt \
    --model MMDiT-B/2 \
    --image_size 256 \
    --batch_size 32 \
    --epochs 1600 \
    --ckpt_every 100000 \