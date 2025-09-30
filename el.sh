cleanup() {
    echo "Terminating all processes..."
    pkill -f gpt_eval.py   
    echo "All processes terminated."
}

# 捕获 EXIT 和 SIGINT 信号
trap cleanup EXIT SIGINT
#sleep 5h 
for i in {0,1,2,3,4,5,6,7}
do
    CUDA_VISIBLE_DEVICES=$((i)) python gpt_eval.py --model_id $((i))  &
done

wait