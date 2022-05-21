rm g1_add_single
command='hipcc g1_add.cu -o g1_add_single -std=c++14 -O3 -I/opt/include/'
echo $command
eval "$command"
./g1_add_single