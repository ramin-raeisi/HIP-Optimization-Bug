rm g1_add_multi
command='hipcc g1_add.cu -o g1_add_multi -std=c++14 -O3 -I/opt/include/ -DMULTI_ASM'
echo $command
eval "$command"
./g1_add_multi