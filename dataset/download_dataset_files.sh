#!/bin/bash
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

unzip *.gz
mv t10k-labels-idx1-ubyte t10k-labels-idx1-ubyte.bin
mv t10k-images-idx3-ubyte t10k-images-idx3-ubyte.bin
mv train-labels-idx1-ubyte train-labels-idx1-ubyte.bin
mv train-images-idx3-ubyte train-images-idx3-ubyte.bin
rm -f *.gz

