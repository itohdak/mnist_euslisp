# mnist_euslisp
learning mnist with euslisp

## Download mnist dataset
- http://yann.lecun.com/exdb/mnist/

## Decompress mnist dataset
```
$ gzip -dc train-images-idx3-ubyte.gz >train-images-idx3-ubyte
$ gzip -dc train-labels-idx1-ubyte.gz >train-labels-idx1-ubyte
$ gzip -dc t10k-images-idx3-ubyte.gz >test-images-idx3-ubyte
$ gzip -dc t10k-labels-idx1-ubyte.gz >test-labels-idx1-ubyte
```

## Convert dataset to text file
```
$ od -An -v -tu1 -j16 -w784 train-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' >train-images.txt
$ od -An -v -tu1 -j8 -w1 train-labels-idx1-ubyte | tr -d ' ' >train-labels.txt
$ od -An -v -tu1 -j16 -w784 test-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' >test-images.txt
$ od -An -v -tu1 -j8 -w1 test-labels-idx1-ubyte | tr -d ' ' >test-labels.txt

$ rm *-ubyte
$ rm *.gz
```

## Compile euslisp file
```
$ roseus
irteusgl$ compile-file "mnist.l"
irteusgl$ exit

$ roseus
irteusgl$ load "mnist.so"
```

