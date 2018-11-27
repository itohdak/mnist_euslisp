file = "/tmp/mnist-loss-accuracy.dat"
plot file u 1:2 t "train loss" w lp lw 2 ps 1
rep file u 1:4 t "test loss" w lp lw 2 ps 1
rep file u 1:3 t "train accuracy" w lp lw 2 ps 1 axes x1y2
rep file u 1:5 t "test accuracy" w lp lw 2 ps 1 axes x1y2
set grid
set xl "epoch"
set yl "loss"
set yrange [0.0:0.5]
set y2tics
set y2l "accuracy"
set y2range [0.8:1.05]
rep
pause -1 "hit Enter key"
