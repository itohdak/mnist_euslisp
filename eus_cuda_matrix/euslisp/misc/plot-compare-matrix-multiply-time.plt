file = "/tmp/compare-matrix-multiply-time.dat"
plot file u 1:2 t "cuda" w lp lw 2 ps 2
rep file u 1:3 t "cuda-cublas" w lp lw 2 ps 2
rep file u 1:4 t "cblas" w lp lw 2 ps 2
rep file u 1:5 t "openblas" w lp lw 2 ps 2
set grid
set xl "row-a"
set yl "time [sec]"
rep
pause -1 "hit Enter key"
