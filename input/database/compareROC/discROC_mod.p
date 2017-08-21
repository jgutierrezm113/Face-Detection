# Compare your continuous ROC curves with other methods
# At terminal: gnuplot contROC.p
set terminal png size 1280, 960 enhanced font 'Verdana,18'
set size 1,1
set xtics 500
set ytics 0.1
set grid
set ylabel "True positive rate"
set xlabel "False positive"
set xr [0:2000]
set yr [0:1.0]
set key below
set output "discROC.png"
plot  "MTCNN-DiscROC.txt" using 2:1 title 'MTCNN' with linespoints pointinterval 50 lw 3 , \
"DiscROC.txt" using 2:1 title 'JGM' with linespoints pointinterval 50 lw 3 

 
