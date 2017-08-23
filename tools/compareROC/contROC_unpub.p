set terminal png size 1280, 960 enhanced font 'Verdana,18'
set key font ",12"
set size 1,1
set xtics 500
set ytics 0.1
set grid
set ylabel "True positive rate"
set xlabel "False positive"
set xr [0:2000]
set yr [0:1.0]
set key right bottom
set output "contROC_unpub.png"
plot  "rocCurves/BAIDU-IDL-cont.txt" using 2:1 title 'Baidu-IDL' with lines lw 2 , \
 "rocCurves/Linkface-ContROC_v1.txt" using 2:1 title 'Linkface' with lines lw 2 , \
 "rocCurves/Uni-Ubi_ContROC-2.txt" using 2:1 title 'Uni-Ubi-2' with lines lw 2 , \
 "rocCurves/LandmarkbasedContROC.txt" using 2:1 title 'IMC Lab' with lines lw 2 , \
 "rocCurves/Uni-Ubi_ContROC.txt" using 2:1 title 'Uni-Ubi' with lines lw 2 , \
 "rocCurves/Eyedea_ContROC.txt" using 2:1 title 'EyedeaRecognition' with lines lw 2 , \
 "rocCurves/StradVision-ContROC.txt" using 2:1 title 'StradVision' with lines lw 2 , \
 "rocCurves/tencent_ContROC.txt" using 2:1 title 'Tencent-BestImage' with lines lw 2 , \
 "rocCurves/Shenzhen2014-ContROC.txt" using 2:1 title 'Shenzhen2014' with lines lw 2 , \
 "rocCurves/tvplay.cn-ContROC.txt" using 2:1 title 'TVplay' with lines lw 2 , \
 "rocCurves/HisignContROC.txt" using 2:1 title 'Hisign' with lines lw 2 , \
 "rocCurves/Face++ContROC.txt" using 2:1 title 'Face++' with lines lw 2 , \
 "rocCurves/shenzenContROC.txt" using 2:1 title 'Shenzhen Uni' with lines lw 2 , \
 "rocCurves/OlaworksContROC.txt" using 2:1 title 'Olaworks' with lines lw 2 , \
 "rocCurves/illuxtechFrontalFaceContROC.txt" using 2:1 title 'Illuxtech Inc.' with lines lw 2
