
# DS="EN"
# BK="GATv2"
# grep "MRR" logs/final/${DS}/original-*/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-bx
# grep "MRR" logs/final/${DS}/ours-loss/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-bx
# grep "MRR" logs/final/${DS}/inc-bx-a-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-bx

BK="SAGE"
LS="xm"
grep "AUC" logs/final/EN/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
grep "AUPR" logs/final/EN/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
grep "AUC" logs/final/IC/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
grep "AUPR" logs/final/IC/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
grep "AUC" logs/final/GPCR/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
grep "AUPR" logs/final/GPCR/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
grep "AUC" logs/final/NR/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
grep "AUPR" logs/final/NR/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
# grep "AUC" logs/final/Shao/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}
# grep "AUPR" logs/final/Shao/inc-bx-a-10-all/**.log | awk '{gsub(".*/|.log:.+","",$1); arr[$1]+=$NF; count[$1]++} END{for (a in arr) print a, arr[a]/count[a]}' | sort -t '-' -k 1,1 -k 2,2n | grep ${BK}-${LS}