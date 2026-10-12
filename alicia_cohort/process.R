rm(list=ls())
setwd('/Volumes/hua_mac/research/aris/deeplearning/tangerine/tangerine_data/meta')

#d1=read.csv('/Volumes/hua_mac/research/aris/deeplearning/tangerine/tangerine_data/meta/LCSP Completed Scans(CY2016-2026TD) LHS data share.csv',header=T,stringsAsFactors = F,check.names=F)
d1=read.csv('/Volumes/For_Hua/20260709_FullList_from_AlisaB/LCSP Completed Scans(CY2016-2026TD) LHS data share.csv',header=T,stringsAsFactors = F,check.names=F)

d2=read.csv('/Volumes/hua_mac/research/aris/deeplearning/tangerine/tangerine_data/meta/sybil_data_17200_with_mrn_anonacc.csv',header=T,stringsAsFactors = F,check.names=F)
d3=read.csv('/Volumes/hua_mac/research/aris/deeplearning/tangerine/tangerine_data/meta/lungct_with_mrn_anonacc.csv',header=T,stringsAsFactors = F,check.names=F)


d1$have_scan=FALSE
d1$have_scan[d1$AccessionNumber %in% d2$acc_num]=TRUE
write.csv(d1,'/Volumes/For_Hua/20260709_FullList_from_AlisaB/processed2/fullset.csv',row.names=F)
write.csv(d1[!d1$have_scan,-29],'/Volumes/For_Hua/20260709_FullList_from_AlisaB/processed2/missing_set.csv',row.names=F)

length(intersect(d1$AccessionNumber,d3$acc_num))
length(intersect(d1$AccessionNumber,d2$acc_num))
