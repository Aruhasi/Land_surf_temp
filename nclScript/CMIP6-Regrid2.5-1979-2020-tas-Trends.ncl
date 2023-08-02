load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "/disk_raid6/pub/aru/Proposal-reproduce/Figure-Global-SAT-trend-inModel/unweighted-seasonal-mean.ncl"

begin

    iyear = ispan(1979, 2020, 1)
    latS   =  20.
    latN   =  70. 
    lonL   = 120.
    lonR   = 180.

    yrStrt = 197801
    yrLast = 202012 ; 2015-2020 is the SSP585 simulation

   cold_season = "NDJFMA"    ; choose Dec-Jan-Feb seasonal mean
   warm_season = "MJJASO"

Name = (/"ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-CSM2-MR", "CAMS-CSM1-0", "CanESM5-CanOE", "CanESM5",\
        "CESM2-WACCM", "CIESM", "CMCC-CM2-SR5", "CNRM-CM6-1-HR", "CNRM-CM6-1","CNRM-ESM2-1", "E3SM-1-1", "EC-Earth3",\
        "EC-Earth3-Veg", "EC-Earth3-Veg-LR","FGOALS-f3-L", "FGOALS-g3", "FIO-ESM-2-0", "GFDL-CM4","GFDL-ESM4", "GISS-E2-1-G",\
        "HadGEM3-GC31-LL", "HadGEM3-GC31-MM","IITM-ESM", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KIOST-ESM",\
        "MCM-UA-1-0", "MIROC-ES2L", "MIROC6","MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NESM3", "NorESM2-LM", \
        "NorESM2-MM", "TaiESM1", "UKESM1-0-LL"/)

diri = "/disk_raid6/pub/aru/Projection-WP-Variability/CMIP6/1850-2100/tas/"
;===================================Surface air temperature============================
fil1        = "tas_Amon_"+Name(0)+"_hist_ssp585_1850-2100.nc"
fil2        = "tas_Amon_"+Name(1)+"_hist_ssp585_1850-2100.nc"
fil3        = "tas_Amon_"+Name(2)+"_hist_ssp585_1850-2100.nc"
fil4        = "tas_Amon_"+Name(3)+"_hist_ssp585_1850-2100.nc"
fil5        = "tas_Amon_"+Name(4)+"_hist_ssp585_1850-2099.nc"
fil6        = "tas_Amon_"+Name(5)+"_hist_ssp585_1850-2100.nc"
fil7        = "tas_Amon_"+Name(6)+"_hist_ssp585_1850-2100.nc"
fil8        = "tas_Amon_"+Name(7)+"_hist_ssp585_1850-2100.nc"
fil9        = "tas_Amon_"+Name(8)+"_hist_ssp585_1850-2100.nc"
fil10       = "tas_Amon_"+Name(9)+"_hist_ssp585_1850-2100.nc"
fil11       = "tas_Amon_"+Name(10)+"_hist_ssp585_1850-2100.nc"
fil12       = "tas_Amon_"+Name(11)+"_hist_ssp585_1850-2100.nc"
fil13       = "tas_Amon_"+Name(12)+"_hist_ssp585_1850-2100.nc"
fil14       = "tas_Amon_"+Name(13)+"_hist_ssp585_1850-2100.nc"
fil15       = "tas_Amon_"+Name(14)+"_hist_ssp585_1850-2100.nc"
fil16       = "tas_Amon_"+Name(15)+"_hist_ssp585_1850-2100.nc"
fil17       = "tas_Amon_"+Name(16)+"_hist_ssp585_1850-2100.nc"
fil18       = "tas_Amon_"+Name(17)+"_hist_ssp585_1850-2100.nc"
fil19       = "tas_Amon_"+Name(18)+"_hist_ssp585_1850-2100.nc"
fil20       = "tas_Amon_"+Name(19)+"_hist_ssp585_1850-2100.nc"
fil21       = "tas_Amon_"+Name(20)+"_hist_ssp585_1850-2100.nc"
fil22       = "tas_Amon_"+Name(21)+"_hist_ssp585_1850-2100.nc"
fil23       = "tas_Amon_"+Name(22)+"_hist_ssp585_1850-2100.nc"
fil24       = "tas_Amon_"+Name(23)+"_hist_ssp585_1850-2100.nc"
fil25       = "tas_Amon_"+Name(24)+"_hist_ssp585_1850-2100.nc"
fil26       = "tas_Amon_"+Name(25)+"_hist_ssp585_1850-2100.nc"
fil27       = "tas_Amon_"+Name(26)+"_hist_ssp585_1850-2100.nc"
fil28       = "tas_Amon_"+Name(27)+"_hist_ssp585_1850-2100.nc"
fil29       = "tas_Amon_"+Name(28)+"_hist_ssp585_1850-2100.nc"
fil30       = "tas_Amon_"+Name(29)+"_hist_ssp585_1850-2100.nc"
fil31       = "tas_Amon_"+Name(30)+"_hist_ssp585_1850-2100.nc"
fil32       = "tas_Amon_"+Name(31)+"_hist_ssp585_1850-2100.nc"
fil33       = "tas_Amon_"+Name(32)+"_hist_ssp585_1850-2100.nc"
fil34       = "tas_Amon_"+Name(33)+"_hist_ssp585_1850-2100.nc"
fil35       = "tas_Amon_"+Name(34)+"_hist_ssp585_1850-2100.nc"
fil36       = "tas_Amon_"+Name(35)+"_hist_ssp585_1850-2100.nc"
fil37       = "tas_Amon_"+Name(36)+"_hist_ssp585_1850-2100.nc"
fil38       = "tas_Amon_"+Name(37)+"_hist_ssp585_1850-2100.nc"
fil39       = "tas_Amon_"+Name(38)+"_hist_ssp585_1850-2100.nc"
fil40       = "tas_Amon_"+Name(39)+"_hist_ssp585_1850-2100.nc"
fil41       = "tas_Amon_"+Name(40)+"_hist_ssp585_1850-2100.nc"
;;========================================
f1          = addfile(diri+fil1, "r")   
SAT1        = f1->tas(:,:,:) 
utc_date1   = cd_calendar(SAT1&time, -1)  ;change gregorian time to the UT 
ist1        = ind(yrStrt.eq.utc_date1)
ied1        = ind(yrLast.eq.utc_date1)            
SAT_model1  = SAT1(ist1:ied1,:,:)
printMinMax(SAT_model1, 0)
printVarSummary(SAT_model1)

f2          = addfile(diri+fil2, "r")   
SAT2        = f2->tas(:,:,:) 
utc_date2   = cd_calendar(SAT2&time, -1)  ;change gregorian time to the UT 
ist2        = ind(yrStrt.eq.utc_date2)
ied2        = ind(yrLast.eq.utc_date2)            
SAT_model2  = SAT2(ist2:ied2,:,:)
printMinMax(SAT_model2, 0)
printVarSummary(SAT_model2)

f3          = addfile(diri+fil3, "r")   
SAT3        = f3->tas(:,:,:) 
utc_date3   = cd_calendar(SAT3&time, -1)  ;change gregorian time to the UT 
ist3        = ind(yrStrt.eq.utc_date3)
ied3        = ind(yrLast.eq.utc_date3)            
SAT_model3  = SAT3(ist3:ied3,:,:)
printMinMax(SAT_model3, 0)
printVarSummary(SAT_model3)

f4          = addfile(diri+fil4, "r")   
SAT4        = f4->tas(:,:,:) 
utc_date4   = cd_calendar(SAT4&time, -1)  ;change gregorian time to the UT 
ist4        = ind(yrStrt.eq.utc_date4)
ied4        = ind(yrLast.eq.utc_date4)            
SAT_model4  = SAT4(ist4:ied4,:,:)
printMinMax(SAT_model4, 0)
printVarSummary(SAT_model4)

f5          = addfile(diri+fil5, "r")   
SAT5        = f5->tas(:,:,:) 
utc_date5   = cd_calendar(SAT5&time, -1)  ;change gregorian time to the UT 
ist5        = ind(yrStrt.eq.utc_date5)
ied5        = ind(yrLast.eq.utc_date5)            
SAT_model5  = SAT5(ist5:ied5,:,:)
printMinMax(SAT_model5, 0)
printVarSummary(SAT_model5)

f6          = addfile(diri+fil6, "r")   
SAT6        = f6->tas(:,:,:) 
utc_date6   = cd_calendar(SAT6&time, -1)  ;change gregorian time to the UT 
ist6        = ind(yrStrt.eq.utc_date6)
ied6        = ind(yrLast.eq.utc_date6)            
SAT_model6  = SAT6(ist6:ied6,:,:)
printMinMax(SAT_model6, 0)
printVarSummary(SAT_model6)

f7          = addfile(diri+fil7, "r")   
SAT7        = f7->tas(:,:,:) 
utc_date7   = cd_calendar(SAT7&time, -1)  ;change gregorian time to the UT 
ist7        = ind(yrStrt.eq.utc_date7)
ied7        = ind(yrLast.eq.utc_date7)            
SAT_model7  = SAT7(ist7:ied7,:,:)
printMinMax(SAT_model7, 0)
printVarSummary(SAT_model7)

f8          = addfile(diri+fil8, "r")   
SAT8        = f8->tas(:,:,:) 
utc_date8   = cd_calendar(SAT8&time, -1)  ;change gregorian time to the UT 
ist8        = ind(yrStrt.eq.utc_date8)
ied8        = ind(yrLast.eq.utc_date8)            
SAT_model8  = SAT8(ist8:ied8,:,:)
printMinMax(SAT_model8, 0)
printVarSummary(SAT_model8)

f9          = addfile(diri+fil9, "r")   
SAT9        = f9->tas(:,:,:) 
utc_date9   = cd_calendar(SAT9&time, -1)  ;change gregorian time to the UT 
ist9        = ind(yrStrt.eq.utc_date9)
ied9        = ind(yrLast.eq.utc_date9)            
SAT_model9  = SAT9(ist9:ied9,:,:)
printMinMax(SAT_model9, 0)
printVarSummary(SAT_model9)

f10          = addfile(diri+fil10, "r")   
SAT10        = f10->tas(:,:,:) 
utc_date10   = cd_calendar(SAT10&time, -1)  ;change gregorian time to the UT 
ist10        = ind(yrStrt.eq.utc_date10)
ied10        = ind(yrLast.eq.utc_date10)            
SAT_model10  = SAT10(ist10:ied10,:,:)
printMinMax(SAT_model10, 0)
printVarSummary(SAT_model10)

f11          = addfile(diri+fil11, "r")   
SAT11        = f11->tas(:,:,:) 
utc_date11   = cd_calendar(SAT11&time, -1)  ;change gregorian time to the UT 
ist11        = ind(yrStrt.eq.utc_date11)
ied11        = ind(yrLast.eq.utc_date11)            
SAT_model11  = SAT11(ist11:ied11,:,:)
printMinMax(SAT_model11, 0)
printVarSummary(SAT_model11)

f12          = addfile(diri+fil12, "r")   
SAT12        = f12->tas(:,:,:) 
utc_date12   = cd_calendar(SAT12&time, -1)  ;change gregorian time to the UT 
ist12        = ind(yrStrt.eq.utc_date12)
ied12        = ind(yrLast.eq.utc_date12)            
SAT_model12  = SAT12(ist12:ied12,:,:)
printMinMax(SAT_model12, 0)
printVarSummary(SAT_model12)


f13          = addfile(diri+fil13, "r")   
SAT13        = f13->tas(:,:,:) 
utc_date13   = cd_calendar(SAT13&time, -1)  ;change gregorian time to the UT 
ist13        = ind(yrStrt.eq.utc_date13)
ied13        = ind(yrLast.eq.utc_date13)            
SAT_model13  = SAT13(ist13:ied13,:,:)
printMinMax(SAT_model13, 0)
printVarSummary(SAT_model13)

f14          = addfile(diri+fil14, "r")   
SAT14        = f14->tas(:,:,:) 
utc_date14   = cd_calendar(SAT14&time, -1)  ;change gregorian time to the UT 
ist14        = ind(yrStrt.eq.utc_date14)
ied14        = ind(yrLast.eq.utc_date14)            
SAT_model14  = SAT14(ist14:ied14,:,:)
printMinMax(SAT_model14, 0)
printVarSummary(SAT_model14)

f15          = addfile(diri+fil15, "r")   
SAT15        = f15->tas(:,:,:) 
utc_date15   = cd_calendar(SAT15&time, -1)  ;change gregorian time to the UT 
ist15        = ind(yrStrt.eq.utc_date15)
ied15        = ind(yrLast.eq.utc_date15)            
SAT_model15  = SAT15(ist15:ied15,:,:)
printMinMax(SAT_model15, 0)
printVarSummary(SAT_model15)

f16          = addfile(diri+fil16, "r")   
SAT16        = f16->tas(:,:,:) 
utc_date16   = cd_calendar(SAT16&time, -1)  ;change gregorian time to the UT 
ist16        = ind(yrStrt.eq.utc_date16)
ied16        = ind(yrLast.eq.utc_date16)            
SAT_model16  = SAT16(ist16:ied16,:,:)
printMinMax(SAT_model16, 0)
printVarSummary(SAT_model16)

f17          = addfile(diri+fil17, "r")   
SAT17        = f17->tas(:,:,:) 
utc_date17   = cd_calendar(SAT17&time, -1)  ;change gregorian time to the UT 
ist17        = ind(yrStrt.eq.utc_date17)
ied17        = ind(yrLast.eq.utc_date17)            
SAT_model17  = SAT17(ist17:ied17,:,:)
printMinMax(SAT_model17, 0)
printVarSummary(SAT_model17)

f18          = addfile(diri+fil18, "r")   
SAT18        = f18->tas(:,:,:) 
utc_date18   = cd_calendar(SAT18&time, -1)  ;change gregorian time to the UT 
ist18        = ind(yrStrt.eq.utc_date18)
ied18        = ind(yrLast.eq.utc_date18)            
SAT_model18  = SAT18(ist18:ied18,:,:)
printMinMax(SAT_model18, 0)
printVarSummary(SAT_model18)

f19          = addfile(diri+fil19, "r")   
SAT19        = f19->tas(:,:,:) 
utc_date19   = cd_calendar(SAT19&time, -1)  ;change gregorian time to the UT 
ist19        = ind(yrStrt.eq.utc_date19)
ied19        = ind(yrLast.eq.utc_date19)            
SAT_model19  = SAT19(ist19:ied19,:,:)
printMinMax(SAT_model19, 0)
printVarSummary(SAT_model19)

f20          = addfile(diri+fil20, "r")   
SAT20        = f20->tas(:,:,:) 
utc_date20   = cd_calendar(SAT20&time, -1)  ;change gregorian time to the UT 
ist20        = ind(yrStrt.eq.utc_date20)
ied20        = ind(yrLast.eq.utc_date20)            
SAT_model20  = SAT20(ist20:ied20,:,:)
printMinMax(SAT_model20, 0)
printVarSummary(SAT_model20)

f21          = addfile(diri+fil21, "r")   
SAT21        = f21->tas(:,:,:) 
utc_date21   = cd_calendar(SAT21&time, -1)  ;change gregorian time to the UT 
ist21        = ind(yrStrt.eq.utc_date21)
ied21        = ind(yrLast.eq.utc_date21)            
SAT_model21  = SAT21(ist21:ied21,:,:)
printMinMax(SAT_model21, 0)
printVarSummary(SAT_model21)

f22          = addfile(diri+fil22, "r")   
SAT22        = f22->tas(:,:,:) 
utc_date22   = cd_calendar(SAT22&time, -1)  ;change gregorian tiyesme to the UT 
ist22        = ind(yrStrt.eq.utc_date22)
ied22        = ind(yrLast.eq.utc_date22)            
SAT_model22  = SAT22(ist22:ied22,:,:)
printMinMax(SAT_model22, 0)
printVarSummary(SAT_model22)

f23          = addfile(diri+fil23, "r")   
SAT23        = f23->tas(:,:,:) 
utc_date23   = cd_calendar(SAT23&time, -1)  ;change gregorian time to the UT 
ist23        = ind(yrStrt.eq.utc_date23)
ied23        = ind(yrLast.eq.utc_date23)            
SAT_model23  = SAT23(ist23:ied23,:,:)
printMinMax(SAT_model23, 0)
printVarSummary(SAT_model23)

f24          = addfile(diri+fil24, "r")   
SAT24        = f24->tas(:,:,:) 
utc_date24   = cd_calendar(SAT24&time, -1)  ;change gregorian time to the UT 
ist24        = ind(yrStrt.eq.utc_date24)
ied24        = ind(yrLast.eq.utc_date24)            
SAT_model24  = SAT24(ist24:ied24,:,:)
printMinMax(SAT_model24, 0)
printVarSummary(SAT_model24)

f25          = addfile(diri+fil25, "r")   
SAT25        = f25->tas(:,:,:) 
utc_date25   = cd_calendar(SAT25&time, -1)  ;change gregorian time to the UT 
ist25        = ind(yrStrt.eq.utc_date25)
ied25        = ind(yrLast.eq.utc_date25)            
SAT_model25  = SAT25(ist25:ied25,:,:)
printMinMax(SAT_model25, 0)
printVarSummary(SAT_model25)

f26          = addfile(diri+fil26, "r")   
SAT26        = f26->tas(:,:,:) 
utc_date26   = cd_calendar(SAT26&time, -1)  ;change gregorian time to the UT 
ist26        = ind(yrStrt.eq.utc_date26)
ied26        = ind(yrLast.eq.utc_date26)            
SAT_model26  = SAT26(ist26:ied26,:,:)
printMinMax(SAT_model26, 0)
printVarSummary(SAT_model26)

f27          = addfile(diri+fil27, "r")   
SAT27        = f27->tas(:,:,:) 
utc_date27   = cd_calendar(SAT27&time, -1)  ;change gregorian time to the UT 
ist27        = ind(yrStrt.eq.utc_date27)
ied27        = ind(yrLast.eq.utc_date27)            
SAT_model27  = SAT27(ist27:ied27,:,:)
printMinMax(SAT_model27, 0)
printVarSummary(SAT_model27)

f28          = addfile(diri+fil28, "r")   
SAT28        = f28->tas(:,:,:) 
utc_date28   = cd_calendar(SAT28&time, -1)  ;change gregorian time to the UT 
ist28        = ind(yrStrt.eq.utc_date28)
ied28        = ind(yrLast.eq.utc_date28)            
SAT_model28  = SAT28(ist28:ied28,:,:)
printMinMax(SAT_model28, 0)
printVarSummary(SAT_model28)

f29          = addfile(diri+fil29, "r")   
SAT29        = f29->tas(:,:,:) 
utc_date29   = cd_calendar(SAT29&time, -1)  ;change gregorian time to the UT 
ist29        = ind(yrStrt.eq.utc_date29)
ied29        = ind(yrLast.eq.utc_date29)            
SAT_model29  = SAT29(ist29:ied29,:,:)
printMinMax(SAT_model29, 0)
printVarSummary(SAT_model29)

f30          = addfile(diri+fil30, "r")   
SAT30        = f30->tas(:,:,:) 
utc_date30   = cd_calendar(SAT30&time, -1)  ;change gregorian time to the UT 
ist30        = ind(yrStrt.eq.utc_date30)
ied30        = ind(yrLast.eq.utc_date30)            
SAT_model30  = SAT30(ist30:ied30,:,:)
printMinMax(SAT_model30, 0)
printVarSummary(SAT_model30)

f31          = addfile(diri+fil31, "r")   
SAT31        = f31->tas(:,:,:) 
utc_date31   = cd_calendar(SAT31&time, -1)  ;change gregorian time to the UT 
ist31        = ind(yrStrt.eq.utc_date31)
ied31        = ind(yrLast.eq.utc_date31)            
SAT_model31  = SAT31(ist31:ied31,:,:)
printMinMax(SAT_model31, 0)
printVarSummary(SAT_model31)

f32          = addfile(diri+fil32, "r")   
SAT32        = f32->tas(:,:,:) 
utc_date32   = cd_calendar(SAT32&time, -1)  ;change gregorian time to the UT 
ist32        = ind(yrStrt.eq.utc_date32)
ied32        = ind(yrLast.eq.utc_date32)            
SAT_model32  = SAT32(ist32:ied32,:,:)
printMinMax(SAT_model32, 0)
printVarSummary(SAT_model32)

f33          = addfile(diri+fil33, "r")   
SAT33        = f33->tas(:,:,:) 
utc_date33   = cd_calendar(SAT33&time, -1)  ;change gregorian time to the UT 
ist33        = ind(yrStrt.eq.utc_date33)
ied33        = ind(yrLast.eq.utc_date33)            
SAT_model33  = SAT33(ist33:ied33,:,:)
printMinMax(SAT_model33, 0)
printVarSummary(SAT_model33)

f34          = addfile(diri+fil34, "r")   
SAT34        = f34->tas(:,:,:) 
utc_date34   = cd_calendar(SAT34&time, -1)  ;change gregorian time to the UT 
ist34        = ind(yrStrt.eq.utc_date34)
ied34        = ind(yrLast.eq.utc_date34)            
SAT_model34  = SAT34(ist34:ied34,:,:)
printMinMax(SAT_model34, 0)
printVarSummary(SAT_model34)

f35          = addfile(diri+fil35, "r")   
SAT35        = f35->tas(:,:,:) 
utc_date35   = cd_calendar(SAT35&time, -1)  ;change gregorian time to the UT 
ist35        = ind(yrStrt.eq.utc_date35)
ied35        = ind(yrLast.eq.utc_date35)            
SAT_model35  = SAT35(ist35:ied35,:,:)
printMinMax(SAT_model35, 0)
printVarSummary(SAT_model35)

f36          = addfile(diri+fil36, "r")   
SAT36        = f36->tas(:,:,:) 
utc_date36   = cd_calendar(SAT36&time, -1)  ;change gregorian time to the UT 
ist36        = ind(yrStrt.eq.utc_date36)
ied36        = ind(yrLast.eq.utc_date36)            
SAT_model36  = SAT36(ist36:ied36,:,:)
printMinMax(SAT_model36, 0)
printVarSummary(SAT_model36)

f37          = addfile(diri+fil37, "r")   
SAT37        = f37->tas(:,:,:) 
utc_date37   = cd_calendar(SAT37&time, -1)  ;change gregorian time to the UT 
ist37        = ind(yrStrt.eq.utc_date37)
ied37        = ind(yrLast.eq.utc_date37)            
SAT_model37  = SAT37(ist37:ied37,:,:)
printMinMax(SAT_model37, 0)
printVarSummary(SAT_model37)

f38          = addfile(diri+fil38, "r")   
SAT38        = f38->tas(:,:,:) 
utc_date38   = cd_calendar(SAT38&time, -1)  ;change gregorian time to the UT 
ist38        = ind(yrStrt.eq.utc_date38)
ied38        = ind(yrLast.eq.utc_date38)            
SAT_model38  = SAT38(ist38:ied38,:,:)
printMinMax(SAT_model38, 0)
printVarSummary(SAT_model38)

f39          = addfile(diri+fil39, "r")   
SAT39        = f39->tas(:,:,:) 
utc_date39   = cd_calendar(SAT39&time, -1)  ;change gregorian time to the UT 
ist39        = ind(yrStrt.eq.utc_date39)
ied39        = ind(yrLast.eq.utc_date39)            
SAT_model39  = SAT39(ist39:ied39,:,:)
printMinMax(SAT_model39, 0)
printVarSummary(SAT_model39)

f40          = addfile(diri+fil40, "r")   
SAT40        = f40->tas(:,:,:) 
utc_date40   = cd_calendar(SAT40&time, -1)  ;change gregorian time to the UT 
ist40        = ind(yrStrt.eq.utc_date40)
ied40        = ind(yrLast.eq.utc_date40)            
SAT_model40  = SAT40(ist40:ied40,:,:)
printMinMax(SAT_model40, 0)
printVarSummary(SAT_model40)

f41          = addfile(diri+fil41, "r")   
SAT41        = f41->tas(:,:,:) 
utc_date41   = cd_calendar(SAT41&time, -1)  ;change gregorian time to the UT 
ist41        = ind(yrStrt.eq.utc_date41)
ied41        = ind(yrLast.eq.utc_date41)            
SAT_model41  = SAT41(ist41:ied41,:,:)
printMinMax(SAT_model41, 0)
printVarSummary(SAT_model41)
;======================regression seasonal DJF data======================
SAT_seasonal_model1            = seasonal_mean(SAT_model1, iyear)
SAT_seasonal_model2            = seasonal_mean(SAT_model2, iyear)
SAT_seasonal_model3            = seasonal_mean(SAT_model3, iyear)
SAT_seasonal_model4            = seasonal_mean(SAT_model4, iyear)
SAT_seasonal_model5            = seasonal_mean(SAT_model5, iyear)
SAT_seasonal_model6            = seasonal_mean(SAT_model6, iyear)
SAT_seasonal_model7            = seasonal_mean(SAT_model7, iyear)
SAT_seasonal_model8            = seasonal_mean(SAT_model8, iyear)
SAT_seasonal_model9            = seasonal_mean(SAT_model9, iyear)
SAT_seasonal_model10           = seasonal_mean(SAT_model10, iyear)
SAT_seasonal_model11           = seasonal_mean(SAT_model11, iyear)
SAT_seasonal_model12           = seasonal_mean(SAT_model12, iyear)
SAT_seasonal_model13           = seasonal_mean(SAT_model13, iyear)
SAT_seasonal_model14           = seasonal_mean(SAT_model14, iyear)
SAT_seasonal_model15           = seasonal_mean(SAT_model15, iyear)
SAT_seasonal_model16           = seasonal_mean(SAT_model16, iyear)
SAT_seasonal_model17           = seasonal_mean(SAT_model17, iyear)
SAT_seasonal_model18           = seasonal_mean(SAT_model18, iyear)
SAT_seasonal_model19           = seasonal_mean(SAT_model19, iyear)
SAT_seasonal_model20           = seasonal_mean(SAT_model20, iyear)
SAT_seasonal_model21           = seasonal_mean(SAT_model21, iyear)
SAT_seasonal_model22           = seasonal_mean(SAT_model22, iyear)
SAT_seasonal_model23           = seasonal_mean(SAT_model23, iyear)
SAT_seasonal_model24           = seasonal_mean(SAT_model24, iyear)
SAT_seasonal_model25           = seasonal_mean(SAT_model25, iyear)
SAT_seasonal_model26           = seasonal_mean(SAT_model26, iyear)
SAT_seasonal_model27           = seasonal_mean(SAT_model27, iyear)
SAT_seasonal_model28           = seasonal_mean(SAT_model28, iyear)
SAT_seasonal_model29           = seasonal_mean(SAT_model29, iyear)
SAT_seasonal_model30           = seasonal_mean(SAT_model30, iyear)
SAT_seasonal_model31           = seasonal_mean(SAT_model31, iyear)
SAT_seasonal_model32           = seasonal_mean(SAT_model32, iyear)
SAT_seasonal_model33           = seasonal_mean(SAT_model33, iyear)
SAT_seasonal_model34           = seasonal_mean(SAT_model34, iyear)
SAT_seasonal_model35           = seasonal_mean(SAT_model35, iyear)
SAT_seasonal_model36           = seasonal_mean(SAT_model36, iyear)
SAT_seasonal_model37           = seasonal_mean(SAT_model37, iyear)
SAT_seasonal_model38           = seasonal_mean(SAT_model38, iyear)
SAT_seasonal_model39           = seasonal_mean(SAT_model39, iyear)
SAT_seasonal_model40           = seasonal_mean(SAT_model40, iyear)
SAT_seasonal_model41           = seasonal_mean(SAT_model41, iyear)
opt = True
; printVarSummary(SAT_seasonal_model1[0])
; printMinMax(SAT_seasonal_model1[0], opt)
; exit()

Cold_season_trend1 = regCoef(iyear, SAT_seasonal_model1[0])
Cold_season_trend2 = regCoef(iyear, SAT_seasonal_model2[0])
Cold_season_trend3 = regCoef(iyear, SAT_seasonal_model3[0])
Cold_season_trend4 = regCoef(iyear, SAT_seasonal_model4[0])
Cold_season_trend5 = regCoef(iyear, SAT_seasonal_model5[0]); EOF1 needed _reverse
Cold_season_trend6 = regCoef(iyear, SAT_seasonal_model6[0])
Cold_season_trend7 = regCoef(iyear, SAT_seasonal_model7[0])
Cold_season_trend8 = regCoef(iyear, SAT_seasonal_model8[0])
Cold_season_trend9 = regCoef(iyear, SAT_seasonal_model9[0])
Cold_season_trend10 = regCoef(iyear, SAT_seasonal_model10[0])
Cold_season_trend11 = regCoef(iyear, SAT_seasonal_model11[0])
Cold_season_trend12 = regCoef(iyear, SAT_seasonal_model12[0])
Cold_season_trend13 = regCoef(iyear, SAT_seasonal_model13[0])
Cold_season_trend14 = regCoef(iyear, SAT_seasonal_model14[0])
Cold_season_trend15 = regCoef(iyear, SAT_seasonal_model15[0])
Cold_season_trend16 = regCoef(iyear, SAT_seasonal_model16[0])
Cold_season_trend17 = regCoef(iyear, SAT_seasonal_model17[0])
Cold_season_trend18 = regCoef(iyear, SAT_seasonal_model18[0]) ;_reverse EOF1 needed
Cold_season_trend19 = regCoef(iyear, SAT_seasonal_model19[0])
Cold_season_trend20 = regCoef(iyear, SAT_seasonal_model20[0])
Cold_season_trend21 = regCoef(iyear, SAT_seasonal_model21[0])
Cold_season_trend22 = regCoef(iyear, SAT_seasonal_model22[0])
Cold_season_trend23 = regCoef(iyear, SAT_seasonal_model23[0]) ; EOF2*-1.0 needs
Cold_season_trend24 = regCoef(iyear, SAT_seasonal_model24[0])
Cold_season_trend25 = regCoef(iyear, SAT_seasonal_model25[0])
Cold_season_trend26 = regCoef(iyear, SAT_seasonal_model26[0])
Cold_season_trend27 = regCoef(iyear, SAT_seasonal_model27[0])
Cold_season_trend28 = regCoef(iyear, SAT_seasonal_model28[0])
Cold_season_trend29 = regCoef(iyear, SAT_seasonal_model29[0])
Cold_season_trend30 = regCoef(iyear, SAT_seasonal_model30[0])
Cold_season_trend31 = regCoef(iyear, SAT_seasonal_model31[0])
Cold_season_trend32 = regCoef(iyear, SAT_seasonal_model32[0])
Cold_season_trend33 = regCoef(iyear, SAT_seasonal_model33[0])
Cold_season_trend34 = regCoef(iyear, SAT_seasonal_model34[0]) ;_reverse (EOF1 needed)
Cold_season_trend35 = regCoef(iyear, SAT_seasonal_model35[0])
Cold_season_trend36 = regCoef(iyear, SAT_seasonal_model36[0])
Cold_season_trend37 = regCoef(iyear, SAT_seasonal_model37[0])
Cold_season_trend38 = regCoef(iyear, SAT_seasonal_model38[0])
Cold_season_trend39 = regCoef(iyear, SAT_seasonal_model39[0])
Cold_season_trend40 = regCoef(iyear, SAT_seasonal_model40[0])
Cold_season_trend41 = regCoef(iyear, SAT_seasonal_model41[0])

copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend1)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend2)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend3)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend4)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend5)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend6)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend7)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend8)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend9)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend10)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend11)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend12)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend13)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend14)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend15)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend16)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend17)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend18)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend19)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend20)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend21)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend22)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend23)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend24)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend25)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend26)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend27)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend28)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend29)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend30)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend31)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend32)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend33)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend34)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend35)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend36)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend37)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend38)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend39)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend40)
copy_VarMeta(SAT_model1(0,:,:), Cold_season_trend41)


Warm_season_trend1 = regCoef(iyear, SAT_seasonal_model1[1])
Warm_season_trend2 = regCoef(iyear, SAT_seasonal_model2[1])
Warm_season_trend3 = regCoef(iyear, SAT_seasonal_model3[1])
Warm_season_trend4 = regCoef(iyear, SAT_seasonal_model4[1])
Warm_season_trend5 = regCoef(iyear, SAT_seasonal_model5[1]); EOF1 needed _reverse
Warm_season_trend6 = regCoef(iyear, SAT_seasonal_model6[1])
Warm_season_trend7 = regCoef(iyear, SAT_seasonal_model7[1])
Warm_season_trend8 = regCoef(iyear, SAT_seasonal_model8[1])
Warm_season_trend9 = regCoef(iyear, SAT_seasonal_model9[1])
Warm_season_trend10 = regCoef(iyear, SAT_seasonal_model10[1])
Warm_season_trend11 = regCoef(iyear, SAT_seasonal_model11[1])
Warm_season_trend12 = regCoef(iyear, SAT_seasonal_model12[1])
Warm_season_trend13 = regCoef(iyear, SAT_seasonal_model13[1])
Warm_season_trend14 = regCoef(iyear, SAT_seasonal_model14[1])
Warm_season_trend15 = regCoef(iyear, SAT_seasonal_model15[1])
Warm_season_trend16 = regCoef(iyear, SAT_seasonal_model16[1])
Warm_season_trend17 = regCoef(iyear, SAT_seasonal_model17[1])
Warm_season_trend18 = regCoef(iyear, SAT_seasonal_model18[1]) ;_reverse EOF1 needed
Warm_season_trend19 = regCoef(iyear, SAT_seasonal_model19[1])
Warm_season_trend20 = regCoef(iyear, SAT_seasonal_model20[1])
Warm_season_trend21 = regCoef(iyear, SAT_seasonal_model21[1])
Warm_season_trend22 = regCoef(iyear, SAT_seasonal_model22[1])
Warm_season_trend23 = regCoef(iyear, SAT_seasonal_model23[1]) ; EOF2*-1.0 needs
Warm_season_trend24 = regCoef(iyear, SAT_seasonal_model24[1])
Warm_season_trend25 = regCoef(iyear, SAT_seasonal_model25[1])
Warm_season_trend26 = regCoef(iyear, SAT_seasonal_model26[1])
Warm_season_trend27 = regCoef(iyear, SAT_seasonal_model27[1])
Warm_season_trend28 = regCoef(iyear, SAT_seasonal_model28[1])
Warm_season_trend29 = regCoef(iyear, SAT_seasonal_model29[1])
Warm_season_trend30 = regCoef(iyear, SAT_seasonal_model30[1])
Warm_season_trend31 = regCoef(iyear, SAT_seasonal_model31[1])
Warm_season_trend32 = regCoef(iyear, SAT_seasonal_model32[1])
Warm_season_trend33 = regCoef(iyear, SAT_seasonal_model33[1])
Warm_season_trend34 = regCoef(iyear, SAT_seasonal_model34[1]) ;_reverse (EOF1 needed)
Warm_season_trend35 = regCoef(iyear, SAT_seasonal_model35[1])
Warm_season_trend36 = regCoef(iyear, SAT_seasonal_model36[1])
Warm_season_trend37 = regCoef(iyear, SAT_seasonal_model37[1])
Warm_season_trend38 = regCoef(iyear, SAT_seasonal_model38[1])
Warm_season_trend39 = regCoef(iyear, SAT_seasonal_model39[1])
Warm_season_trend40 = regCoef(iyear, SAT_seasonal_model40[1])
Warm_season_trend41 = regCoef(iyear, SAT_seasonal_model41[1])


copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend1)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend2)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend3)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend4)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend5)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend6)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend7)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend8)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend9)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend10)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend11)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend12)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend13)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend14)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend15)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend16)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend17)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend18)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend19)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend20)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend21)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend22)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend23)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend24)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend25)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend26)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend27)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend28)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend29)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend30)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend31)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend32)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend33)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend34)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend35)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend36)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend37)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend38)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend39)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend40)
copy_VarMeta(SAT_model1(0,:,:), Warm_season_trend41)

printVarSummary(Warm_season_trend41)
printVarSummary(Warm_season_trend11)
printVarSummary(Warm_season_trend1)
printVarSummary(Warm_season_trend21)
printVarSummary(Warm_season_trend31)

printMinMax(Cold_season_trend31, opt)
printMinMax(Warm_season_trend31, opt)
; exit()
;=======================Trends====================
warm_model = new((/41,73,144/), float)
warm_model!0 = "Number"
warm_model&Number = ispan(1, 41, 1)
warm_model(0,:,:) = Warm_season_trend1
warm_model(1,:,:) = Warm_season_trend2
warm_model(2,:,:) = Warm_season_trend3
warm_model(3,:,:) = Warm_season_trend4
warm_model(4,:,:) = Warm_season_trend5
warm_model(5,:,:) = Warm_season_trend6
warm_model(6,:,:) = Warm_season_trend7
warm_model(7,:,:) = Warm_season_trend8
warm_model(8,:,:) = Warm_season_trend9
warm_model(9,:,:) = Warm_season_trend10
warm_model(10,:,:) = Warm_season_trend11
warm_model(11,:,:) = Warm_season_trend12
warm_model(12,:,:) = Warm_season_trend13
warm_model(13,:,:) = Warm_season_trend14
warm_model(14,:,:) = Warm_season_trend15
warm_model(15,:,:) = Warm_season_trend16
warm_model(16,:,:) = Warm_season_trend17
warm_model(17,:,:) = Warm_season_trend18
warm_model(18,:,:) = Warm_season_trend19
warm_model(19,:,:) = Warm_season_trend20
warm_model(20,:,:) = Warm_season_trend21
warm_model(21,:,:) = Warm_season_trend22
warm_model(22,:,:) = Warm_season_trend23
warm_model(23,:,:) = Warm_season_trend24
warm_model(24,:,:) = Warm_season_trend25
warm_model(25,:,:) = Warm_season_trend26
warm_model(26,:,:) = Warm_season_trend27
warm_model(27,:,:) = Warm_season_trend28
warm_model(28,:,:) = Warm_season_trend29
warm_model(29,:,:) = Warm_season_trend30
warm_model(30,:,:) = Warm_season_trend31
warm_model(31,:,:) = Warm_season_trend32
warm_model(32,:,:) = Warm_season_trend33
warm_model(33,:,:) = Warm_season_trend34
warm_model(34,:,:) = Warm_season_trend35
warm_model(35,:,:) = Warm_season_trend36
warm_model(36,:,:) = Warm_season_trend37
warm_model(37,:,:) = Warm_season_trend38
warm_model(38,:,:) = Warm_season_trend39
warm_model(39,:,:) = Warm_season_trend40
warm_model(40,:,:) = Warm_season_trend41

printVarSummary(warm_model)
;=====================confidence level===================
cold_model = new((/41,73,144/), float)
; copy_Var(cold_model1, cold_model(0,:,:,:))
cold_model(0,:,:) = Cold_season_trend1
cold_model(1,:,:) = Cold_season_trend2
cold_model(2,:,:) = Cold_season_trend3
cold_model(3,:,:) = Cold_season_trend4
cold_model(4,:,:) = Cold_season_trend5
cold_model(5,:,:) = Cold_season_trend6
cold_model(6,:,:) = Cold_season_trend7
cold_model(7,:,:) = Cold_season_trend8
cold_model(8,:,:) = Cold_season_trend9
cold_model(9,:,:) = Cold_season_trend10
cold_model(10,:,:) = Cold_season_trend11
cold_model(11,:,:) = Cold_season_trend12
cold_model(12,:,:) = Cold_season_trend13
cold_model(13,:,:) = Cold_season_trend14
cold_model(14,:,:) = Cold_season_trend15
cold_model(15,:,:) = Cold_season_trend16
cold_model(16,:,:) = Cold_season_trend17
cold_model(17,:,:) = Cold_season_trend18
cold_model(18,:,:) = Cold_season_trend19
cold_model(19,:,:) = Cold_season_trend20
cold_model(20,:,:) = Cold_season_trend21
cold_model(21,:,:) = Cold_season_trend22
cold_model(22,:,:) = Cold_season_trend23
cold_model(23,:,:) = Cold_season_trend24
cold_model(24,:,:) = Cold_season_trend25
cold_model(25,:,:) = Cold_season_trend26
cold_model(26,:,:) = Cold_season_trend27
cold_model(27,:,:) = Cold_season_trend28
cold_model(28,:,:) = Cold_season_trend29
cold_model(29,:,:) = Cold_season_trend30
cold_model(30,:,:) = Cold_season_trend31
cold_model(31,:,:) = Cold_season_trend32
cold_model(32,:,:) = Cold_season_trend33
cold_model(33,:,:) = Cold_season_trend34
cold_model(34,:,:) = Cold_season_trend35
cold_model(35,:,:) = Cold_season_trend36
cold_model(36,:,:) = Cold_season_trend37
cold_model(37,:,:) = Cold_season_trend38
cold_model(38,:,:) = Cold_season_trend39
cold_model(39,:,:) = Cold_season_trend40
cold_model(40,:,:) = Cold_season_trend41


Cold_season_MME = dim_avg_n_Wrap(cold_model, 0)
Warm_season_MME = dim_avg_n_Wrap(warm_model, 0)
printVarSummary(Cold_season_MME)
printMinMax(Cold_season_MME, opt)
printMinMax(Warm_season_MME, opt)

;************************************************
; use mask function to mask out land and then ocean data
; ocean=0,[0]land=1,sea_ice=2
;************************************************
  a         = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc", "r") 
  lsdata    = a->LSMASK
  lsm       = landsea_mask(lsdata, SAT_model1&lat, SAT_model1&lon)
  Warm_season_MME       = mask(Warm_season_MME, lsm.eq.0, False)
  Warm_season_MME       = mask(Warm_season_MME, lsm.eq.2, False)

  Cold_season_MME       = mask(Cold_season_MME, lsm.eq.0, False)
  Cold_season_MME       = mask(Cold_season_MME, lsm.eq.2, False)

delete(f1)
delete(a)
delete(lsdata)
delete(lsm)
;======================Obs=============================
iyrs = 197801
iyre = 202012

;HadCRU 5.1 version ensemble series global annual 
; the dimension is tas [200(realization), 172 (time)]
opt = True
; for the our focused period
f1 = addfile("/disk_raid6/pub/aru/Proposal-reproduce/MLOST/air.mon.anom.RG.nc", "r")
time  = f1->time
ymd = cd_calendar(time, -1)
iStrt=ind(ymd.eq.iyrs)              
iLast=ind(ymd.eq.iyre)

Land = f1->air(iStrt:iLast,:,:)
printVarSummary(Land) 
Land_clm = clmMonTLL(Land)
Land_ano = calcMonAnomTLL(Land, Land_clm)
printVarSummary(Land_ano)
printMinMax(Land_ano, opt)

Land_ano_cold = new((/42, 73,144/), float)
j = 0
do i= 10,502,12
    Land_ano_cold(j,:,:)=(Land_ano(i,:,:)+Land_ano(i+1,:,:)+Land_ano(i+2,:,:)+Land_ano(i+3,:,:)+Land_ano(i+4,:,:)+Land_ano(i+5,:,:))/6.0
    j = j+1
end do
copy_VarMeta(Land_ano(0,:,:), Land_ano_cold(0,:,:))
Land_ano_cold!0 = "time"
Land_ano_cold&time = ispan(1979, 2020, 1)
printMinMax(Land_ano_cold, opt)

Land_ano_warm = new((/42, 73,144/), float) ;"MJJASO"
j = 0
do i= 16,508,12
    Land_ano_warm(j,:,:)=(Land_ano(i,:,:)+Land_ano(i+1,:,:)+Land_ano(i+2,:,:)+Land_ano(i+3,:,:)+Land_ano(i+4,:,:)+Land_ano(i+5,:,:))/6.0
    j = j+1
end do
copy_VarMeta(Land_ano(0,:,:), Land_ano_warm(0,:,:))
Land_ano_warm!0 = "time"
Land_ano_warm&time = ispan(1979, 2020, 1)
printMinMax(Land_ano_warm, opt)

x = ispan(1979, 2020, 1)
Trend_cold_season = regCoef(x, Land_ano_cold(lat|:, lon|:, time|:))
copy_VarMeta(Land_ano_cold(0,:,:), Trend_cold_season)
printVarSummary(Trend_cold_season)
printMinMax(Trend_cold_season, opt)

Trend_warm_season = regCoef(x, Land_ano_warm(lat|:, lon|:, time|:))
copy_VarMeta(Land_ano_warm(0,:,:), Trend_warm_season)
printVarSummary(Trend_warm_season)
printMinMax(Trend_warm_season, opt)

;************************************************
; use mask function to mask out land and then ocean data
; ocean=0,[0]land=1,sea_ice=2
;************************************************
  a         = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc", "r") 
  lsdata    = a->LSMASK
  lsm       = landsea_mask(lsdata, Land_ano&lat, Land_ano&lon)
  Trend_warm_season       = mask(Trend_warm_season, lsm.eq.0, False)
  Trend_warm_season       = mask(Trend_warm_season, lsm.eq.2, False)

  Trend_cold_season       = mask(Trend_cold_season, lsm.eq.0, False)
  Trend_cold_season       = mask(Trend_cold_season, lsm.eq.2, False)

;=================================================================
wks = gsn_open_wks("eps", "MLOST&CMIP6-1979-2020-Land-SAT-trend-Smask")
gsn_define_colormap(wks, "BlueDarkRed18")
plot = new(4, graphic)

res = True
res@gsnFrame     = False
res@gsnDraw      = False
res@gsnAddCyclic = True

res@mpFillOn        = False
res@mpMinLatF       = -60       
res@mpMaxLatF       = 90  
res@mpMinLonF       = -180
res@mpMaxLonF       = 180
res@mpCenterLonF    = 0
res@cnFillOn             = True     ; turn on color fill
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnLevels             = (/-0.2,-0.1, -0.02, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1/)
res@cnFillColors         = (/2, 4, 6, 10, 11,12, 13, 14, 15, 16, 17, 18, 19/)

; res@cnLevelSelectionMode = "ManualLevels"
; res@cnMaxLevelValF       = 0.1
; res@cnMinLevelValF       = -0.1
; res@cnLevelSpacingF      = 0.01

res@cnLinesOn            = True    ; True is default
res@cnLineColor          = "Gray"
res@cnLineLabelsOn       = False      ; True is default
res@cnInfoLabelOn         = False  
res@lbLabelBarOn         = False      ; turn off individual lb's
res@tmXTOn                = True
res@tmYROn                = True 
res@tmXBLabels            = False 
res@tmYLOn                = True 
res@pmTickMarkDisplayMode = "NoCreate" 

res@gsnContourNegLineDashPattern = 16
res@gsnContourPosLineDashPattern = 0
res@gsnContourZeroLineThicknessF = 0 
res@gsnLeftStringFontHeightF = 0.023
res@gsnRightStringFontHeightF = 0.023
; res@gsnCenterStringFontHeightF = 0.023 
res@tmXBLabelFontHeightF = 0.025
res@tmYLLabelFontHeightF = 0.025
res@tiMainString        = " "
res@gsnLeftString       = " "
res@gsnRightString      = " "
; res@lbBoxEndCapStyle = "TriangleBothEnds"
; res@lbBoxLineColor   = 0
res@pmLabelBarOrthogonalPosF = 0.18
res@tmYLLabelStride    = 1
; res@tmXBLabelStride    = 3
; res@tmXTLabelStride    = 3
plot(0) = gsn_csm_contour_map(wks, Trend_cold_season, res)

res2 = res
res2@tmYLLabels = False
plot(1) = gsn_csm_contour_map(wks, Cold_season_MME, res2)


res1 = True
res1@gsnFrame     = False
res1@gsnDraw      = False
res1@gsnAddCyclic = True

res1@mpFillOn        = False
res1@mpMinLatF       = -60       
res1@mpMaxLatF       = 90  
res1@mpMinLonF       = -180
res1@mpMaxLonF       = 180
res1@mpCenterLonF    = 0
res1@cnFillOn             = True     ; turn on color fill
res1@cnLevelSelectionMode = "ExplicitLevels"
res1@cnLevels             = (/-0.2,-0.1, -0.02, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1/)
res1@cnFillColors         = (/2, 4, 6, 10, 11,12, 13, 14, 15, 16, 17, 18, 19/)

; res1@cnLevelSelectionMode = "ManualLevels"
; res1@cnMaxLevelValF       = 0.1
; res1@cnMinLevelValF       = -0.1
; res1@cnLevelSpacingF      = 0.01

res1@cnLinesOn            = True    ; True is default
res1@cnLineColor          = "Gray"
res1@cnLineLabelsOn       = False      ; True is default
res1@cnInfoLabelOn         = False  
res1@lbLabelBarOn         = False      ; turn off individual lb's
res1@tmXTOn                = True
res1@tmYROn                = True 
res1@tmYLOn                = True 
res1@pmTickMarkDisplayMode = "Never" 

res1@gsnContourNegLineDashPattern = 16
res1@gsnContourPosLineDashPattern = 0
res1@gsnContourZeroLineThicknessF = 0 
res1@gsnLeftStringFontHeightF = 0.023
res1@gsnRightStringFontHeightF = 0.023
; res1@gsnCenterStringFontHeightF = 0.023 
res1@tmXBLabelFontHeightF = 0.025
res1@tmYLLabelFontHeightF = 0.025
res1@tiMainString        = " "
res1@gsnLeftString       = " "
res1@gsnRightString      = " "
; res1@lbBoxEndCapStyle = "TriangleBothEnds"
; res1@lbBoxLineColor   = 0
res1@pmLabelBarOrthogonalPosF = 0.18
res1@tmYLLabelStride    = 1
res1@tmXBLabelStride    = 3
plot(2) = gsn_csm_contour_map(wks, Trend_warm_season, res1)

res3 = res1
res3@tmYLLabels = False
plot(3) = gsn_csm_contour_map(wks, Warm_season_MME, res3)

pres  = True
pres@gsnMaximize = True
pres@gsnPanelLabelBar  = True
pres@pmLabelBarWidthF                   = 0.7             ; make thinner
; pres@pmLabelBarHeightF                = 0.04
pres@lbLabelFontHeightF                 = 0.016
; pres@lbBoxEndCapStyle                 = "TriangleBothEnds"
; pres@lbBoxLinesOn                     = False
pres@pmLabelBarOrthogonalPosF           = -0.03
pres@lbTitleString                      = "~F22~SAT~F18~(~F18~I~F22~C~F22~ per 42 years~F18~)"
pres@lbTitleFontHeightF                 = 0.016
pres@lbTitlePosition                    = "Bottom"
pres@gsnPanelBottom                     = 0.03
; pres@gsnPanelYWhiteSpacePercent=0.0001
; pres@gsnPanelXWhiteSpacePercent=0.0001
; ; panel plot only resources
;   resP                     = True         ; modify the panel plot
;   resP@gsnFrame            = False
;   resP@gsnMaximize         = True         ; large format
;   resP@gsnPanelLabelBar    = True         ; add common colorbar
;   resP@lbLabelFontHeightF       = 0.012
;   resP@lbBoxLineColor           = 0
;   resP@pmLabelBarWidthF         = 0.65
; ; resP@gsnPanelCenter           = False
;   resP@pmLabelBarOrthogonalPosF =-0.03
;   resP@lbTitleString                      = "~F22~SAT~F18~(~F18~I~F22~C~F22~ per 42 years~F18~)"
;   resP@lbTitleFontHeightF                 = 0.020
;   resP@lbTitlePosition                    = "Bottom"
;   resP@gsnPanelBottom                     = 0.03
;   ; resP@gsnPanelMainString  = "Projection"+season+": "+ "1930-2099"
; ; *******************************************
; ; first plot
; ; *******************************************
;   do i=0,40,1
;      res@gsnLeftString  = "Cold season Trend:"
;      res@gsnRightString = Name(i)+"-M"+(i+1)
;      plot(i)=gsn_csm_contour_map(wks, cold_model(i,:,:), res)
;   end do
;   do i = 0, 10
;   gsn_panel(wks,plot(4*i:4*(i+1)-1),(/2,2/),resP)     ; now draw as one plot
;   frame(wks)
;   end do 
gsn_panel(wks, plot, (/2,2/), pres)
end