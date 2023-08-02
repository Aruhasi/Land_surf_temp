load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin

iyr_str = 190001
iyr_end = 201412

iStr_15yr = 200001
iStr_35yr = 198001
iStr_65yr = 195001

season2 = "DJF" 
season1 = "JJA"

opt = True

dir = "/work/mh0033/m301036/LSAT/Data/"


f1 = addfile(dir+"air.mon.anom.RG.nc", "r")  ;MLOST land SAT
f2 = addfile(dir+"CRUTEM.5.0.1.0.anomalies.nc", "r") ;CRUTEM5 land SAT
f3 = addfile(dir+"air.2x2.250.mon.anom.land.nc", "r") ;GISTEMP

time = f1->time
ym=cd_calendar(time, -1)
iStrt = ind(ym.eq.iStr_65yr)
iLast = ind(ym.eq.iyr_end)

LSAT_MLOST = f1->air(iStrt:iLast,:,:)  ;360 needs to convert to the -180 to 180
; LSAT_MLOST = lonFlip(LSAT_MLOST)
printVarSummary(LSAT_MLOST)
; exit
LSAT_MLOST@_FillValue = -9.96921e+36
LSAT_MLOST_clm = clmMonTLL(LSAT_MLOST)
LSAT_MLOST_ano = calcMonAnomTLL(LSAT_MLOST, LSAT_MLOST_clm)
printVarSummary(LSAT_MLOST_ano)
printMinMax(LSAT_MLOST_ano, opt)

time1 = f2->time
ym1=cd_calendar(time1, -1)
iStrt1 = ind(ym1.eq.iStr_65yr)
iLast1 = ind(ym1.eq.iyr_end)

LSAT_CRUTEM5 = f2->tas(iStrt1:iLast1,:,:)
LSAT_CRUTEM5@_FillValue = -1e+30
LSAT_CRUTEM5 = LSAT_CRUTEM5 -273.15
LSAT_CRUTEM5@units = "degC"
printVarSummary(LSAT_CRUTEM5)
LSAT_CRUTEM5_clm = clmMonTLL(LSAT_CRUTEM5)
LSAT_CRUTEM5_ano = calcMonAnomTLL(LSAT_CRUTEM5, LSAT_CRUTEM5_clm)
printVarSummary(LSAT_CRUTEM5_ano)
printMinMax(LSAT_CRUTEM5_ano, opt)


time2 = f3->time
ym2=cd_calendar(time2, -1)
iStrt2 = ind(ym2.eq.iStr_65yr)
iLast2 = ind(ym2.eq.iyr_end)

LSAT_GISTEMP = f3->air(iStrt2:iLast2,::-1,:) ;360 needs to convert to the -180 to 180
LSAT_GISTEMP@_FillValue = 9999
LSAT_GISTEMP = lonFlip(LSAT_GISTEMP)
printVarSummary(LSAT_GISTEMP)
LSAT_GISTEMP_clm = clmMonTLL(LSAT_GISTEMP)
printVarSummary(LSAT_GISTEMP_clm)
; exit()

LSAT_GISTEMP_ano = calcMonAnomTLL(LSAT_GISTEMP, LSAT_GISTEMP_clm)
printVarSummary(LSAT_GISTEMP_ano)
printMinMax(LSAT_GISTEMP_ano, opt)

dims = dimsizes(LSAT_MLOST_ano)
;************************************************
; use mask function to mask out land and then ocean data
; ocean=0,[0]land=1,sea_ice=2
;************************************************
a         = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc", "r") 
lsdata    = a->LSMASK
lsm       = landsea_mask(lsdata, LSAT_MLOST_ano&lat, LSAT_MLOST_ano&lon)
LSAT_MLOST_ano       = mask(LSAT_MLOST_ano, lsm.eq.0, False)
LSAT_MLOST_ano       = mask(LSAT_MLOST_ano, lsm.eq.2, False)

; MLOST_JJA_trend       = mask(MLOST_JJA_trend, lsm.eq.0, False)
; MLOST_JJA_trend       = mask(MLOST_JJA_trend, lsm.eq.2, False)

; MLOST_DJF_trend       = mask(MLOST_DJF_trend, lsm.eq.0, False)
; MLOST_DJF_trend       = mask(MLOST_DJF_trend, lsm.eq.2, False)
;calculate the annual mean SAT anomalies, DJF and JJA 
nyears = dims(0)/12  ; Assuming monthly data
MLOST_annual_mean = new((/nyears, dims(1), dims(2)/), typeof(LSAT_MLOST_ano))

do iy = 0, nyears-1
    MLOST_annual_mean(iy,:,:) = dim_avg_n_Wrap(LSAT_MLOST_ano(iy*12:iy*12+11,:,:), 0)  ; Calculate mean along the time dimension (0)
end do
MLOST_annual_mean!0 = "year"
MLOST_annual_mean@year = ispan(1950,2014,1)
; MLOST_annual = month_to_annual(LSAT_MLOST_ano, 1)
printVarSummary(MLOST_annual_mean)
printMinMax(MLOST_annual_mean, opt)

MLOST_JJA = month_to_season(LSAT_MLOST_ano, season1)
printVarSummary(MLOST_JJA)
printMinMax(MLOST_JJA, opt)

MLOST_DJF = month_to_season(LSAT_MLOST_ano, season2)
printVarSummary(MLOST_DJF)
printMinMax(MLOST_DJF, opt)

dims1 = dimsizes(LSAT_CRUTEM5_ano)
CRUTEM5_annual_mean = new((/nyears, dims1(1), dims1(2)/), typeof(LSAT_CRUTEM5_ano))

do iy = 0, nyears-1
    CRUTEM5_annual_mean(iy,:,:) = dim_avg_n_Wrap(LSAT_CRUTEM5_ano(iy*12:iy*12+11,:,:), 0)  ; Calculate mean along the time dimension (0)
end do
CRUTEM5_annual_mean!0 = "year"
CRUTEM5_annual_mean@year = ispan(1950,2014,1)
; CRUTEM5_annual = month_to_annual(LSAT_CRUTEM5_ano, 1)
printVarSummary(CRUTEM5_annual_mean)
printMinMax(CRUTEM5_annual_mean, opt)

CRUTEM5_JJA = month_to_season(LSAT_CRUTEM5_ano, season1)
printVarSummary(CRUTEM5_JJA)
printMinMax(CRUTEM5_JJA, opt)

CRUTEM5_DJF = month_to_season(LSAT_CRUTEM5_ano, season2)
printVarSummary(CRUTEM5_DJF)
printMinMax(CRUTEM5_DJF, opt)

delete(dims1)
dims1 = dimsizes(LSAT_GISTEMP_ano)
GISTEMP_annual_mean = new((/nyears, dims1(1), dims1(2)/), typeof(LSAT_GISTEMP_ano))

do iy = 0, nyears-1
    GISTEMP_annual_mean(iy,:,:) = dim_avg_n_Wrap(LSAT_GISTEMP_ano(iy*12:iy*12+11,:,:), 0)  ; Calculate mean along the time dimension (0)
end do
GISTEMP_annual_mean!0 = "year"
GISTEMP_annual_mean@year = ispan(1950,2014,1)
; GISTEMP_annual = month_to_annual(LSAT_GISTEMP_ano, 1)
printVarSummary(GISTEMP_annual_mean)
printMinMax(GISTEMP_annual_mean, opt)

GISTEMP_JJA = month_to_season(LSAT_GISTEMP_ano, season1)
printVarSummary(GISTEMP_JJA)
printMinMax(GISTEMP_JJA, opt)

GISTEMP_DJF = month_to_season(LSAT_GISTEMP_ano, season2)
printVarSummary(GISTEMP_DJF)
printMinMax(GISTEMP_DJF, opt)

;calcuate NH annual mean and seasonal mean LSAT anomalies
;=================================================================
;create weights:  sqrt(cos(lat))   [or sqrt(gw) ]
;================================================
lat = f1->lat
lon = f1->lon
jlat  = dimsizes( lat )
; print(lat)

rad    = 4.0*atan(1.0)/180.0
re     = 6371220.0
rr     = re*rad

dlon   = abs(lon(2)-lon(1))*rr
dx     = dlon*cos(lat*rad)
;                                     lat can have variable spacing
dy     = new ( jlat, typeof(dx))
                                                          ; close enough
dy(0)  = abs(lat(2)-lat(1))*rr
dy(1:jlat-2)  = abs(lat(2:jlat-1)-lat(0:jlat-3))*rr*0.5   
dy(jlat-1)    = abs(lat(jlat-1)-lat(jlat-2))*rr

area   = dx*dy    
clat   = cos(lat*rad) 
; print(area)
; =================================================================
; weight all observations 
; =================================================================
LSAT_MLOST_weighted = LSAT_MLOST_ano*conform(LSAT_MLOST_ano, clat, 1)

MLOST_NH = wgt_areaave_Wrap(LSAT_MLOST_ano(:,18:35,:), area(18:35), 1.0, 1)
printVarSummary(MLOST_NH)
printMinMax(MLOST_NH, 0)

MLOST_NH_annual = month_to_annual(MLOST_NH, 1)
printVarSummary(MLOST_NH_annual)
printMinMax(MLOST_NH_annual, opt)
;calculate the 15 year trend 
x =ispan(1950,2014,1)

MLOST_annual_trend = regCoef(x, MLOST_annual_mean(lat|:, lon|:, year|:))
MLOST_JJA_trend = regCoef(x, MLOST_JJA(lat|:, lon|:, time|:))
MLOST_DJF_trend = regCoef(x, MLOST_DJF(lat|:, lon|:, time|:))
MLOST_annual_trend = MLOST_annual_trend*65
MLOST_JJA_trend = MLOST_JJA_trend*65
MLOST_DJF_trend = MLOST_DJF_trend*65
copy_VarMeta(MLOST_annual_mean(0,:,:), MLOST_annual_trend)
copy_VarMeta(MLOST_annual_mean(0,:,:), MLOST_JJA_trend)
copy_VarMeta(MLOST_annual_mean(0,:,:), MLOST_DJF_trend)

printMinMax(MLOST_annual_trend,0)
printMinMax(MLOST_JJA_trend,0)
printMinMax(MLOST_DJF_trend,0)

CRUTEM5_annual_trend = regCoef(x, CRUTEM5_annual_mean(latitude|:, longitude|:, year|:))
CRUTEM5_JJA_trend = regCoef(x, CRUTEM5_JJA(latitude|:, longitude|:, time|:))
CRUTEM5_DJF_trend = regCoef(x, CRUTEM5_DJF(latitude|:, longitude|:, time|:))
CRUTEM5_annual_trend = CRUTEM5_annual_trend*65
CRUTEM5_JJA_trend = CRUTEM5_JJA_trend*65
CRUTEM5_DJF_trend = CRUTEM5_DJF_trend*65
copy_VarMeta(CRUTEM5_annual_mean(0,:,:), CRUTEM5_annual_trend)
copy_VarMeta(CRUTEM5_annual_mean(0,:,:), CRUTEM5_JJA_trend)
copy_VarMeta(CRUTEM5_annual_mean(0,:,:), CRUTEM5_DJF_trend)

; printMinMax(CRUTEM5_annual_trend,0)
; printMinMax(CRUTEM5_JJA_trend,0)
; printMinMax(CRUTEM5_DJF_trend,0)


GISTEMP_annual_trend = regCoef(x, GISTEMP_annual_mean(lat|:, lon|:, year|:))
GISTEMP_JJA_trend = regCoef(x, GISTEMP_JJA(lat|:, lon|:, time|:))
GISTEMP_DJF_trend = regCoef(x, GISTEMP_DJF(lat|:, lon|:, time|:))
GISTEMP_annual_trend = GISTEMP_annual_trend*65
GISTEMP_JJA_trend = GISTEMP_JJA_trend*65
GISTEMP_DJF_trend = GISTEMP_DJF_trend*65
copy_VarMeta(GISTEMP_annual_mean(0,:,:), GISTEMP_annual_trend)
copy_VarMeta(GISTEMP_annual_mean(0,:,:), GISTEMP_JJA_trend)
copy_VarMeta(GISTEMP_annual_mean(0,:,:), GISTEMP_DJF_trend)

; printMinMax(GISTEMP_annual_trend,0)
; printMinMax(GISTEMP_JJA_trend,0)
; printMinMax(GISTEMP_DJF_trend,0)


printMinMax(MLOST_annual_trend,0)
printMinMax(MLOST_JJA_trend,0)
printMinMax(MLOST_DJF_trend,0)
; exit

;=================================================================
wks = gsn_open_wks("pdf", "Obs-1950-2014-Land-SAT-trend-Smask")
gsn_define_colormap(wks, "BlueDarkRed18")
plot = new(9, graphic)

res = True
res@gsnFrame     = False
res@gsnDraw      = False
res@gsnAddCyclic = True

res@mpFillOn        = False
res@mpMinLatF       = 0       
res@mpMaxLatF       = 90  
res@mpMinLonF       = -180
res@mpMaxLonF       = 180
res@mpCenterLonF    = 0
res@cnFillOn             = True     ; turn on color fill

; res@cnLevelSelectionMode = "ExplicitLevels"
; res@cnLevels             = (/-5.0,-4.5, -4.0, -3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0/)
; res@cnFillColors         = (/2, 4, 6, 10, 11,12, 13, 14, 15, 16, 17, 18, 19/)

res@cnLevelSelectionMode = "ManualLevels"
res@cnMaxLevelValF       = 3.0
res@cnMinLevelValF       = -3.0
res@cnLevelSpacingF      = 0.3

res@cnLinesOn            = False    ; True is default
res@cnLineColor          = "Gray"
res@cnLineLabelsOn       = False      ; True is default
res@cnInfoLabelOn         = False  
res@lbLabelBarOn          = False      ; turn off individual lb's
res@tmXTOn                = True
res@tmYROn                = True 
res@tmXBOn                = False
res@tmYLOn                = True 

res@gsnContourNegLineDashPattern = 16
res@gsnContourPosLineDashPattern = 0
res@gsnContourZeroLineThicknessF = 0 
res@gsnLeftStringFontHeightF  = 0.023
res@gsnRightStringFontHeightF = 0.023
; res@gsnCenterStringFontHeightF = 0.023 
res@tmXBLabelFontHeightF = 0.018
res@tmYLLabelFontHeightF = 0.018
res@tiMainString        = " "
res@gsnLeftString       = "a) Annual"
res@gsnRightString      = "CRUTEM5"
; res@lbBoxEndCapStyle = "TriangleBothEnds"
; res@lbBoxLineColor   = 0
res@pmLabelBarOrthogonalPosF = 0.18
res@tiYAxisString = " "
res@tiYAxisFontHeightF = 0.025
res@tiYAxisOffsetXF  = -0.025
plot(0) = gsn_csm_contour_map(wks, CRUTEM5_annual_trend, res)

res2 = res
res2@gsnLeftString       = "b)JJA"
res2@gsnRightString      = " "
res2@tiYAxisString = " "
plot(3) = gsn_csm_contour_map(wks, CRUTEM5_JJA_trend, res2)

res3 = res
res3@gsnLeftString         = "c) DJF"
res3@gsnRightString        = " "
res3@tmXBOn                = True
res3@tmXBLabelsOn          = True 
res3@tiYAxisString         = " "
plot(6) = gsn_csm_contour_map(wks, CRUTEM5_DJF_trend, res3)

res4 = res
res4@tmYLLabelsOn = False
res4@tmYLOn       = False
res4@gsnLeftString = "d) Annual"
res4@gsnRightString = "GISTEMP"
res4@tiYAxisString = " "
plot(1) = gsn_csm_contour_map(wks, GISTEMP_annual_trend, res4)

res5 = res4
res5@gsnLeftString       = "e) JJA"
res5@gsnRightString      = " "
res5@tiYAxisString       = " "
plot(4) = gsn_csm_contour_map(wks, GISTEMP_JJA_trend, res5)

res6 = res4
res6@gsnLeftString       = "f) DJF"
res6@gsnRightString      = " "
res6@tiYAxisString = " "
res6@tmXBOn        = True
res6@tmXBLabelsOn  = True
plot(7) = gsn_csm_contour_map(wks, GISTEMP_DJF_trend, res6)


res7 = res
res7@tmYLLabelsOn = False
res7@tmYLOn       = False
res7@gsnLeftString  = "g) Annual"
res7@gsnRightString = "MLOST"
res7@tiYAxisString       = " "
plot(2) = gsn_csm_contour_map(wks, MLOST_annual_trend, res7)

res8 = res7
res8@gsnLeftString       = "h) JJA"
res8@gsnRightString      = " "
res8@tiYAxisString = " "
plot(5) = gsn_csm_contour_map(wks, MLOST_JJA_trend, res8)

res9 = res7
res9@gsnLeftString       = "i) DJF"
res9@gsnRightString      = " "
res9@tiYAxisString= " "
res9@tmXBOn = True
res9@tmXBLabelsOn = True
plot(8) = gsn_csm_contour_map(wks, MLOST_DJF_trend, res9)

pres  = True
pres@gsnMaximize = True
pres@gsnPanelLabelBar  = True
pres@pmLabelBarWidthF                   = 0.65             ; make thinner
; pres@pmLabelBarHeightF                = 0.04
pres@lbLabelFontHeightF                 = 0.012
; pres@lbBoxEndCapStyle                 = "TriangleBothEnds"
; pres@lbBoxLinesOn                     = False
pres@pmLabelBarOrthogonalPosF           = -0.03
pres@lbTitleString                      = "~F22~Obs SAT~F18~(~F18~I~F22~C~F22~ /65 years~F18~)"
pres@lbTitleFontHeightF                 = 0.012
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
gsn_panel(wks, plot, (/3,3/), pres)
end
