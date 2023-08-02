load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

begin

yrStrt = 195001
yrLast = 201412

season1 = "DJF" 
season2 = "JJA"

dir_model = "/work/mh0033/m301036/josie/LSAT/Data/"

model_file = addfile(dir_model+"MPI-ESM1-LR-30run-tas.nc","r")
time = model_file->time

date_time = cd_calendar(time, -1)
; print(date_time)
iStr = ind(date_time.eq.yrStrt)
iEnd = ind(date_time.eq.yrLast)
tas_multi = model_file->tas(:, iStr:iEnd, {0:90},:)

; Define the calendar type and reference date
calendar = "proleptic_gregorian"
ref_date = "1850-01-01"

; Convert the time to the Gregorian calendar
tas_multi&time = cd_convert(tas_multi&time, "days since 1850-01-01")

tas_multi = lonFlip(tas_multi)
;Convert the units of the tas
tas_multi = tas_multi -273.15 
tas_multi@units = "degC"
printVarSummary(tas_multi)
printMinMax(tas_multi, 0)
exit

dims = dimsizes(tas_multi)
print(dims)

tas_clma = new((/dims(0), 12, dims(2), dims(3)/), typeof(tas_multi))
tas_ano = new((/dims(0), dims(1), dims(2), dims(3)/), typeof(tas_multi))
tas_ano@_FillValue = 1e+20
do i = 0, 29
    tas_clma(i,:,:,:) = clmMonTLL(tas_multi(i,:,:,:))
    tas_ano(i,:,:,:) = calcMonAnomTLL(tas_multi(i,:,:,:),tas_clma(i,:,:,:))
end do
copy_VarMeta(tas_multi(:,0,0,0), tas_ano(:,0,0,0))
printVarSummary(tas_ano)
printMinMax(tas_ano,0)
; exit

;Claculate the annual and mothly mean for each run

nyears = dims(1)/12  ; Assuming monthly data
print(nyears)
; exit
tas_mutli_annual_mean = new((/dims(0), nyears, dims(2), dims(3)/), typeof(tas_ano))

do iy = 0, nyears-1
    tas_mutli_annual_mean(:,iy,:,:) = dim_avg_n_Wrap(tas_ano(:,iy*12:iy*12+11,:,:), 1)  ; Calculate mean along the time dimension (0)
end do

tas_mutli_annual_mean!1 = "year"
tas_mutli_annual_mean@year = ispan(1950,2014,1)
; tas_mutli_annual = month_to_annual(tas_ano, 1)
printVarSummary(tas_mutli_annual_mean)
printMinMax(tas_mutli_annual_mean, 0)

tas_mutli_JJA = new((/nyears, dims(0), dims(2), dims(3)/), typeof(tas_ano))
; copy_VarMeta(tas_mutli_annual_mean, tas_mutli_JJA)
j = 0
do i =0, 29
    tas_mutli_JJA(:,j,:,:)= month_to_season(tas_ano(time|:, run|i, lat|:, lon|:), season1)
    j = j+1
end do
tas_mutli_JJA!1 = "run"
tas_mutli_JJA&run = ispan(1,30,1)
printVarSummary(tas_mutli_JJA)
printMinMax(tas_mutli_JJA, 0)

tas_mutli_DJF = new((/nyears, dims(0),dims(2), dims(3)/), typeof(tas_ano))
; copy_VarMeta(tas_mutli_annual_mean, tas_mutli_DJF)
k = 0
do i =0, 29
    tas_mutli_DJF(:,k,:,:)= month_to_season(tas_ano(time|:, run|i, lat|:, lon|:), season2)
    k = k+1
end do
tas_mutli_DJF!1 = "run"
tas_mutli_DJF&run = ispan(1,30,1)
printVarSummary(tas_mutli_DJF)
printMinMax(tas_mutli_DJF, 0)
;=========================Calculate the trend spatial pattern for each run=======================
tas_annual_trend = new((/dims(0),dims(2), dims(3)/), float)
copy_VarMeta(tas_mutli_annual_mean(:,0,:,:), tas_annual_trend)

tas_JJA_trend = new((/dims(0),dims(2), dims(3)/), float)
copy_VarMeta(tas_mutli_JJA(:,0,:,:), tas_JJA_trend)

tas_DJF_trend = new((/dims(0),dims(2), dims(3)/), float)
copy_VarMeta(tas_mutli_DJF(:,0,:,:), tas_DJF_trend)

x = ispan(1950, 2014, 1)

do i = 0, 29
    print("run number = "+i)
    tas_annual_trend(i,:,:) = regCoef(x, tas_mutli_annual_mean(run|i, lat|:, lon|:, year|:))
    tas_JJA_trend(i,:,:) = regCoef(x, tas_mutli_JJA(run|i, lat|:, lon|:, time|:))
    tas_DJF_trend(i,:,:) = regCoef(x, tas_mutli_DJF(run|i, lat|:, lon|:, time|:))
end do


tas_annual_trend = tas_annual_trend*15
tas_JJA_trend = tas_JJA_trend*15
tas_DJF_trend = tas_DJF_trend*15
; copy_VarMeta(tas_mutli_annual_mean(:,0,:,:), tas_annual_trend)
; copy_VarMeta(tas_mutli_annual_mean(:,0,:,:), tas_JJA_trend)
; copy_VarMeta(tas_mutli_annual_mean(:,0,:,:), tas_DJF_trend)

printVarSummary(tas_annual_trend)

end
