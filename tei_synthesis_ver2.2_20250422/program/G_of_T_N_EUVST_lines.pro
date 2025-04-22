;================================================
; Definitions
;================================================
nl=23                           ;number of spectral lines

nN=27+1                         ;number of density points
logN_max=20.5                   ;higher limit of density range
logN_min=7.0                    ;lower limit of density range

nT=100+1                        ;number of temperature points
logT_max=9.0                    ;higher limit of density range
logT_min=4.0                    ;lower limit of density range

;dir='/Volumes/BUFSSD2T/Data/MURaM/3Dall/G_of_T_N/'
dir='../G_of_T_N/'


;================================================
; Get G(T) w/ CHIANTI
;================================================
; ion balance file
ioneq_file='/home/jm/solar/ssw/packages/chianti/dbase/ioneq/chianti.ioneq'
; abundance file
abund_file='/home/jm/solar/ssw/packages/chianti/dbase/abundance/sun_coronal_2021_chianti.abund'
; Grid Array for Electron Number Density Ne
logNarr=findgen(nN)/(nN-1)*(logN_max-logN_min)+logN_min
; Grid Array for Temperature T
logTarr=findgen(nT)/(nT-1)*(logT_max-logT_min)+logT_min

;--------------------------------------------------------------------------------------------
; plot G(T)
;--------------------------------------------------------------------------------------------
dir_plot=dir+'plot/'
window,1,xs=800,ys=500
window,2,xs=800,ys=500
!p.multi=[0,1,2]
!p.charsize=1.5
loadct,0,/sil


nlfix=5
;================================================
;
;================================================
;for ll=0,nl-1 do begin
;for ll=0,0 do begin
;for ll=1,nl-1 do begin
for ll=nlfix,nlfix do begin
   print,'ll='+sttri(ll)
   ;------------------------------------------------
   ; Choose transition
   ;------------------------------------------------
   case ll of
      0:begin
         ;------------------------------------------------
         ; Ne VIII 770.428 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=10                  ;Ne
         ion=8                  ;VIII
         index=2921
         la_0=770.428*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Ne_VIII_7704'
         max_lin=1e-23
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      1:begin
         ;------------------------------------------------
         ; Fe IX 171.073 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=9                  ;IX
         index=13946
         la_0=171.073*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_IX_1711'
         max_lin=4e-23
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
     end
      2:begin
         ;------------------------------------------------
         ; Fe X 174.531 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=10                 ;X
         index=5100
         la_0=174.531*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_X_1745'
         max_lin=2e-23
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      3:begin
         ;------------------------------------------------
         ; Mg X 624.962 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=12                  ;Mg
         ion=10                 ;X
         index=2957
         la_0=624.968*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Mg_X_6250'
         max_lin=2e-24
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      4:begin
         ;------------------------------------------------
         ; Fe XI 180.408 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=11                 ;XI
         index=20245
         la_0=180.401*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XI_1804'
         max_lin=1e-23
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      5:begin
         ;------------------------------------------------
         ; Fe XII 195.119 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=12                 ;XII
         index=30433            ;transition of Fe XII 195.2A line
         la_0=195.119*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XII_1952'
         max_lin=1e-23
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
;         min_lin=1e-30
;         min_log=-30
;         max_log=-23
      end
      6:begin
         ;------------------------------------------------
         ; Fe XII 1241.950 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=12                 ;XII
         index=80987
         la_0=1242.01*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XII_12420'
         max_lin=2e-25
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      7:begin
         ;------------------------------------------------
         ; Fe XIII 202.044 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=13                 ;XIII
         index=26520
         la_0=202.044*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XIII_2020'
         max_lin=1e-23
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      8:begin
         ;------------------------------------------------
         ; Si XII 520.666 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=14                  ;Si
         ion=12                 ;XII
         index=3016
         la_0=520.665*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Si_XII_5207'
         max_lin=2e-24
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      9:begin
         ;------------------------------------------------
         ; Fe XIV 211.317 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=14                 ;XIV
         index=28720
         la_0=211.317*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XIV_2113'
         max_lin=6e-24
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      10:begin
         ;------------------------------------------------
         ; Ca XIV 193.974 in EUVST Tab 6.2 -> should be 197.874 (Warren+11)
         ;------------------------------------------------
         iz=20                  ;Ca
         ion=14                 ;XIV
         index=1306
         la_0=193.866*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Ca_XIV_1939'
         max_lin=1e-25
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      11:begin
         ;------------------------------------------------
         ; Ca XIV 943.587 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=20                  ;Ca
         ion=14                 ;XIV
         index=3046
         la_0=943.587*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Ca_XIV_9439'
         max_lin=3e-27
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      12:begin
         ;------------------------------------------------
         ; Ca XV 200.972 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=20                  ;Ca
         ion=15                 ;XV
         index=297
         la_0=200.972*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Ca_XV_2010'
         max_lin=8e-26
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      13:begin
         ;------------------------------------------------
         ; Ca XVI 208.604 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=20                  ;Ca
         ion=16                 ;XVI
         index=4222
         la_0=208.585*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Ca_XVI_2086'
         max_lin=6e-26
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      14:begin
         ;------------------------------------------------
         ; Ca XVII 192.858 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=20                  ;Ca
         ion=17                 ;XVII
         index=2020
         la_0=192.853*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Ca_XVII_1929'
         max_lin=2e-25
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      15:begin
         ;------------------------------------------------
         ; Fe XVIII 974.860 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=18                 ;XVIII
         index=32896
         la_0=974.860*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XVIII_9749'
         max_lin=1e-25
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      16:begin
         ;------------------------------------------------
         ; Fe XIX 592.236 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=19                 ;XIX
         index=26268
         la_0=592.235*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XIX_5922'
         max_lin=8e-26
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      17:begin
         ;------------------------------------------------
         ; Fe XIX 1118.07 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=19                 ;XIX
         index=26743
         la_0=1118.06*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XIX_11181'
         max_lin=7e-26
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      18:begin
         ;------------------------------------------------
         ; Fe XX 721.559 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=20                 ;XX
         index=43156
         la_0=721.559*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XX_7216'
         max_lin=7e-26
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      19:begin
         ;------------------------------------------------
         ; Fe XXI 786.162 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=21                 ;XXI
         index=18249
         la_0=786.162*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XXI_7862'
         max_lin=1e-26
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      20:begin
         ;------------------------------------------------
         ; Fe XXII 845.57 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=22                 ;XXII
         index=15201
         la_0=845.571*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XXII_8456'
         max_lin=8e-26
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      21:begin
         ;------------------------------------------------
         ; Fe XXIII 1079.414 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=23                 ;XXII
         index=6178
         la_0=1079.41*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XXIII_10794'
         max_lin=9e-27
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
      22:begin
         ;------------------------------------------------
         ; Fe XXIV 192.03 in EUVST Tab 6.2
         ;------------------------------------------------
         iz=26                  ;Fe
         ion=24                 ;XXIV
         index=3989
         la_0=192.028*1e-8      ;[A]->[cm] line center wavelength (lambda)
         lstr='Fe_XXIV_1920'
         max_lin=9e-25
         min_lin=max_lin*1e-10
         max_log=alog10(max_lin)
         min_log=alog10(min_lin)
      end
   endcase

   goto,here
   ;--------------------------------------------------------------------
   ; G(T) = 0.83 * Ab * delta_E * n_j * F(T) / N_e [erg /s cm^3]
   ;--------------------------------------------------------------------
   G_of_T_N=dblarr(nT,nN)
   for dd=0,nN-1 do begin & pp,dd
      G_of_T_N[*,dd]=g_of_t(iz,ion,index=index,ioneq_file=ioneq_file,abund_file=abund_file,dens=logNarr[dd])
   endfor

   save,f=dir+'G_of_T_N_'+lstr+'.sav',G_of_T_N,logTarr,logNarr

;   stop

   here:
   restore,dir+'G_of_T_N_'+lstr+'.sav',/verbose

   ;--------------------------------------------------------------------------------------------
   ;
   ;--------------------------------------------------------------------------------------------
;   min_lin=1e-30
;   max_lin=1e-23
;   min_log=-30
;   max_log=-23

   ;--------------------------------------------------------------------------------------------
   ; plot G(T)
   ;--------------------------------------------------------------------------------------------
   ;--------------------------------------------------------------------------------------------
   ; G(T) in linear scale
   wset,1
   plot_image,G_of_T_N,$
              xst=1,yst=1,ori=[4,7],sca=[0.05,0.5],$
              min=min_lin,max=max_lin,$
              xtit='Log!D10!N ( T [K] )',ytit='Log!D10!N ( N!De!N [cm!U-3!N] )',$
              pos=[0.1,0.11,0.95,0.81]
   plot_image,[[indgen(256)],[indgen(256)]],$
              xst=1,yst=1,ori=[min_lin,0],sca=[(max_lin-min_lin)/255.,1],$
;              xtickin=2e-24,$
              yticklen=1e-5,ytickna=replicate(' ',50),$
              xtit='G = 0.83 * Ab * '+gr('Delta')+'E * n!Dj!N * F(T) / N!De!N [erg s!U-1!N cm!U3!N]',$
              pos=[0.1,0.92,0.95,0.97]
   write_png,dir_plot+'G_of_T_N_'+lstr+'_lin.png',tvrd(/true)
   ;--------------------------------------------------------------------------------------------
   ; G(T) in log scale
   wset,2
   plot_image,alog10(G_of_T_N),$
              xst=1,yst=1,ori=[4,7],sca=[0.05,0.5],$
              min=min_log,max=max_log,$
              xtit='Log!D10!N ( T [K] )',ytit='Log!D10!N ( N!De!N [cm!U-3!N] )',$
              pos=[0.1,0.11,0.95,0.81]
   plot_image,[[indgen(256)],[indgen(256)]],$
              xst=1,yst=1,ori=[min_log,0],sca=[(max_log-min_log)/255.,1],$
;              xtickin=100,$
              yticklen=1e-5,ytickna=replicate(' ',50),$
              xtit='Log!D10!N ( G [erg s!U-1!N cm!U3!N] )',$
              pos=[0.1,0.92,0.95,0.97]
   write_png,dir_plot+'G_of_T_N_'+lstr+'_log.png',tvrd(/true)
   ;--------------------------------------------------------------------------------------------

endfor

!p.multi=0

; NOTE:
;    Log10(density)=10 is assumed to calcurate emissivities (default for keyward "dens").
;    Background radiation temperature is 6000 K (default for keyword "radtemp.")
;    No photoexcitation (default for keyword "rphot").
;    Proton rates are used (by default).
;    Delta-E is included (by default).
end
