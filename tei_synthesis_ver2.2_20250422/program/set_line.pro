;Last Updated: <2025/04/01 11:36:46 from wlan-65-003.mtk.nao.ac.jp by teiakiko>

;================================================
; Define # of spectral lines
;================================================
nl=23                           ;# of the EUVST hot lines in Tab 6.2


;------------------------------------------------
; Choose transition
;------------------------------------------------
case ll of
   0:begin
    ;------------------------------------------------
    ; Ne VIII 770.428 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=10                     ;Ne
      ion=8                     ;VIII
      index=2921
      la_0=770.428*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Ne_VIII_7704'
      W_Z=20.1797d              ; atomic weight of Ne [g/mol]
      min_LIlog=1.0             ;log10 scale [erg/s/cm^2/sr]
      max_LIlog=4.0             ;log10 scale [erg/s/cm^2/sr]
      min_LIlin=3e+1            ;linear scale [erg/s/cm^2/sr]
      max_LIlin=4e+2            ;linear scale [erg/s/cm^2/sr]
   end
   1:begin
    ;------------------------------------------------
    ; Fe IX 171.073 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=9                     ;IX
      index=10124               ;13946 ;ok 20240805
      la_0=171.073*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_IX_1711'
      logT=5.9                  ;log10 scale [K]
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=2.0
      max_LIlog=4.8
      min_LIlin=3e+2
      max_LIlin=5e+3
      min_SI=1.                 ;not yet checked! ;log10 scale [erg/s/cm^2/sr/A]
   end
   2:begin
    ;------------------------------------------------
    ; Fe X 174.531 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=10                    ;X
      index=5100
      la_0=174.531*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_X_1745'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=1.5
      max_LIlog=4.5
      min_LIlin=5e+1
      max_LIlin=4e+3
   end
   3:begin
    ;------------------------------------------------
    ; Mg X 624.962 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=12                     ;Mg
      ion=10                    ;X
      index=2957
      la_0=624.968*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Mg_X_6250'
      W_Z=24.305d               ; atomic weight of Mg [g/mol]
      min_LIlog=1.2
      max_LIlog=4.0
      min_LIlin=1e+1
      max_LIlin=5e+2
   end
   4:begin
    ;------------------------------------------------
    ; Fe XI 180.408 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=11                    ;XI
      index=20245
      la_0=180.401*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XI_1804'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=2.0
      max_LIlog=4.8
      min_LIlin=1e+2
      max_LIlin=3e+3
   end
   5:begin
    ;------------------------------------------------
    ; Fe XII 195.119 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=12                    ;XII
      index=30051               ;30433 ok 20240805            ;transition of Fe XII 195.2A line
      la_0=195.119*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XII_1952'
      logT=6.2                  ;log10 scale [K]
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=alog10(5e2)     ;2.0          ;-6.0            ;log10 scale [erg/s/cm^2/sr]
      max_LIlog=alog10(7e3)     ;4.8          ;-3.2            ;log10 scale [erg/s/cm^2/sr]
      min_LIlin=1e+3            ;8e+1         ;1e-6            ;linear scale [erg/s/cm^2/sr]
      max_LIlin=5e+3            ;2e+3         ;3e-5            ;linear scale [erg/s/cm^2/sr]
      min_SI=10.         ; -6. ; 1.0             ;-7.0               ;log10 scale [erg/s/cm^2/sr/A]
      min_DV=-20         ;[km/s]
      max_DV=+20         ;[km/s]
      min_LW=15          ;[km/s]
      max_LW=+25         ;[km/s]
      min_WN=0           ;[km/s]
      max_WN=+30         ;[km/s]
   end
   6:begin
    ;------------------------------------------------
    ; Fe XII 1241.950 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=12                    ;XII
      index=80987
      la_0=1242.01*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XII_12420'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=0.5
      max_LIlog=2
      min_LIlin=1e+0
      max_LIlin=5e+1
   end
   7:begin
    ;------------------------------------------------
    ; Fe XIII 202.044 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=13                    ;XIII
      index=26520
      la_0=202.044*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XIII_2020'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=1.5
      max_LIlog=5.0
      min_LIlin=5e+0
      max_LIlin=1.5e+3
   end
   8:begin
    ;------------------------------------------------
    ; Si XII 520.666 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=14                     ;Si
      ion=12                    ;XII
      index=3016
      la_0=520.665*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Si_XII_5207'
      W_Z=28.0855d              ; atomic weight of Si [g/mol]
      min_LIlog=1.5
      max_LIlog=4
      min_LIlin=1e+0
      max_LIlin=1e+3
   end
   9:begin
    ;------------------------------------------------
    ; Fe XIV 211.317 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=14                    ;XIV
      index=28720               ;ok 20240805
      la_0=211.317*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XIV_2113'
      logT=6.3                  ;log10 scale [K]
      W_Z=55.8450d         ; atomic weight of Fe [g/mol]
      min_SI=1.            ;10.; 1.0; -7.0 ;log10 scale [erg/s/cm^2/sr/A]
;         min_SI=5.;10.; 1.0             ;-7.0               ;log10 scale [erg/s/cm^2/sr/A]
;        => 5. or 10. might be better for min_SI but 1. looks ok and adopted now.
      min_LIlog=2.5        ;2.0
      max_LIlog=4.0        ;4.8
      min_LIlin=1e+1       ;6e+1
      max_LIlin=1e+4       ;3e+3
      min_DV=-30                ;-30             ;[km/s]
      max_DV=+30                ;+30             ;[km/s]
      min_LW=+15                ;10               ;[km/s]
      max_LW=+25                ;+30             ;[km/s]
      min_WN=0                  ;[km/s]
      max_WN=+30                ;[km/s]
   end
   10:begin
    ;------------------------------------------------
    ; Ca XIV 193.974 in EUVST Tab 6.2 -> should be 197.874 (Warren+11)
    ;------------------------------------------------
      iz=20                     ;Ca
      ion=14                    ;XIV
      index=1306
      la_0=193.866*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Ca_XIV_1939'
      logT=6.5                  ;log10 scale [K]
      W_Z=40.078d               ; atomic weight of Ca [g/mol]
      min_LIlog=alog10(1e0)     ;0.4
      max_LIlog=alog10(3e2)     ;3
      min_LIlin=0.1e0           ;8e+0
      max_LIlin=1.8e2           ;2e+2
      min_SI=1.            ;10.; 1.0             ;-7.0               ;log10 scale [erg/s/cm^2/sr/A]
      min_DV=-30           ;-30             ;[km/s]
      max_DV=+30           ;+30             ;[km/s]
      min_LW=+10           ;10               ;[km/s]
      max_LW=+40           ;+30             ;[km/s]
      min_WN=0             ;[km/s]
      max_WN=+40           ;[km/s]
   end
   11:begin
    ;------------------------------------------------
    ; Ca XIV 943.587 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=20                     ;Ca
      ion=14                    ;XIV
      index=3046
      la_0=943.587*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Ca_XIV_9439'
      W_Z=40.078d               ; atomic weight of Ca [g/mol]
      min_LIlog=-1
      max_LIlog=1.5
      min_LIlin=1e-2
      max_LIlin=5e+0
   end
   12:begin
    ;------------------------------------------------
    ; Ca XV 200.972 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=20                     ;Ca
      ion=15                    ;XV
      index=297
      la_0=200.972*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Ca_XV_2010'
      W_Z=40.078d               ; atomic weight of Ca [g/mol]
      min_LIlog=0
      max_LIlog=3
      min_LIlin=1e+0
      max_LIlin=1e+2
   end
   13:begin
    ;------------------------------------------------
    ; Ca XVI 208.604 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=20                     ;Ca
      ion=16                    ;XVI
      index=4222
      la_0=208.585*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Ca_XVI_2086'
      W_Z=40.078d               ; atomic weight of Ca [g/mol]
      min_LIlog=0
      max_LIlog=3
      min_LIlin=1e+0
      max_LIlin=8e+1
   end
   14:begin
    ;------------------------------------------------
    ; Ca XVII 192.858 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=20                     ;Ca
      ion=17                    ;XVII
      index=2020
      la_0=192.853*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Ca_XVII_1929'
      W_Z=40.078d               ; atomic weight of Ca [g/mol]
      min_LIlog=0
      max_LIlog=3
      min_LIlin=1e+0
      max_LIlin=1e+2
   end
   15:begin
    ;------------------------------------------------
    ; Fe XVIII 974.860 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=18                    ;XVIII
      index=32896
      la_0=974.860*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XVIII_9749'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=0
      max_LIlog=3
      min_LIlin=5e-1
      max_LIlin=1e+2
   end
   16:begin
    ;------------------------------------------------
    ; Fe XIX 592.236 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=19                    ;XIX
      index=26268
      la_0=592.235*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XIX_5922'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=0
      max_LIlog=1.5
      min_LIlin=1e-1
      max_LIlin=1e+1
   end
   17:begin
    ;------------------------------------------------
    ; Fe XIX 1118.07 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=19                    ;XIX
      index=26743
      la_0=1118.06*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XIX_11181'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=0
      max_LIlog=1.5
      min_LIlin=1e-1
      max_LIlin=1e+1
   end
   18:begin
    ;------------------------------------------------
    ; Fe XX 721.559 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=20                    ;XX
      index=43156
      la_0=721.559*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XX_7216'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
      min_LIlog=2.0
      max_LIlog=4.8
      min_LIlin=1e+2
      max_LIlin=3e+3
   end
   19:begin
    ;------------------------------------------------
    ; Fe XXI 786.162 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=21                    ;XXI
      index=18249
      la_0=786.162*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XXI_7862'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
   end
   20:begin
    ;------------------------------------------------
    ; Fe XXII 845.57 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=22                    ;XXII
      index=15201
      la_0=845.571*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XXII_8456'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
   end
   21:begin
    ;------------------------------------------------
    ; Fe XXIII 1079.414 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=23                    ;XXII
      index=6178
      la_0=1079.41*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XXIII_10794'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
   end
   22:begin
    ;------------------------------------------------
    ; Fe XXIV 192.03 in EUVST Tab 6.2
    ;------------------------------------------------
      iz=26                     ;Fe
      ion=24                    ;XXIV
      index=3989
      la_0=192.028*1e-8         ;[A]->[cm] line center wavelength (lambda)
      lstr='Fe_XXIV_1920'
      W_Z=55.8450d              ; atomic weight of Fe [g/mol]
   end
endcase

;================================================
; Define wavelength arr
;================================================
nw=101                          ; # of wavelength points -> odd
vr=600.*1e5                     ; wavelength range in velocity [cm/s] to be ~ve_sig*50
dvar=findgen(nw)/(nw-1)*vr-vr/2 ; wavelength array in velocity [cm/s] => -300 ~ +300 (x 1e5)
dv=dvar[1]-dvar[0]              ; wavelength grid size in velocity [cm/s]
dlar=dvar/ccc*la_0      ; wavelength array in wavelength [cm] => -0.19525407 ~ +0.19525407 (x 1e-8)
dl=dlar[1]-dlar[0]      ; wavelength grid size in wavelength [cm]



print,'A spectral line to synthesize is set.'
