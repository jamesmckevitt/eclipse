; hgcr_muram2em.pro
;
; Project: Heliophysics Grand Challenges Research
;          Physics and Diagnostics of the Drivers of Solar Eruptions
; Author: Mark Cheung
; Revision history: 2016-01-16: v0.8
; Purpose: Read in corona_emission_adj_dem_<dir>.XXXXXX cubes output from
;          MURaM simulations and convert to Emisssion Measure (EM) cubes
; Usage: IDL> e = hgcr_muram2em(340000l, dir='y')
;        IDL> e = hgcr_muram2em(340000l, dir='z')
; Output: an idl structure
; IDL> help, e, /str
;** Structure <2d21768>, 7 tags, length=33030424, data length=33030420, refs=1:
;   T6L             FLOAT     Array[21] ; left edge of temperature bin
;   T6R             FLOAT     Array[21] ; right edge of temperature bin
;   T6M             FLOAT     Array[21] ; center of temperature bin
;   EMCUBE          FLOAT     Array[512, 768, 21] ; EM in a temperature bin, units of cm^-5
;   TIME            FLOAT           32310.0 ; Time of simulation snapshot
;   NMOD            LONG            340000  ; Snapshot number
;   DIR             STRING    'z'     ; Viewing direction 

function hgcr_muram2em2, nmod, dir=dir, time=time, lgt=lgt, t6l=t6l, t6r=t6r, t6m=t6m, resample=resample, offset=offset, vlos=vlos
  if n_elements(dir) EQ 0 then dir = 'y'
  if n_elements(resample) EQ 0 then resample = 1

  if dir EQ 'y' then print, "Note: dir='y' is the vertical direction"
  case dir of
     ;'y': L = 49.152e8/768.         ; vertical
     ;'x': L = 98.304e8/512.
     ;'z': L = 49.152e8/256.
     'y': L = 1.         ; vertical
     'x': L = 1.
     'z': L = 1.
     else: print, "Please give dir='x', 'y' or 'z'"
  endcase

  common roi, i0, i1, j0, j1
  if n_elements(i0) eq 0 then i0 = 0
  if n_elements(j0) eq 0 then j0 = 0
  
  nmodstr = string(nmod,format="(I07)")
  openr,u, "./2D/corona_emission_adj_dem_"+dir+"_every200steps/corona_emission_adj_dem_"+dir+'.'+nmodstr, /get_lun
  a=assoc(u,fltarr(6))
  header=a(0)
  ntemp = round(header[0])
  nx = round(header[1])
  ny = round(header[2])
  time=header[3]
  leftT = header[4]
  dlgT = header[5]
  i1 = nx-1
  j1 = ny-1
  a=assoc(u,fltarr(nx,ny,ntemp),6*4)
  if n_elements(offset) EQ 0 then offset = 0
  emcube = a(offset)
  close, u
  free_lun, u
  emcube = emcube[i0:i1,j0:j1,*]

  If KEYWORD_SET(vlos) THEN BEGIN
     openr,u, "./2D/corona_emission_adj_vlos_"+dir+'.'+nmodstr, /get_lun
     a=assoc(u,fltarr(nx,ny,ntemp),6*4)
     vlos = a(offset)
     close, u
     free_lun, u
     vlos = vlos[i0:i1,j0:j1,*]
  ENDIF

  ;t6l = [0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0,10.0,12.5,15.0,20.0,25.0,30.0,35.0,40.0,45.0]
  ;t6r = [t6l[1:*],50.0]
  ;dlgT = 0.1
  ;tempgrid = read_ascii('Temperature_Bins.log')
  ;ntemp = tempgrid.field01[0]-1
  ;t6l = 10.^tempgrid.field01[1:1+ntemp-1]/1e6
  ;t6r = 10.^tempgrid.field01[2:2+ntemp-1]/1e6

  t6l = (10.^(findgen(ntemp)*dlgT + leftT))/1e6
  t6r = (10.^((findgen(ntemp)+1)*dlgT + leftT))/1e6
  
  ;t6r = t6l + dlgT  
  t6m = 0.5*(t6l+t6r)
  
  ; Cumulative EM distribution, used for resampling later
  emcumu = [ [[fltarr(nx,ny)]],[[total(emcube,3,/cumu)]]]
  
  ; Resample EM into new temperature bins
  IF KEYWORD_SET(resample) EQ 1 THEN BEGIN
     
     IF N_ELEMENTS(lgt) EQ 0 THEN BEGIN
        dlgt=0.1
        lgt = 5.5+findgen(21)*dlgt
     ENDIF ELSE BEGIN
        dlgt = lgt[1]-lgt[0]
     ENDELSE
     lgtl= lgt-0.5*dlgt
     lgtr= lgt+0.5*dlgt
     emnew = fltarr(nx,ny,n_elements(lgt))
     vlosnew = fltarr(nx,ny,n_elements(lgt))

     for n=0,n_elements(lgt)-1 do begin
        lgtl_ind = max(where(t6l LE (10.^lgtl[n])/1e6)); > 0
        lgtr_ind = max(where(t6l LE (10.^lgtr[n])/1e6)); > 0
        ;print, alog10(1e6*t6l[lgtl_ind]), lgtl[n], alog10(1e6*t6l[lgtr_ind]), lgtr[n]
        ;print, lgtl_ind, lgtr_ind
        IF (lgtl_ind GE 0) AND (lgtl_ind LT n_elements(t6l)-1) THEN BEGIN
           mu_l = (10.^lgtl[n]*1e-6 - t6l[lgtl_ind])/(t6l[lgtl_ind+1] - t6l[lgtl_ind])
           mu_l = (lgtl[n] - alog10(t6l[lgtl_ind]*1e6))/(alog10(t6l[lgtl_ind+1])-alog10(t6l[lgtl_ind]))
           ;print, alog10(t6l[lgtl_ind]*1e6), lgtl[n], alog10(t6l[lgtr_ind]*1e6), lgtr[n], mu_l
           cumu_em_l = emcumu[*,*,lgtl_ind]*(1.0-mu_l)+emcumu[*,*,lgtl_ind+1]*mu_l
           IF KEYWORD_SET(vlos) THEN vlosnew[*,*,n] = vlos[*,*,lgtl_ind]*(1.0-mu_l)+vlos[*,*,lgtl_ind+1]*mu_l
           
        ENDIF ELSE BEGIN
           cumu_em_l = fltarr(nx,ny)
           IF KEYWORD_SET(vlos) THEN vlosnew[*,*,n] = 0.0
        ENDELSE

        IF (lgtr_ind GE 0) AND (lgtr_ind LT n_elements(t6l)-1) THEN BEGIN
           mu_r = (10.^lgtr[n]*1e-6 - t6l[lgtr_ind])/(t6l[lgtr_ind+1] - t6l[lgtr_ind])
           mu_r = (lgtr[n] - alog10(t6l[lgtr_ind]*1e6))/(alog10(t6l[lgtr_ind+1])-alog10(t6l[lgtr_ind]))
           cumu_em_r = emcumu[*,*,lgtr_ind]*(1.0-mu_r)+emcumu[*,*,lgtr_ind+1]*mu_r
        ENDIF ELSE BEGIN
           cumu_em_r = fltarr(nx,ny)
        ENDELSE
        emnew[*,*,n] = (cumu_em_r-cumu_em_l)> 0
        
        if min(emnew[*,*,n] LT 0.0) then stop
     endfor
     t6l = 10.^lgtl/1e6
     t6r = 10.^lgtr/1e6
     t6m = 10.^lgt/1e6
     ;print, total(emcube), total(emnew)
     emcube = emnew
     vlos = vlosnew
  ENDIF

  ; Mean molecular weight for fully-ionized solar gas mixture
  mu = 0.62
  m_h = 1.6733e-24
  ;ne_factor = 1./(mu*m_h*1e14)^2 ; original
                                ;ne_factor = 0.86/(m_h*1e14)^2

  X = 0.73                      ;
  ;ne_fact = (1.0-X)/4.*(X)
  ne_factor = 0.5*X*(1+X)/(m_h*1e14)^2.0 ; This converts rho^2 to n_e n_H
  

  ;print, ne_factor,'=ne_factor'
  ;print, mean(emcube*L*ne_factor)
  IF NOT KEYWORD_SET(vlos) THEN  return, {t6l:t6l, t6r:t6r, t6m:t6m, emcube:emcube*L*ne_factor, $
                                          time:time, nmod:nmod, dir:dir}
  return, {t6l:t6l, t6r:t6r, t6m:t6m, emcube:emcube*L*ne_factor, vlos:(-vlos), $  ; flip sign of velocity
           time:time, nmod:nmod, dir:dir}
end
