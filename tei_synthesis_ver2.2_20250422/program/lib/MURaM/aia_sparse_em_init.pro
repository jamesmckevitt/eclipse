; Given a set of AIA EUV observations (excluding 304), find sparse EM
; solution by using IDL's simplex solver for linear programming.
;
; INPUT: 	image	--	; [nx,ny,6] stack of aligned images [DN/pixel/s, with the
; third dimension ordered by in AIA channel following this order: 
; [94, 131, 171, 193, 211, 335]
;
; Example usage: 
; First IDL> .compile aia_sparse_em
;       IDL> aia_sparse_em_init, use_lgtaxis = findgen(21) + 5.5, timedepend = '2011-02-12T00:00'
;       IDL> aia_sparse_em_solve, image, oem=em, status=status
; 
; Beware!!!! By default all EM are in units of 10^26 cm^{-5}.
; Given an EM solution, recontruct AIA EUV images
;       IDL>  aia_sparse_em_em2image, em, image=rimage

; To given an arbitrary axis in lgt, call aia_sparse_em_init with the
; use_lgtaxis parameter

; To add an XRT channel, set the xrt parameter in aia_sparse_em_init
; with a string for the XRT channel name.

; WRITTEN: Mark Cheung, first version 2014/01/13
;
; Revision History
; 2014-06-11 V0.6 Added /noblend parameter for AIA_SPARSE_EM_INIT
; 2014-09-30 V0.7 Added routine for calculating zeroth, first and second
;            moments of EM solution
;            Added routine for calculating synthetic EIS rasters. 
;            SIMPLEX very occasionally returns negative coeffs even 
;            though positivity is a constraint. Now there are set to
;            zero and the corresponding status code is 10.
; 2015-10-05 V0.8 Fixed typo in common block declaration in
;aia_sparse_em_init, specifically spelling of chiantifix in common
;block. Does not affect inversion results chiantifix is not used
;elsewhere after initialization.
; 2015-10-19 V0.9 Cleaned up some unused, old commented code.
; 2016-03-25 V0.91 Minor changes to aia_sparse_dem_inspect
; 2017-06-06 V0.95 Implemented adaptive tolfac. tolmap now returned to
;                  indicate which tolfac values were used for which pixels.

; Do summing 
function aia_summing, img, sx, sy
; Given a 2D array 

  imgdim = size(img,/dim)
  if n_elements(imgdim) NE 2 then return, -1
  if imgdim[0] MOD sx NE 0 then return, -1
  if imgdim[1] MOD sy NE 0 then return, -1
  
  resdim = imgdim/[sx,sy]
  
  result = make_array(resdim, type=size(img,/type))
  
  xstride = indgen(resdim[0])*sx
  ystride = indgen(resdim[1])*sy
  indx = indgen(resdim[0])
  indy = indgen(resdim[1])
  
  for k=0, resdim[1]-1 do for i=0,sx-1 do for j=0,sy-1 do $
     result[indx,k]+= img[xstride+i,ystride[k]+j]
  return, result  
end

FUNCTION aia_sparse_binup, a, sx
  return, aia_summing(reform(a,n_elements(a),1),sx,1)
end

FUNCTION aia_sparse_em_basis_funcs
  COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs

  return, basis_funcs
END

FUNCTION AIA_SPARSE_EM_UNITS
   COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs

   return, Dictfac
END

pro aia_sparse_dem_inspect, coeff, emcube, status, img=img, ylog=ylog
   COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs
   dim=size(coeff,/dim)
   window, 0,  xs=dim[0], ys=dim[1]
   tv, bytscl(alog10(total(emcube,3)*aia_sparse_em_units()),min=26,max=29)
   xyouts, 3, 3, 'Right click to end', /device, charsize=2, charthick=2
   print, 'Showing total EM'
   oldXMargin = !X.margin
   oldYMargin = !Y.margin
   oldMulti = !P.multi
   
   !P.multi=0
   !P.charsize=1
   window, 1, xs=800, ys=600
   !MOUSE.button =0 
   lgtindex = 0
   key = ''
   while (!MOUSE.button NE 4) do begin
      wset, 0
      ;tv, bytscl(alog10(emcube[*,*,lgtindex]*aia_sparse_em_units()),min=26,max=29)
      ;xyouts, 3, 3, 'Right click to end', /device, charsize=2, charthick=2
      ;read, key
      ;case key of 
      ;   '=': if (lgtindex LT n_elements(lgtaxis)-2) then lgtindex += 1
      ;   '-': if (lgtindex GE 1) then lgtindex -= 1
      ;endcase
      cursor, x, y, /device, /change
      wset, 1
      if status[x,y] EQ 0 then aia_sparse_em_plot_components, reform(Coeff[x,y,*]), ylog=ylog, $
         title=strtrim(string(x),2)+','+strtrim(string(y),2) + ' Total(EM) =' +string(total(emcube[x,y,*]), format='(G14.6)')
      ;if FIX(PRODUCT(size(img,/dim) [dim[0],dim[0],6])) EQ 1 THEN
   endwhile
   
end

; Plot components of inversion
pro aia_sparse_em_plot_components, coeff, ylog=ylog, title=title
   COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs
   lzero = total(coeff GT 0)
   colors = (findgen(lzero)+1)*(240.)/float(lzero)
   emcube = coeff#basis_funcs
   loadct, 0, /silent
   plot, lgtaxis, emcube, thick=4, /xstyle, /ystyle, psym=10, $
         ytitle='EM ['+string(aia_sparse_em_units(),format='(G8.2)')+' cm^-5]', charsize=2.4, $
         xtitle='Log T/K',yrange=minmax(emcube) > 1e-1, ylog=ylog, title=title
   oplot, lgtaxis, emcube, thick=2, color=1, line=2, psym=10
   ncoeff = n_elements(coeff)
   counter = 0
   loadct, 22, /silent
   for n=0,ncoeff-1 do begin
      if coeff[n] GT 0 then begin
         oplot, lgtaxis, coeff[n]*basis_funcs[n,*], thick=2, color=colors[counter]
         counter+=1
      endif
   endfor
end




FUNCTION AIA_SPARSE_EM2XRT, em, status, xrt_resp=xrt_resp, $
                        date=date, channel=channel, differential=differential

   COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs

; Set differential = 1 if suppling EM as input. Otherwise the code
; assumes EMs are given.

IF( N_ELEMENTS(xrt_resp) EQ 0) THEN BEGIN

   ; Generate temperature response function for lgtaxis
   modelname = ''
   wave = aia_resp.total.wave
   temp = 10.^(aia_resp.total.logte)
   spectrum = aia_resp.total.emissivity
   abund_model = aia_resp.general.abundfile
   ioneq_model = aia_resp.general.ioneq_name
   dens_model = aia_resp.general.model_name +', p='+trim(aia_resp.general.model_pe, 1)   
   emiss_model = make_xrt_emiss_model(modelname, wave, temp, spectrum, $
                                       abund_model, ioneq_model,dens_model,$
                                       data_files=data_files)

   wresps_all = make_xrt_wave_resp(contam_time=date)
   xrt_resp = make_xrt_temp_resp(wresps_all, emiss_model)
   ;IF n_elements(date) EQ 0 THEN date = anytim('2012-Jan-16 05:50')
   ;wave_resp = make_xrt_wave_resp(contam_time=anytim(date,/ccsds))
   ;xrt_resp = make_xrt_temp_resp(wave_resp, /apec_default)
ENDIF

if N_ELEMENTS(channel) EQ 0 then channel = 'unspecified'
case channel of 
   'Al-mesh': c = 0
   'Al-poly': c = 1
   'C-poly': c = 2
   'Ti-poly': c = 3
   'Be-thin': c = 4
   'Be-med': c = 5
   'Al-med': c = 6
   'Al-thick': c = 7
   'Be-thick': c = 8
   'Al-poly/Al-mesh': c = 9
   'Al-poly/Ti-poly': c = 10
   'Al-poly/Al-thick': c = 11
   'Al-poly/Be-thick': c = 12
   'C-poly/Ti-poly': c = 13
   'C-poly/Al-thick': c = 14
   else : begin
      print, 'XRT:'+channel +': not found'
      return, -1
   endcase
endcase

; channel number
this_xrt_channel = xrt_resp[c]
print, this_xrt_channel

; Interpolate response function onto lgtaxis (for which we have
; the EM)

good = where(this_xrt_channel.temp_resp NE 0)
xtemp_resp = this_xrt_channel.temp_resp[good]
xtemp_coord= this_xrt_channel.temp[good]

ntemp = n_elements(lgtaxis)
this_resp = (interpol((xtemp_resp),alog10(xtemp_coord), lgtaxis))
;plot, xtemp_coord, xtemp_resp, /xlog, /ylog & oplot, 10^lgtaxis, this_resp, psym=4
;stop

dim = size(em,/dim)
xrt_img = fltarr(dim[0],dim[1])

for i=0,dim[0]-1 do for j=0,dim[1]-1 do xrt_img[i,j] = total(em[i,j,0:ntemp-1]*this_resp)
   

return, xrt_img*(status EQ 0)

end

PRO AIA_SPARSE_EM_EM2Moments, em, status, emsum=emsum, emwlgt=emwlgt, emwidth=emwidth
  ; Calculate 0th, 1st and 2nd moments of EM distribution
  COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs

  dim = size(em,/dim)

  ; Compute 0th moment
  emsum = fltarr(dim[0],dim[1])
  emsum = total(em, 3)
  bad = where(status NE 0)
  IF bad[0] NE -1 THEN emsum[bad] = 0.0

  ; Compute 1st moment if either 1st or 2nd request
  ;IF ( KEYWORD_SET(emwlgt) OR keyword_set(emwidth) ) THEN BEGIN
     lgtaxiscube = fltarr(dim[0],dim[1],dim[2])
     FOR i=0l,dim[0]-1 DO for j=0l,dim[1]-1 DO lgtaxiscube[i,j,0:dim[2]-1] = lgtaxis
     emwlgT = total(lgtaxiscube*em,3)/emsum
     IF bad[0] NE -1 THEN emwlgT[bad] = 0.0
  ;ENDIF

  ; Compute 2nd moment if requested
  ;IF keyword_set(emwidth) THEN BEGIN
     FOR n=0,dim[2]-1 DO lgtaxiscube[*,*,n] -= emwlgt[*,*]
     emwidth = sqrt(total(lgtaxiscube*lgtaxiscube*em,3)/emsum)
     IF bad[0] NE -1 THEN emwidth[bad] = 0.0
  ;ENDIF

END


PRO AIA_SPARSE_EM_EM2IMAGE, em, image=image, status=status
  COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs

  dim = size(em,/dim)
  ntemp = n_elements(lgtaxis)
  image = fltarr(dim[0],dim[1],nchannels)
  for i=0,dim[0]-1 do begin
     for j=0,dim[1]-1 do begin
        image[i,j,*]  = Dict[*,0:ntemp-1] # reform(em[i,j,*]>0)        
     endfor
  endfor
END


; Example call for aia_sparse_em_em2lines
;aia_sparse_em_em2lines, em, image=eisimage, status=status, lines=lines, $
;                        eislines=['Mg V 276.570','Mg VI 270.394','Mg VII 280.737', 'Si VII 275.368',$
;                                  'Fe IX 197.862', 'Fe X 184.536', 'Fe XI 180.401', 'Fe XII 195.119',$
;                                  'Fe XIII 202.044', 'Fe XIII 203.826', 'Fe XIV 264.787', 'Fe XV 284.160',$
;                                  'Fe XVI 262.984', 'Ca XIV 193.874', 'Ca XV 200.972', 'Ca XVI 208.604', $
;                                  'Fe XVII 254.87', 'Ca XVII 192.858']

PRO AIA_SPARSE_EM_EM2LINES, em, image=image, status=status, snote=snote, wvlrange=wvlrange, eislines=eislines, lines=lines

  COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs

  ;lines = [147398l,90588l,68481l]
  IF n_elements(eislines) NE 0 THEN BEGIN
     lines = lonarr(n_elements(eislines))
     FOR l=0,n_elements(eislines)-1 do begin
        snote = strsplit(eislines[l],' ',/extract)
        wvl = float(snote[2])
        snote = strjoin(snote[0:1],' ')
        wvlrange=wvl+[-0.2,0.2]
        line = where((strmatch(aia_resp.lines.snote,snote) EQ 1) AND $
                   (aia_resp.lines.wvl GE wvlrange[0]) AND $
                   (aia_resp.lines.wvl LE wvlrange[1]))
        line = line[ sort(abs(aia_resp.lines[line].wvl-wvl))]
        if line[0] NE -1 then lines[l] = line[0]
     ENDFOR
  ENDIF

  ;if n_elements(lines) EQ 0 then begin
  ;   snote = strtrim(snote,2)
  ;   lines = where((strmatch(aia_resp.lines.snote,snote) EQ 1) AND $
  ;                 (aia_resp.lines.wvl GE wvlrange[0]) AND $
  ;                 (aia_resp.lines.wvl LE wvlrange[1]))
  ;endif

  IF (N_ELEMENTS(lines) GT 50) THEN BEGIN 
     PRINT, 'Too many lines selected for '+snote+'! Maximum 50 allowed.' 
     return
  ENDIF

  IF (lines[0] NE -1) then BEGIN
     lines = (aia_resp.lines)[lines]
  ENDIF ELSE BEGIN
     print, 'No lines found for '+snote+' in '+string(wvlrange)
     return
  ENDELSE
  nchannels = n_elements(lines)

  dim = size(em,/dim)
  ntemp = n_elements(lgtaxis)
  image = fltarr(dim[0],dim[1],nchannels)
  line_resp = fltarr(nchannels, ntemp)

  for l=0,nchannels-1 do begin
     line_resp[l,0:ntemp-1] = interpol(lines[l].goft, aia_resp.total.logte, lgtaxis)
  endfor
  
  for i=0,dim[0]-1 do begin
     for j=0,dim[1]-1 do begin
        image[i,j,*]  = line_resp[0:nchannels-1,0:ntemp-1] # reform(em[i,j,0:ntemp-1]*AIA_SPARSE_EM_UNITS() > 0.0)        
     endfor
  endfor
END

PRO AIA_SPARSE_EM_SOLVE, image, oem=oem, lgt=lgt, zmax=zmax, status=status, relax=relax, eps=eps, tolfac=tolfac, $
                         zfac=zfac, exptimes=exptimes, symmbuff=symmbuff, tolfunc=tolfunc, coeffs=coeffs, tolmap=tolmap, $
                         adaptive_tolfac=adaptive_tolfac

  start = anytim(systim())
                         
  IF N_ELEMENTS(eps) EQ 0 THEN eps = 1e-3
  If N_ELEMENTS(tolfac) EQ 0 THEN tolfac = 1.4
  If N_ELEMENTS(relax) EQ 0 THEN relax = 1
  IF N_ELEMENTS(symmbuff) EQ 0 THEN symmbuff = 1.0
  IF N_ELEMENTS(tolfunc) EQ 0 THEN tolfunc='sqrt(y)'
  IF N_ELEMENTS(adaptive_tolfac) EQ 0 then adaptive_tolfac = 1 

  COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs
  IF N_ELEMENTS(Dict) EQ 0 THEN BEGIN
     print, 'First run AIA_SPARSE_EM_INIT'
  ENDIF

  print, 'AIA_SPARSE_EM_SOLVE: lgt=',string(lgtaxis,format='(F5.2)')

  ; Image must have first and second dimensions as spatial dimensions.
                                ; Third dimension is AIA channel in
                                ; order [94, 131, 171, 193, 211, 335]
  dim = size(image,/dim)
  NOCOUNTS = WHERE(total(image,3) LT 10*eps)

  ntemp   = n_elements(Dict[0,*])
  coeffs  = fltarr(dim[0],dim[1],ntemp)
  zmax    = fltarr(dim[0],dim[1])
  status  = fltarr(dim[0],dim[1])
  tolmap  = fltarr(dim[0],dim[1])
  
  ; Objective function is the negative of the L1-norm of the EM
  if (n_elements(zfac) EQ ntemp) then begin
     zequation = -zfac
  endif else begin
     zequation = fltarr(ntemp)
     zequation[*] = -1.0
  endelse

  tolcmd = 'tol = tolfac*'+tolfunc
  print, tolcmd

  ; Put response matrix in constraint matrix
  IF keyword_set(relax) THEN BEGIN
                                ; This version gives some room for error 
     m1 = nchannels             ; Ax <= y + tol constraints
     m2 = nchannels             ; Ax >= y - tol constraints
     m3 = 0                     ; Ax  = y       constraints (not used for relaxed option)      
     m  = m1+m2+m3
  ENDIF
  constraint = fltarr(m,ntemp+1)
  constraint[0:m1-1,1:ntemp]    = -Dict
  constraint[m1:m1+m2-1,1:ntemp]= -Dict


  for i=0l,dim[0]-1 do begin
     for j=0l,dim[1]-1 do begin
        y = reform(image[i,j,0:nchannels-1]) > 0
        
        IF keyword_set(relax) THEN BEGIN
           ; This version gives some room for error 
           ;m1 = nchannels               ; Ax <= y + tol constraints
           ;m2 = nchannels               ; Ax >= y - tol constraints
           ;m3 = 0               ; Ax  = y       constraints (not used for relaxed option)
           ;m  = m1+m2+m3

           res = execute(tolcmd)
           ;tol = tolfac*sqrt(y) ; Use Sqrt(AIA DN/s as tolerance)
                                ;tol[0] *= 2
                                ;tol = ([0.2,4.3,320,170,38,4])
           ;constraint = fltarr(m,ntemp+1)
           constraint[0:m1-1,0]          = y + tol ; m1 constraints
           ;constraint[0:m1-1,1:ntemp]    = -Dict
           ; If exptime times given
           if (n_elements(exptimes) EQ nchannels) then for c=0,nchannels-1 do constraint[0+c,1:ntemp] /= exptimes[c]
           constraint[m1:m1+m2-1,0]      =(y - symmbuff*tol) > 0.0 ; m2 constraints
           ;constraint[m1:m1+m2-1,1:ntemp]= -Dict
           ; If exptime times given
           if (n_elements(exptimes) EQ nchannels) then for c=0,nchannels-1 do constraint[m1+c,1:ntemp] /= exptimes[c]           
           r = simplex(zequation, transpose(constraint), m1, m2, m3, status=s,eps=eps*max(y)*8e-4) ; 8e-4 has been fine-tuned from trial and error to get max #solutions
           ;r = simplex(zequation, transpose(constraint), m1, m2, m3, status=s,eps=eps*(y[5]>0.5)*0.10) 
        ENDIF ELSE BEGIN
           ; This version enforces equality Ax = y
           constraint = [[reform(y,nchannels,1)],[-Dict]]
           r = simplex(zequation, transpose(constraint), 0, 0, nchannels,status=s)
     ENDELSE 
        
        if (min(r[1:ntemp]) LT 0.0)  THEN begin
           coeffs[i,j,0:ntemp-1] = 0.0
           s = 10
        endif else begin
           coeffs[i,j,0:ntemp-1] = r[1:ntemp]
        endelse

        zmax[i,j] = r[0]
        status[i,j]=s
     endfor
  endfor

  ; Set the status value of nocount pixels to 11.0
  IF NOCOUNTS[0] NE -1 then BEGIN
     status[NOCOUNTS] = 11.0
     for n=0,n_elements(lgtaxis)-1 do begin
        coeffsnew =reform(coeffs[*,*,n])
        coeffsnew[NOCOUNTS] = 0.0
        coeffs[*,*,n] = coeffsnew
     endfor
  ENDIF

  ; Reconstruct EM using basis functions and coefficients
  oem  = fltarr(dim[0],dim[1],n_elements(lgtaxis))
  for i=0l,dim[0]-1 do for j=0l,dim[1]-1 do oem[i,j,*] = reform(reform(coeffs[i,j,*])#basis_funcs)

  ; Adaptively change tolfac to get more solutions
  if keyword_set(adaptive_tolfac) then begin
  bad = where(status NE 0)
  badimg = (reform(image, dim[0]*dim[1], 1, n_elements(image[0,0,*])))[bad,0,*]
  newtolfac = 1.5*tolfac
  aia_sparse_em_solve, badimg, oem=newems, tolfunc=tolfunc, tolfac=newtolfac, status=newstatus, symmbuff=symmbuff, coeffs=newcoeffs, adapt=0
  oem = reform(oem, dim[0]*dim[1], n_elements(lgtaxis))
  oem[bad,*] = newems
  oem = reform(oem,dim[0],dim[1],n_elements(lgtaxis))
  status = reform(status, dim[0]*dim[1])
  status[bad] = newstatus
  status = reform(status, dim[0], dim[1])
  coeffs=reform(coeffs,dim[0]*dim[1], ntemp)
  coeffs[bad,*] = newcoeffs
  coeffs = reform(coeffs, dim[0], dim[1], ntemp)
  tolmap[bad] = newtolfac
  
  ; Adaptively change tolfac to get more solutions                                                                                                                      
  bad = where(status NE 0)
  badimg = (reform(image, dim[0]*dim[1], 1, n_elements(image[0,0,*])))[bad,0,*]
  newtolfac = 1.5*newtolfac
  aia_sparse_em_solve, badimg, oem=newems, tolfunc=tolfunc, tolfac=newtolfac, status=newstatus, symmbuff=symmbuff, coeffs=newcoeffs, adapt=0
  oem = reform(oem, dim[0]*dim[1], n_elements(lgtaxis))
  oem[bad,*] = newems
  oem = reform(oem,dim[0],dim[1],n_elements(lgtaxis))
  status = reform(status, dim[0]*dim[1])
  status[bad] = newstatus
  status = reform(status, dim[0], dim[1])
  coeffs=reform(coeffs,dim[0]*dim[1], ntemp)
  coeffs[bad,*] = newcoeffs
  coeffs = reform(coeffs, dim[0], dim[1], ntemp)
  tolmap[bad] = newtolfac
  endif

  lgt = lgtaxis
  print, 'AIA_SPARSE_EM_SOLVE: Fraction of pixels with solutions = ', string(float(n_elements(where(status EQ 0)))/n_elements(zmax),format='(F5.2)')
  print, 'AIA_SPARSE_EM_SOLVE: EM in units of', Dictfac, ' cm^-5'
  print, 'AIA_SPARSE_EM_SOLVE: pixels / s', float(dim[0])*dim[1]/(anytim(systim()) - start)
  
  
END

FUNCTION AIA_SPARSE_EM_LGTAXIS
   COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs
   return, lgtaxis
END

PRO AIA_SPARSE_EM_INIT, restore=restore, evenorm=evenorm, save=save, timedepend_date=timedepend_date, $
                        use_lgtaxis = use_lgtaxis, noblend=noblend, xrt=xrt, photospheric=photospheric, bases_sigmas=bases_sigmas, $
                        bases_powers=bases_powers, emunit=emunit, normalize=normalize, chianti94fix=chianti94fix

   ; Normalize keyword makes the basis functions normalized by sum. Not recommended
   COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs
   nchannels = 6

   ; Semi-empirical correction to the
   ; temperature response function of the
   ; 94 channel
   chiantifix = 1
   IF N_ELEMENTS(chianti94fix) NE 0 then chiantifix = chianti94fix[0]

   IF N_ELEMENTS(xrt) NE 0 THEN nchannels = 6 + n_elements(XRT)
   IF N_ELEMENTS(evenorm) EQ 0 then evenorm = 1
   IF N_ELEMENTS(restore) EQ 0 then restore=0
      
   IF N_ELEMENTS(bases_sigmas) EQ 0 then bases_sigmas=[0.0,0.1,0.2,0.6]
   print, '>>>>>>>>> Initializing aia_sparse_em quantities <<<<<<<<<'
   print, '>>>>>>>>> bases_sigmas = [', strjoin(strtrim(string(bases_sigmas),2),',')+']'
   print, '>>>>>>>>> normalize basis vectors = '+strtrim(string(keyword_set(normalize)),2)

   IF keyword_set(photospheric) THEN BEGIN
      ; photospheric
      ;restgen, file='/net/crom/Volumes/disk2/data/aia/response/aia_V6_photo_fullemiss.genx', str=aia_resp
      ;restgen, file='/sanhome/cheung/AIA/aia_tresp_photospheric.genx', str=aia_temp_resp
      restgen, file='/Users/cheung/AIA/aia_V6_photo_fullemiss.genx', str=aia_resp
      restgen, file='/Users/cheung/AIA/aia_tresp_photospheric.genx', str=aia_temp_resp
      print, '>>>>>>>>>>Photospheric abundances! <<<<<<<<<<'
   ENDIF ELSE BEGIN
      ;print, 'Calling aia_get_response(evenorm='+string(evenorm,format='(I2)')+')'
      aia_resp = aia_get_response(/emiss, /full,noblend=noblend)
      cmd_aia_temp_resp = 'aia_temp_resp=aia_get_response(/dn,/temp,chiantifix='+string(chiantifix,format='(I1)')+',evenorm='+string(evenorm,format='(I1)')+",timedepend_date='"+timedepend_date+"',noblend="+string(keyword_set(noblend),format='(I1)')+')'
      res = execute(cmd_aia_temp_resp)
      print, '>>>>>>>>> '+cmd_aia_temp_resp
      ;aia_temp_resp = aia_get_response(/dn,/temp,chiantifix=chiantifix,evenorm=evenorm,timedepend_date=timedepend_date,noblend=noblend)
   ENDELSE
      ; If use_lgtaxis is given then 
      if (n_elements(use_lgtaxis) GT 6) then begin
         lgtaxis = use_lgtaxis
         print, ">>>>>>>>> lgtaxis = ["+strjoin(strtrim(string(lgtaxis,format='(F4.2)'),2),',')+']'
         ntemp = n_elements(lgtaxis)
         dlgt = lgtaxis[1]-lgtaxis[0]
      endif else begin
         lgtaxis = findgen(21)*0.1 + 5.5 ; Default lgT axis used in Cheung et al. (2015)
         ;print, ">>>>>>>>> Need more than 6 degrees of freedom for inversion!"
      endelse

      print, ">>>>>>>>> lgtaxis = ["+strjoin(strtrim(string(lgtaxis,format='(F4.2)'),2),',')+']'
      ntemp = n_elements(lgtaxis)
      dlgt = lgtaxis[1]-lgtaxis[0]
         

      if n_elements(bases_sigmas) EQ 0 then  bases_sigmas = [0.0]
      nsigmas= n_elements(bases_sigmas)
      npowers= n_elements(bases_powers)
      basis_funcs = fltarr((nsigmas+npowers)*ntemp,ntemp)
      Dict = fltarr(nchannels,(nsigmas+npowers)*ntemp)

      print, '>>>>>>>>> Size of the matrix D=KB [rows,columns] = [', strjoin(strtrim(string(size(dict,/dim),format='(I5)'),2),',')+']' 

      for s=0,nsigmas-1 do begin
         if bases_sigmas[s] EQ 0.0 then begin
            for m=0,ntemp-1 do basis_funcs[ntemp*s + m,m] = 1.0 ; Basis with delta functions
         endif else begin
            ; Compute normalization factor
            extended_lgtaxis = (findgen(50)-25)*(lgtaxis[1]-lgtaxis[0]) + 6.0
            line = exp(-((extended_lgtaxis-6.0)/bases_sigmas[s])^2.0)
            cut = where((line LT 0.04))
            IF cut[0] NE -1 then line[cut] = 0.0
            norm = total(line)
            for m=0,ntemp-1 do begin
               line = exp(-((lgtaxis-lgtaxis[m])/bases_sigmas[s])^2.0)
               cut = where((line LT 0.04))
               IF cut[0] NE -1 then line[cut] = 0.0
               IF KEYWORD_SET(normalize) THEN line = line/norm;total(line)
               basis_funcs[ntemp*s + m,0:ntemp-1] = line
            endfor
         endelse
      endfor
      
      for s=0,npowers-1 do begin
            for m=0,ntemp-1 do begin
               if bases_powers[s] LT 0.0 THEN $
               basis_funcs[ntemp*(s+nsigmas) + m , m:ntemp-1] = exp((lgtaxis[m:ntemp-1]-lgtaxis[m])/bases_powers[s])
               if bases_powers[s] GT 0.0 THEN $
               basis_funcs[ntemp*(s+nsigmas) + m , 0:m] = exp((lgtaxis[0:m]-lgtaxis[m])/bases_powers[s])
            endfor
      endfor

      for m=0,(nsigmas+npowers)*ntemp-1 do begin
         Dict[0,m] =  total(basis_funcs[m,0:ntemp-1]*(10.^interpol(alog10(aia_temp_resp.a94.tresp ), aia_temp_resp.a94.logte,  lgtaxis)))
         Dict[1,m] =  total(basis_funcs[m,0:ntemp-1]*(10.^interpol(alog10(aia_temp_resp.a131.tresp), aia_temp_resp.a131.logte, lgtaxis)))
         Dict[2,m] =  total(basis_funcs[m,0:ntemp-1]*(10.^interpol(alog10(aia_temp_resp.a171.tresp), aia_temp_resp.a171.logte, lgtaxis)))
         Dict[3,m] =  total(basis_funcs[m,0:ntemp-1]*(10.^interpol(alog10(aia_temp_resp.a193.tresp), aia_temp_resp.a193.logte, lgtaxis)))
         Dict[4,m] =  total(basis_funcs[m,0:ntemp-1]*(10.^interpol(alog10(aia_temp_resp.a211.tresp), aia_temp_resp.a211.logte, lgtaxis)))
         Dict[5,m] =  total(basis_funcs[m,0:ntemp-1]*(10.^interpol(alog10(aia_temp_resp.a335.tresp), aia_temp_resp.a335.logte, lgtaxis)))
      endfor

      ; Add XRT response functions if needed
      IF (N_ELEMENTS(XRT) NE 0) THEN BEGIN

         ; Generate temperature response function for lgtaxis
         modelname = 'AIA SPARSE EM'
         wave = aia_resp.total.wave
         temp = 10^(aia_resp.total.logte)
         spectrum = aia_resp.total.emissivity
         abund_model = aia_resp.general.abundfile
         ioneq_model = aia_resp.general.ioneq_name
         dens_model = aia_resp.general.model_name +', p='+trim(aia_resp.general.model_pe, 1)   
         emiss_model = make_xrt_emiss_model(modelname, wave, temp, spectrum, $
                                            abund_model, ioneq_model,dens_model,$
                                            data_files=data_files)

         wresps_all = make_xrt_wave_resp(contam_time=timedepend_date)
         xrt_resp = make_xrt_temp_resp(wresps_all, emiss_model)
         FOR d=0,n_elements(XRT)-1 DO BEGIN
            channel = XRT[d]
            case channel of 
               'Al-mesh': c = 0
               'Al-poly': c = 1
               'C-poly': c = 2
               'Ti-poly': c = 3
               'Be-thin': c = 4
               'Be-med': c = 5
               'Al-med': c = 6
               'Al-thick': c = 7
               'Be-thick': c = 8
               'Al-poly/Al-mesh': c = 9
               'Al-poly/Ti-poly': c = 10
               'Al-poly/Al-thick': c = 11
               'Al-poly/Be-thick': c = 12
               'C-poly/Ti-poly': c = 13
               'C-poly/Al-thick': c = 14
               default: begin
                  print, 'XRT:'+channel +': not found'
               endcase
            endcase
            ; channel number
            this_xrt_channel = xrt_resp[c]

          ; Interpolate response function onto lgtaxis 
            good = where(this_xrt_channel.temp_resp NE 0)
            xtemp_resp = this_xrt_channel.temp_resp[good]
            xtemp_coord= this_xrt_channel.temp[good]

            ntemp = n_elements(lgtaxis)
            for m=0,(nsigmas+npowers)*ntemp-1 do begin
               Dict[6+d,m] = total(basis_funcs[m,0:ntemp-1]*(10.^(interpol(alog10(xtemp_resp), alog10(xtemp_coord), lgtaxis))))
            endfor
         ENDFOR
      ENDIF

      Dictfac = 1e26; Units of EM solution will be in 1e26 
      IF n_elements(emunit) NE 0 THEN Dictfac = emunit
      Dict *= Dictfac 

      print, '>>>>>>>>> Units of EM solution will be aia_sparse_em_units() = '+string(Dictfac,format='(G12.7)')+' cm^-5'
      print, '>>>>>>>>> aia_sparse_em_init executed. See Cheung et al. (2015, http://adsabs.harvard.edu/abs/2015ApJ...807..143C)'
      print, '>>>>>>>>> for information about the sparse inversion technique.                                                   <<<<<<<<<'
END


FUNCTION AIA_SPARSE_EM_DISPLAY, em=em, status=status, maxdims=maxdims, lgemrange=lgemrange
  ; returns an image displaying slices of the DEM.
    COMMON AIAEM, Dict, Dictfac, halfts, indices, lgtaxis, dlgt, chiantifix, aia_resp, aia_temp_resp, nchannels, basis_funcs

  IF N_ELEMENTS(maxdims) NE 2 THEN maxdims = get_screen_size()
  lgemrange = [26, 29.5] ; Range for displaying EM, also used for colorbar
  
  dim = size(em,/dim)
  npanels = [3,2]
  dlgt = 0.5*(lgtaxis[1]-lgtaxis[0])
  lgtstride = floor(n_elements(lgtaxis)/float(npanels[0]*npanels[1]))
  lgtindices = indgen(npanels[0]*npanels[1])*lgtstride
  lgtindices += lgtstride/2

  DNAME = !D.NAME
  PFONT = !P.font
  OLDPOSITION=!P.position
  !P.font = 1

  set_plot,'Z'
  device, set_resolution=[npanels[0]*dim[0],npanels[1]*dim[1]], set_pixel_depth=24, decompose=0
  tvlct, r, g, b, /get
  ioffset = 0
  for i=0,npanels[0]-1 do begin
     joffset = 0
     for j=0,npanels[1]-1 do begin
        k = j*npanels[0] + i
        loadct, 22, /silent
        tv, bytscl(alog10(total(em[*,*,lgtindices[k]:lgtindices[k]+lgtstride-1],3)*aia_sparse_em_units()),min=lgemrange[0],max=lgemrange[1]), ioffset, joffset
        caption = 'EM in lgt=['+$
                strtrim(string(lgtaxis[lgtindices[k]]-dlgt,format='(F5.2)'),2)+','+$
                strtrim(string(lgtaxis[lgtindices[k]+lgtstride-1]+dlgt,format='(F5.2)'),2)+']'

        xyouts, ioffset+5, joffset+5, caption,charsize=2.4, charthick=3, color=0, /device
        xyouts, ioffset+7, joffset+7, caption,charsize=2.4, charthick=3, color=255, /device

        joffset += dim[1]
     endfor
     ioffset += dim[0]
  endfor
  position = [float(ioffset-70)/(npanels[0]*dim[0]),float(joffset-0.9*dim[1])/(npanels[1]*dim[1]),$
              float(ioffset-40)/(npanels[0]*dim[0]),float(joffset-0.1*dim[1])/(npanels[1]*dim[1])]
  colorbar, /vertical, range=lgemrange, position=position, title='EM [cm^-5]', $
            charsize=2.4, divisions=(lgemrange[1]-lgemrange[0])/0.5,format='(F4.1)'
  loadct, 0, /silent
  !P.position=position
  ;axis, xaxis=0, charthick=2, xrange=[0,1], color=1, /normal, /xstyle
  ;axis, xaxis=1, charthick=2, xrange=[0,1], color=1, /normal, /xstyle
  ;axis, yaxis=0, charthick=2, yrange=lgemrange, color=1, /normal, /ystyle
  ;axis, yaxis=1, charthick=2, yrange=lgemrange, color=1, /normal, /ystyle

  plotimg = tvrd(/true)
  
  tvlct, r, g, b
  !P.font=PFONT
  set_plot, DNAME
  !P.position=OLDPOSITION
  IF (npanels[0]*dim[0] GT maxdims[0]) OR (npanels[1]*dim[1] GT maxdims[1]) THEN BEGIN
     ratio = maxdims[0]/float(npanels[0]*dim[0])
     ratio = min([ratio,maxdims[1]/float(npanels[1]*dim[1])])
     nx = round(npanels[0]*dim[0]*ratio)
     ny = round(npanels[1]*dim[1]*ratio)
     newplotimg = bytarr(3,nx,ny)
     for c=0,2 do newplotimg[c,*,*] = congrid(reform(plotimg[c,*,*]),nx,ny,/interp)
     return, newplotimg

  ENDIF

  return, plotimg
END
