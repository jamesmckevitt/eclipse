; hgcr_iout.pro
;
; Project: Heliophysics Grand Challenges Research
;          Physics and Diagnostics of the Drivers of Solar Eruptions
; Author: Mark Cheung
; Revision history: 2016-01-16: First draft v0.8
; Purpose: Returns vertical outgoing intensity in units of erg/cm^2/s,
;          reads in files like Iout.XXXXXX
; Usage: IDL> e = hgcr_iout(340000l, time=time) ; time is sim time

function hgcr_Iout, nmod, congrid=congrid,scale=scale,time=time
  common roi, i0, i1, j0, j1
  if n_elements(i0) eq 0 then i0 = 0
  if n_elements(j0) eq 0 then j0 = 0
  
  nmodstr = string(nmod,format="(I06)")
  openr,u, "./2D/I_out."+nmodstr, /get_lun
  a=assoc(u,fltarr(4))
  header=a(0)
  nx = round(header[1])
  ny = round(header[2])
  time=header[3]
  i1 = nx-1
  j1 = ny-1
  a=assoc(u,fltarr(nx,ny),4*4)
  slice = a(0)
  close, u
  free_lun, u
  slice = slice[i0:i1,j0:j1]
  
  if keyword_set(congrid) then slice = rebin(slice,960,512) ;congrid(slice,960,512)
  if keyword_set(scale) then slice = bytscl(slice,min=1.2e10,max=4.5e10)
  
  return, slice  
end
