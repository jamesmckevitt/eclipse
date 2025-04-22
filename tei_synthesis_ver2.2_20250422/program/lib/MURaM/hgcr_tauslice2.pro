; hgcr_tauslice.pro
;
; Project: Heliophysics Grand Challenges Research
;          Physics and Diagnostics of the Drivers of Solar Eruptions
; Author: Mark Cheung
; Revision history: 2016-01-16: First draft v0.8
; Purpose: Returns 2D slice of MHD variables sampled at tau=0.1,
;          reads in files like tau_slice_0.100.XXXXXX
; Usage: IDL> e = hgcr_tauslice(340000l, 0, time=time) ; time is sim time

; To get 8 MHD variables consistent with a right-handed coordinate system:
; IDL>rho = hgcr_tauslice(340000l, 0, time=time) ; mass density in g/cm^3
; IDL>vx  = hgcr_tauslice(340000l, 1, time=time) ; v_x in km/s
; IDL>vy  = hgcr_tauslice(340000l, 3, time=time) ; v_y in km/s 
; IDL>vz  = hgcr_tauslice(340000l, 2, time=time) ; v_z in km/s VERTICAL!!!
; IDL>eps = hgcr_tauslice(340000l, 4, time=time) ; uint erg/g 
; IDL>bx  = hgcr_tauslice(340000l, 5, time=time)*sqrt(4*!PI) ; b_x in G
; IDL>by  = hgcr_tauslice(340000l, 7, time=time)*sqrt(4*!PI) ; b_y in G
; IDL>bz  = hgcr_tauslice(340000l, 6, time=time)*sqrt(4*!PI) ; b_z in G VERTICAL!!!

; The order above [0,1,3,2,4,5,7,6] is not a typo. 

function hgcr_tauslice2, nmod, index, time=time, tau=tau
  common roi, i0, i1, j0, j1
  if n_elements(i0) eq 0 then i0 = 0
  if n_elements(j0) eq 0 then j0 = 0
  if n_elements(tau) EQ 0 then tau = 0.100
  taustr = string(tau, format='(F5.3)')
  if tau LE 0.0001 then   taustr = string(tau, format='(F8.6)')


  nmodstr = string(nmod,format="(I07)")
  openr,u, "./2D/tau_slices_every200steps/tau_slice_"+taustr+"."+nmodstr, /get_lun
  a=assoc(u,fltarr(4))
  header=a(0)
  nx = round(header[1])
  ny = round(header[2])
  i1 = nx-1
  j1 = ny-1
  time = header[3]
  a=assoc(u,fltarr(nx,ny),4*4)
  case index of 
     9: slice = sqrt(a(5)^2+a(7)^2)
     10: slice = sqrt(a(5)^2+a(6)^2+a(7)^2)
     else:slice = a(index)
  endcase
  close, u
  free_lun, u
  return, slice[i0:i1,j0:j1]
end
