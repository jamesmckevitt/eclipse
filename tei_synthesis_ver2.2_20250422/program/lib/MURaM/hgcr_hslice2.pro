; hgcr_hslice.pro
;
; Project: Heliophysics Grand Challenges Research
;          Physics and Diagnostics of the Drivers of Solar Eruptions
; Author: Mark Cheung
; Revision history: 2016-01-16: First draft v0.8
; Purpose: Returns 2D horizontal slice of MHD variables from a MURaM simulation,
;          reads in files like xz_slice_0256.XXXXXX
; Usage: IDL> e = hgcr_hslice(340000l, 0, k = 256, time=time) ; time is sim time
;        IDL> e = hgcr_hslice(340000l, 0, k = 116, time=time) ; different height

; To get 8 MHD variables consistent with a right-handed coordinate system:
; IDL>rho = hgcr_hslice(340000l, 0, k=256, time=time) ; mass density in g/cm^3
; IDL>vx  = hgcr_hslice(340000l, 1, k=256, time=time) ; v_x in km/s
; IDL>vy  = hgcr_hslice(340000l, 3, k=256, time=time) ; v_y in km/s 
; IDL>vz  = hgcr_hslice(340000l, 2, k=256, time=time) ; v_z in km/s VERTICAL!!!
; IDL>eps = hgcr_hslice(340000l, 4, k=256, time=time) ; uint erg/g 
; IDL>bx  = hgcr_hslice(340000l, 5, k=256, time=time)*sqrt(4*!PI) ; b_x in G
; IDL>by  = hgcr_hslice(340000l, 7, k=256, time=time)*sqrt(4*!PI) ; b_y in G
; IDL>bz  = hgcr_hslice(340000l, 6, k=256, time=time)*sqrt(4*!PI) ; b_z in G VERTICAL!!!

; The order above [0,1,3,2,4,5,7,6] is not a typo. 

function hgcr_hslice2, nmod, index, k=k,congrid=congrid, time=time, copy=copy
  common roi, i0, i1, j0, j1

  if n_elements(i0) eq 0 then i0 = 0
  if n_elements(j0) eq 0 then j0 = 0
  if N_elements(k) EQ 0 then k = 256
  
  k = string(k,format="(I04)")
  
  nmodstr = string(nmod,format="(I07)")
  openr,u, "./2D/xz_slices_every200steps/xz_slice_"+k+"."+nmodstr, /get_lun
  a=assoc(u,fltarr(4))
  header=a(0)
  nx = round(header[1])
  ny = round(header[2])
  i1 = nx-1
  j1 = ny-1
  time = header[3]
  
  a=assoc(u,fltarr(nx,ny),4*4)
  case index of 
     8: slice = sqrt(a(5)^2+a(7)^2) ;a(5):bx, a(7):by, a(6): bz
     9: slice = sqrt(a(5)^2+a(6)^2+a(7)^2)
     10: begin
        bhor = sqrt(a(5)^2+a(7)^2)
        field= where(bhor GT 0.5, complement = nofield)
        slice = fltarr(nx,ny)
        slice[field] = acos((a(5))[field]/bhor[field])*180.0/!PI
        field = where((a(7) LT 0.0) AND (bhor GT 0.5))
        slice[field] = -slice[field]
        slice[nofield]= 181.0
     endcase
     else: slice = a(index)
  endcase
  close, u
  free_lun, u
  if keyword_set(congrid) then return, rebin(slice,nx/2,ny/2)
  IF keyword_set(copy) then return, [[slice[i0:i1,j0:j1],slice[i0:i1,j0:j1]],[slice[i0:i1,j0:j1],slice[i0:i1,j0:j1]]]
  return,  slice[i0:i1,j0:j1]  
end
