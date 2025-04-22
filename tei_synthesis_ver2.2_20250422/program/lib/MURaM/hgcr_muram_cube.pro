; hgcr_muram_cube.pro
;
; Project: Heliophysics Grand Challenges Research
;          Physics and Diagnostics of the Drivers of Solar Eruptions
; Author: Mark Cheung
; Revision history: 2016-01-16: First draft v0.8
; Purpose: Reads in 3D snapshot of MHD variable
; Output : 3D cube, first two dimensions are the horizontal directions
; Usage:
; IDL> t  = hgcr_muram_cube("./3D/eosT.340000") ; Temperature in K
; IDL> p  = hgcr_muram_cube("./3D/eosP.340000") ; Pressure in dyne/cm^2
; IDL> rho= hgcr_muram_cube("./3D/result_prim_0.340000") ; density in g/cm^3
; IDL> vx = hgcr_muram_cube("./3D/result_prim_1.340000") ; vx in cm/s
; IDL> vy = hgcr_muram_cube("./3D/result_prim_3.340000") ; vy in cm/s
; IDL> vz = hgcr_muram_cube("./3D/result_prim_2.340000") ; vz in cm/s !VERTICAL!
; IDL> eps= hgcr_muram_cube("./3D/result_prim_4.340000") ; uint in erg/cm^3
; IDL> bx = hgcr_muram_cube("./3D/result_prim_5.340000")*sqrt(4*!PI)  ;bx in G
; IDL> by = hgcr_muram_cube("./3D/result_prim_7.340000")*sqrt(4*!PI)  ;by in G
; IDL> bz = hgcr_muram_cube("./3D/result_prim_6.340000")*sqrt(4*!PI)  ;bz in G ! VERTICAL !

function hgcr_muram_cube, file, pointer=pointer
  openr, u, file, /get_lun
  a=assoc(u,fltarr(512,768,256))
  cube=a(0)
  close, u
  free_lun, u
  cube = transpose(cube, [0,2,1])
  IF keyword_set(pointer) THEN return, ptr_new(cube, /no_copy)
  return, cube
end
