;Last Updated: <2024/08/02 14:46:51 from dhcp-005-198.mtk.nao.ac.jp by teiakiko>

;================================================
; Get Physical Variables from 3D MURaM Atmosphere
;================================================
; Coordinates: X & Y : horizontal, Z: vertical

; 11 most abundant elements in the solar photosphere are included
;sum_nu $ ; [1] Sum of Relative Abundances => ~ 1 of course!
;   =0.934042096 $ ; 01. H
;   +0.064619943 $ ; 02. He
;   +0.000371849 $ ; 06. C
;   +0.000091278 $ ; 07. N
;   +0.000759218 $ ; 08. O
;   +0.000035511 $ ; 12. Mg
;   +0.000001997 $ ; 11. Na
;   +0.000002140 $ ; 20. Ca
;   +0.000039844 $ ; 26. Fe
;   +0.000033141 $ ; 14. Si
;   +0.000002757   ; 13. Al

print,'   Reading MURaM atmosphere (V_los, T_p)'

case los of
   'x':begin
      print,'      Reading vx...'
      vx=hgcr_muram_cube(dir_in+'vx/result_prim_1.'+tstep) ; Vx [cm/s] Velocity
   end
   'y':begin
      print,'      Reading vy...'
      vy=hgcr_muram_cube(dir_in+'vy/result_prim_3.'+tstep) ; Vy [cm/s] Velocity
   end
   'z':begin
      print,'      Reading vz...'
      vz=hgcr_muram_cube(dir_in+'vz/result_prim_2.'+tstep) ; Vz [cm/s] Velocity
   end
endcase

print,'      Reading t_p...'
t_p=hgcr_muram_cube(dir_in+'temp/eosT.'+tstep) ; T [K] Plasma Temperature
