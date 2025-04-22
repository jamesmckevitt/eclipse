;Last Updated: <2024/08/02 14:32:44 from dhcp-005-198.mtk.nao.ac.jp by teiakiko>

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

print,'   Reading MURaM atmosphere (N_e, T_p)'

print,'      Reading rho...'
rho=hgcr_muram_cube(dir_in+'rho/result_prim_0.'+tstep) ; rho [g/cm^3] Mass Density
n_e=rho/(mu_a*m_0)    ; n_e [/cm^3] electron density => see Vogler05 appendix A
logN=alog10(n_e)
print,'      Removing rho, n_e...'
delvarx,rho,n_e



print,'      Reading t_p...'
t_p=hgcr_muram_cube(dir_in+'temp/eosT.'+tstep) ; T [K] Plasma Temperature
