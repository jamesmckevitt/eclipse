;Last Updated: <2021/12/29 11:46:59 from tambp.local by teiakiko>

;================================================
; Get Physical Variables from 3D MURaM Atmosphere
;================================================
; data length
dx=0.192*1e8                    ; [Mm]->[cm] pix size in x
dy=0.192*1e8                    ; [Mm]->[cm] pix size in y
dz=0.064*1e8                    ; [Mm]->[cm] pix size in z
nx=512                          ; [1] dimension in X
ny=256                          ; [1] dimension in Y
nz=768                          ; [1] dimension in Z
sx=nx*dx                        ; [cm] box size in x = 98.304 *1e8 cm = 49.152*2 *1e8 cm
sy=ny*dy                        ; [cm] box size in y = 49.152 *1e8 cm = 24.576*2 *1e8 cm
sz=nz*dz                        ; [cm] box size in z = 49.152 *1e8 cm ~ 49.1 *1e8 cm = 41.6+7.5 *1e8 cm
sxar_min=-sx/2
syar_min=-sy/2
szar_min=-7.5*1e8
sxar=(findgen(nx)+0.5)*dx+sxar_min  ; [cm] Solar-X arr
syar=(findgen(ny)+0.5)*dy+syar_min  ; [cm] Solar-Y arr
szar=(findgen(nz)+0.5)*dz+szar_min  ; [cm] Solar-Z arr

print,'Atmospheric parameters are set.'
