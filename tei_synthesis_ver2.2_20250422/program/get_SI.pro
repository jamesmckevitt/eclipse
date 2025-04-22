;Last Updated: <2022/01/07 09:53:26 from tambp.local by teiakiko>

print,' '
print,'****************************************'
print,'Getting SI...'

;===============================================================
; restore
;===============================================================
restore,dir_Carr_ll+'Carr_'+lstr+'_'+tstep+'.sav',/verbose
print,'   Removing Carr...'
delvarx,Carr

;===============================================================
; read MURaM atmosphere
; -> Read v_los & t_p
;===============================================================
@read_atm_V_T

;===============================================================
; Total Doppler width (thermal only now) in velocity [cm/s]
;===============================================================
m_Z=W_Z/N_A                     ; mass of the atom [g]
dv_dw=sqrt(2.*k_b*t_p/m_Z)
print,'   Removing t_p...'
delvarx,t_p

;===============================================================
; Get various cubes for the domain
;===============================================================
print,'   Running get_v_domain...'
case los of
   'x': begin
;      get_cubes_domain_los,xst,xen,yst,yen,zst,zen,$
      get_v_domain,xst,xen,yst,yen,zst,zen,nx,ny,nz,$
                   dv_dw,vx,dv_dw_d,vx_d
   end
   'y': begin
      get_v_domain,xst,xen,yst,yen,zst,zen,nx,ny,nz,$
                   dv_dw,vy,dv_dw_d,vy_d
   end
   'z': begin
      get_v_domain,xst,xen,yst,yen,zst,zen,nx,ny,nz,$
                   dv_dw,vz,dv_dw_d,vz_d
   end
endcase
print,'   Removing dv_dw...'
delvarx,dv_dw

;================================================
; Line synthesis for MURaM 3D atmosphere
;================================================

;================================================
; get Carr_d
;================================================
print,'   Running get_Carr_d...'
get_Carr_d,xst,xen,yst,yen,zst,zen,nx,ny,nz,Carr2,Carr_d
print,'   Removing Carr2...'
delvarx,Carr2

;================================================
; Specific Intensity
;================================================
case los of
   'x': begin
      get_SI,Carr_d,dv_dw_d,vx_d,dx,dvar,dl,los='x',SI_yz_dl
      save,f=dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_yz'+'_'+tstep+'.sav',SI_yz_dl
      delvarx,vx,SI_yz_dl
   end
   'y': begin
      get_SI,Carr_d,dv_dw_d,vy_d,dy,dvar,dl,los='y',SI_xz_dl
      save,f=dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_xz'+'_'+tstep+'.sav',SI_xz_dl
      delvarx,vy,SI_xz_dl
   end
   'z': begin
      get_SI,Carr_d,dv_dw_d,vz_d,dz,dvar,dl,los='z',SI_xy_dl
      save,f=dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_xy'+'_'+tstep+'.sav',SI_xy_dl
      delvarx,vz,SI_xy_dl
   end
endcase

; from X direction: SI_yz_dl [erg/s/cm^2/sr/cm]
; from Y direction: SI_xz_dl [erg/s/cm^2/sr/cm]
; from Z direction: SI_xy_dl [erg/s/cm^2/sr/cm]


print,' '
print,'****************************************'
