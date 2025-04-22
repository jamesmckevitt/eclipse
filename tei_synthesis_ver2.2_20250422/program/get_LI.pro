;Last Updated: <2021/12/29 14:00:40 from tambp.local by teiakiko>

;===============================================================
; Get Carr_d for the domain
;===============================================================
print,' '
print,'****************************************'
print,'Getting LI...'

restore,dir_Carr_ll+'Carr_'+lstr+'_'+tstep+'.sav',/verbose
print,'   Removing Carr, ...'
delvarx,Carr

get_Carr_d,xst,xen,yst,yen,zst,zen,nx,ny,nz,Carr2,Carr_d

get_LI,Carr_d,dx,dy,dz,LI_xy,LI_xz,LI_yz

print,'   Removing Carr2, Carr_d, ...'
delvarx,Carr2,Carr_d

print,'   Saving LI_xz, LI_xy, LI_yz, ...'
save,f=dir_LI_ll+'LI_'+lstr+'_d'+sttri(domain)+'_'+tstep+'.sav',LI_xz,LI_xy,LI_yz

print,'****************************************'
