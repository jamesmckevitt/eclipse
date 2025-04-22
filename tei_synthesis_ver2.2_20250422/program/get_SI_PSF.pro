;Last Updated: <2024/08/02 16:30:25 from dhcp-005-198.mtk.nao.ac.jp by teiakiko>

print,' '
print,'****************************************'
print,'Getting SI_PSF...'

;===============================================================
; restore
;===============================================================
case los of
   'x': begin
      restore,dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_yz'+'_'+tstep+'.sav',/verbose
      get_SI_PSF,SI_yz_dl,dl,dvar,SI_PSF_yz_dl
      save,f=dir_SI_PSF_ll+'SI_PSF_'+lstr+'_d'+sttri(domain)+'_yz'+'_'+tstep+'.sav',SI_PSF_yz_dl
   end
   'y': begin
      restore,dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_xz'+'_'+tstep+'.sav',/verbose
      get_SI_PSF,SI_xz_dl,dl,dvar,SI_PSF_xz_dl
      save,f=dir_SI_PSF_ll+'SI_PSF_'+lstr+'_d'+sttri(domain)+'_xz'+'_'+tstep+'.sav',SI_PSF_xz_dl
   end
   'z': begin
      restore,dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_xy'+'_'+tstep+'.sav',/verbose
      get_SI_PSF,SI_xy_dl,dl,dvar,SI_PSF_xy_dl
      save,f=dir_SI_PSF_ll+'SI_PSF_'+lstr+'_d'+sttri(domain)+'_xy'+'_'+tstep+'.sav',SI_PSF_xy_dl
   end
endcase


print,' '
print,'****************************************'
