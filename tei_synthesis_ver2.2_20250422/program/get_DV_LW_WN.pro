;Last Updated: <2025/04/01 12:43:42 from wlan-65-003.mtk.nao.ac.jp by teiakiko>

print,' '
print,'****************************************'
print,'Getting DV,LW,WN...'

;===================================================
; Single Gaussian Fitting of Synthesized Profiles
;===================================================
case psf of

   0: begin

      case los of
         'x': begin
         ; Fitting SI_yz_dl -> SI_yz_dl_fit
            restore,dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_yz'+'_'+tstep+'.sav',/verbose ;SI_yz_dl
            single_gauss_fit,dlar,SI_yz_dl,min_SI,la_0,SI_yz_dl_fit,DV_yz,LW_yz
            get_WN,LW_yz,W_Z,logT,WN_yz
            save,f=dir_DV_LW_WN_ll+'DV_LW_WN_'+lstr+'_d'+sttri(domain)+'_yz'+'_'+tstep+'.sav',$
                 SI_yz_dl_fit,DV_yz,LW_yz,WN_yz
;            delvarx,SI_yz_dl,SI_yz_dl_fit
         end
         'y': begin
         ; Fitting SI_xz_dl -> SI_xz_dl_fit
            restore,dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_xz'+'_'+tstep+'.sav',/verbose ;SI_xz_dl
            single_gauss_fit,dlar,SI_xz_dl,min_SI,la_0,SI_xz_dl_fit,DV_xz,LW_xz
            get_WN,LW_xz,W_Z,logT,WN_xz
            save,f=dir_DV_LW_WN_ll+'DV_LW_WN_'+lstr+'_d'+sttri(domain)+'_xz'+'_'+tstep+'.sav',$
                 SI_xz_dl_fit,DV_xz,LW_xz,WN_xz
            delvarx,SI_xz_dl,SI_xz_dl_fit
         end
         'z': begin
         ; Fitting SI_xy_dl -> SI_xy_dl_fit
            restore,dir_SI_ll+'SI_'+lstr+'_d'+sttri(domain)+'_xy'+'_'+tstep+'.sav',/verbose ;SI_xy_dl
            single_gauss_fit,dlar,SI_xy_dl,min_SI,la_0,SI_xy_dl_fit,DV_xy,LW_xy
            get_WN,LW_xy,W_Z,logT,WN_xy
            save,f=dir_DV_LW_WN_ll+'DV_LW_WN_'+lstr+'_d'+sttri(domain)+'_xy'+'_'+tstep+'.sav',$
                 SI_xy_dl_fit,DV_xy,LW_xy,WN_xy
            delvarx,SI_xy_dl,SI_xy_dl_fit
         end
      endcase

   end

   1: begin
      print,'psf=1 part is not yet implemented...'
   end

endcase


print,' '
print,'****************************************'
