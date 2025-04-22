;Last Updated: <2024/08/02 13:49:18 from dhcp-005-198.mtk.nao.ac.jp by teiakiko>

;================================================
; Directries that already exist
;================================================
;------------------------------------------------
; workspace
;------------------------------------------------
;dir='/Volumes/BUFSSD2T/Data/MURaM/'
dir='../'

;------------------------------------------------
; G(T,N) sav file
;------------------------------------------------
dir_G_of_T_N=dir+'G_of_T_N/'

;------------------------------------------------
; Input atmosphere
;------------------------------------------------
dir_in=dir+'input/'+'AR_64x192x192/'

;------------------------------------------------
; Output data
;------------------------------------------------
dir_out=dir+'output/'


;================================================
; new directries
;================================================
;------------------------------------------------
; for the atmosphere
;------------------------------------------------
dir_Carr=dir_out+'Carr/'            ;Carr
dir_d=dir_out+'d'+sttri(domain)+'/' ;domain

;------------------------------------------------
; for the domain
;------------------------------------------------
dir_LI=dir_d+'LI/'              ;line intensity
dir_SI=dir_d+'SI/'              ;specific intensity
dir_SI_PSF=dir_d+'SI_PSF/'      ;specific intensity after considering PSF
dir_DV_LW_WN=dir_d+'DV_LW_WN/'  ;Doppler velocity, line width, nonthermal width
dir_plot=dir_d+'plot/'          ;plot

;------------------------------------------------
; in the plot dir
;------------------------------------------------
dir_plot_LI_DV_LW_WN=dir_plot+'LI_DV_LW_WN/'

;------------------------------------------------
; for the spectral line
;------------------------------------------------
dir_Carr_ll=dir_Carr+'#'+sttri(ll)+'_'+lstr+'/'
dir_LI_ll=dir_LI+'#'+sttri(ll)+'_'+lstr+'/'
dir_SI_ll=dir_SI+'#'+sttri(ll)+'_'+lstr+'/'
dir_SI_PSF_ll=dir_SI_PSF+'#'+sttri(ll)+'_'+lstr+'/'
dir_DV_LW_WN_ll=dir_DV_LW_WN+'#'+sttri(ll)+'_'+lstr+'/'
dir_plot_LI_DV_LW_WN_ll=dir_plot_LI_DV_LW_WN+'#'+sttri(ll)+'_'+lstr+'/'

;================================================
; make new directries
;================================================
file_mkdir,dir_Carr,dir_d,dir_LI,dir_SI,dir_SI_PSF,dir_plot_LI_DV_LW_WN
file_mkdir,dir_Carr_ll,dir_LI_ll,dir_SI_ll,dir_SI_PSF_ll,dir_DV_LW_WN_ll,dir_plot_LI_DV_LW_WN_ll

print,'The following directries are created:'
print,dir_Carr,dir_d,dir_LI,dir_SI,dir_SI_PSF,dir_plot_LI_DV_LW_WN
print,dir_Carr_ll,dir_LI_ll,dir_SI_ll,dir_SI_PSF_ll,dir_DV_LW_WN_ll,dir_plot_LI_DV_LW_WN_ll


;save,f=dir_Carr_ll    +'Carr_'    +lstr+'_'                   +tstep+'.sav',Carr,Carr2
;save,f=dir_LI_ll      +'LI_'      +lstr+'_d'+sttri(domain)+'_'+tstep+'.sav',LI_xz,LI_xy,LI_yz
;save,f=dir_SI_ll      +'SI_'      +lstr+'_d'+sttri(domain)+'_yz'+'_'+tstep+'.sav',SI_yz_dl
;save,f=dir_DV_LW_WN_ll+'DV_LW_WN_'+lstr+'_d'+sttri(domain)+'_xy'+'_'+tstep+'.sav',SI_xy_dl_fit,DV_xy,LW_xy,WN_xy
