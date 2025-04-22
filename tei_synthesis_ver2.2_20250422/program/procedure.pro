;Last Updated: <2025/04/01 12:54:11 from wlan-65-003.mtk.nao.ac.jp by teiakiko>

;================================================
; Define Program
;================================================

;================================================
;pro get_cubes_domain2,$
pro get_var_domain,$
   xst,xen,$                   ;input: start & end position in x direction [px]
   yst,yen,$                   ;input: start & end position in y direction [px]
   zst,zen,$                   ;input: start & end position in z direction [px]
   dx,dy,dz,$                  ;input: grid size in x,y,z dimension [cm]
   nx,ny,nz,$                  ;input: original cube size in x,y,z direction [px]
   sxar,syar,szar,$            ;input: original solar-x, solar-y, solar-z arr [cm]
   nx_d,ny_d,nz_d,$            ;output: cube size in x,y,z direction [px]
   sxar_d,syar_d,szar_d,$      ;output: solar-x, solar-y, solar-z arr [cm]
   sx_d,sy_d,sz_d              ;output: cube size in x, y, z direction [cm]
  
  if xen ge nx then xen=xen-nx
  if yen ge ny then yen=yen-ny
  if zen ge nz then zen=zen-nz
  sxar_d=sxar
  syar_d=syar
  szar_d=szar
  if xst lt xen then begin
     nx_d=xen-xst+1               ;[px]
     sxar_d=sxar_d[xst:xen]       ;[cm]
  endif else begin
     nx_d=((nx-1)-xst+1)+(xen-0+1)                      ;[px]
;     sxar_d=[sxar_d[xst:nx-1],sxar_d[0:xen]]            ;[cm]
     sxar_d=[sxar_d[xst:nx-1],sxar_d[-1]+sxar_d[0:xen]-sxar_d[0]]            ;[cm]
  endelse
  if yst lt yen then begin
     ny_d=yen-yst+1               ;[px]
     syar_d=syar_d[yst:yen]       ;[cm]
  endif else begin
     ny_d=((ny-1)-yst+1)+(yen-0+1)                      ;[px]
;     syar_d=[syar_d[yst:ny-1],syar_d[0:yen]]            ;[cm]
     syar_d=[syar_d[yst:ny-1],syar_d[-1]+syar_d[0:yen]-syar_d[0]]            ;[cm]
  endelse
  if zst lt zen then begin
     nz_d=zen-zst+1               ;[px]
     szar_d=szar_d[zst:zen]       ;[cm]
  endif else begin
     nz_d=((nz-1)-zst+1)+(zen-0+1)                      ;[px]
;     szar_d=[szar_d[zst:nz-1],szar_d[0:zen]]            ;[cm]
     szar_d=[szar_d[zst:nz-1],szar_d[-1]+szar_d[0:zen]-szar_d[0]]            ;[cm]
  endelse
  sx_d=nx_d*dx                  ;[cm]
  sy_d=ny_d*dy                  ;[cm]
  sz_d=nz_d*dz                  ;[cm]

end

;================================================
;pro get_cubes_domain_los,xst,xen,$        ;input: start & end position in x direction [px]
pro get_v_domain,xst,xen,$          ;input: start & end position in x direction [px]
                 yst,yen,$          ;input: start & end position in y direction [px]
                 zst,zen,$          ;input: start & end position in z direction [px]
                 nx,ny,nz,$         ;input: original cube size in x,y,z direction [px]
                 dv_dw,$            ;input: cube of original Doppler width [cm/s]
                 vx,$               ;vy,vz,$    ;input: cube of original velocity in x,y,z direction [cm/s]
                 dv_dw_d,$          ;output: cube of Doppler width [cm/s]
                 vx_d               ;vy_d,vz_d,$;output: cube of velocity in x,y,z direction [cm/s]
  if xen ge nx then xen=xen-nx
  if yen ge ny then yen=yen-ny
  if zen ge nz then zen=zen-nz
  dv_dw_d=dv_dw
  vx_d=vx
  ;vy_d=vy
  ;vz_d=vz
  if xst lt xen then begin
     dv_dw_d=dv_dw_d[xst:xen,*,*] ;[cm/s]
     vx_d=vx_d[xst:xen,*,*]       ;[cm/s]
     ;vy_d=vy_d[xst:xen,*,*]       ;[cm/s]
     ;vz_d=vz_d[xst:xen,*,*]       ;[cm/s]
  endif else begin
     dv_dw_d=[dv_dw_d[xst:nx-1,*,*],dv_dw_d[0:xen,*,*]]                      ;[cm/s]
     vx_d=[vx_d[xst:nx-1,*,*],vx_d[0:xen,*,*]]                               ;[cm/s]
     ;vy_d=[vy_d[xst:nx-1,*,*],vy_d[0:xen,*,*]]                               ;[cm/s]
     ;vz_d=[vz_d[xst:nx-1,*,*],vz_d[0:xen,*,*]]                               ;[cm/s]
  endelse
  if yst lt yen then begin
     dv_dw_d=dv_dw_d[*,yst:yen,*] ;[cm/s]
     vx_d=vx_d[*,yst:yen,*]       ;[cm/s]
     ;vy_d=vy_d[*,yst:yen,*]       ;[cm/s]
     ;vz_d=vz_d[*,yst:yen,*]       ;[cm/s]
  endif else begin
     dv_dw_d=[dv_dw_d[*,yst:ny-1,*],dv_dw_d[*,0:yen,*]]           ;[cm/s]
     vx_d=[vx_d[*,yst:ny-1,*],vx_d[*,0:yen,*]]                    ;[cm/s]
     ;vy_d=[vy_d[*,yst:ny-1,*],vy_d[*,0:yen,*]]                    ;[cm/s]
     ;vz_d=[vz_d[*,yst:ny-1,*],vz_d[*,0:yen,*]]                    ;[cm/s]
  endelse
  if zst lt zen then begin
     dv_dw_d=dv_dw_d[*,*,zst:zen] ;[cm/s]
     vx_d=vx_d[*,*,zst:zen]       ;[cm/s]
     ;vy_d=vy_d[*,*,zst:zen]       ;[cm/s]
     ;vz_d=vz_d[*,*,zst:zen]       ;[cm/s]
  endif else begin
     dv_dw_d=[dv_dw_d[*,*,zst:nz-1],dv_dw_d[*,*,0:zen]] ;[cm/s]
     vx_d=[vx_d[*,*,zst:nz-1],vx_d[*,*,0:zen]]          ;[cm/s]
     ;vy_d=[vy_d[*,*,zst:nz-1],vy_d[*,*,0:zen]]          ;[cm/s]
     ;vz_d=[vz_d[*,*,zst:nz-1],vz_d[*,*,0:zen]]          ;[cm/s]
  endelse

end

;================================================
;pro get_Carr_domain,xst,xen,$   ;input: start & end position in x direction [px]
pro get_Carr_d,xst,xen,$   ;input: start & end position in x direction [px]
               yst,yen,$   ;input: start & end position in y direction [px]
               zst,zen,$   ;input: start & end position in z direction [px]
               nx,ny,nz,$  ;input: original cube size in x,y,z direction [px]
               Carr,$      ;input: cube of original contribution function [erg/s/cm^3/sr]
               Carr_d      ;output: cube of contribution function [erg/s/cm^3/sr]
  if xen ge nx then xen=xen-nx
  if yen ge ny then yen=yen-ny
  if zen ge nz then zen=zen-nz
  Carr_d=Carr
  if xst lt xen then begin
     Carr_d=Carr_d[xst:xen,*,*]   ;[erg/s/cm^3/sr]
  endif else begin
     Carr_d=[Carr_d[xst:nx-1,*,*],Carr_d[0:xen,*,*]]    ;[erg/s/cm^3/sr]
  endelse
  if yst lt yen then begin
     Carr_d=Carr_d[*,yst:yen,*]   ;[erg/s/cm^3/sr]
  endif else begin
     Carr_d=[Carr_d[*,yst:ny-1,*],Carr_d[*,0:yen,*]]    ;[erg/s/cm^3/sr]
  endelse
  if zst lt zen then begin
     Carr_d=Carr_d[*,*,zst:zen]   ;[erg/s/cm^3/sr]
  endif else begin
     Carr_d=[Carr_d[*,*,zst:nz-1],Carr_d[*,*,0:zen]]    ;[erg/s/cm^3/sr]
  endelse
end

;================================================
pro get_LI,Carr,$               ;input: 3Darr of contribution [erg/s/cm^3/sr]
           d1,$                 ;input: grid size of 1st dimension [cm]
           d2,$                 ;input: grid size of 2nd dimension [cm]
           d3,$                 ;input: grid size of 3rd dimension [cm]
           LI_12,$              ;output: 2Darr of line intensity in 1-2 plane (LOS direction is dimension 3)
           LI_13,$              ;output: 2Darr of line intensity in 1-3 plane (LOS direction is dimension 2)
           LI_23                ;output: 2Darr of line intensity in 2-3 plane (LOS direction is dimension 1)
  LI_12=total(Carr,3)*d3        ; [erg/s/cm^2/sr]
  LI_13=total(Carr,2)*d2        ; [erg/s/cm^2/sr]
  LI_23=total(Carr,1)*d1        ; [erg/s/cm^2/sr]
end

;================================================
pro get_SI,Carr_d,$             ;input: 3Darr of contribution [erg/s/cm^3/sr]
           dv_dw_d,$            ;input: 3Darr of Doppler width [cm/s]
           dv_los_d,$           ;input: 3Darr of LOS velocity [cm/s]
           dlos,$               ;input: grid size in LOS direction [cm]
           dvar,$               ;input: wavelength arr [cm/s]
           dl,$                 ;input: grid size in wavelength direction [cm]
           los=los,$            ;input option to specify LOS direction
           SI_12_dl             ;output: specific intensity in 1-2 plane [erg/s/cm^2/sr/cm]
  print,'   Transposing...'
  case los of
     "z": begin
     ;-----------------------------------------------------------------
     ; Specific Intensity fron Z direction: SI_xy_dl [erg/s/cm^2/sr/cm]
     ; 0=X: horizontal direction -> 0th dimention
     ; 1=Y: vertical direction   -> 1st dimention
     ; 2=Z: LOS direction        -> 2nd dimention
     ;=> Transpose 3D cube with [X,Y,Z] order -> [0,1,2]
     ;-----------------------------------------------------------------
        Carr123=Carr_d          ;transpose 3D array of contribution
        dv_dw_ar=dv_dw_d        ;transpose 3D array of Doppler width
        dv_los_ar=dv_los_d      ;transpose 3D array of LOS velocity
     end
     "y": begin
     ;-----------------------------------------------------------------
     ; Specific Intensity fron Y direction: SI_xz_dl [erg/s/cm^2/sr/cm]
     ; 0=X: horizontal direction -> 0th dimention
     ; 2=Z: vertical direction   -> 1st dimention
     ; 1=Y: LOS direction        -> 2nd dimention
     ;=> Transpose 3D cube with [X,Z,Y] order -> [0,2,1]
     ;-----------------------------------------------------------------
        Carr123=transpose(Carr_d,[0,2,1])     ;transpose 3D array of contribution
        dv_dw_ar=transpose(dv_dw_d,[0,2,1])   ;transpose 3D array of Doppler width
        dv_los_ar=transpose(dv_los_d,[0,2,1]) ;transpose 3D array of LOS velocity
     end
     "x": begin
     ;-----------------------------------------------------------------
     ; Specific Intensity fron X direction: SI_yz_dl [erg/s/cm^2/sr/cm]
     ; 1=Y: horizontal direction -> 0th dimention
     ; 2=Z: vertical direction   -> 1st dimention
     ; 0=X: LOS direction        -> 2nd dimention
     ;=> Transpose 3D cube with [Y,Z,X] order -> [1,2,0]
     ;-----------------------------------------------------------------
        Carr123=transpose(Carr_d,[1,2,0])     ;transpose 3D array of contribution
        dv_dw_ar=transpose(dv_dw_d,[1,2,0])   ;transpose 3D array of Doppler width
        dv_los_ar=transpose(dv_los_d,[1,2,0]) ;transpose 3D array of LOS velocity
     end
     else: begin
        print,'specify "los" option!'
     end
  endcase
  ;pp,los
  k_b=!const.k*1e7              ;Boltzman constant [J/K]->[erg/K]
  nl=nel(dvar)
  n1=nel1(Carr123)
  n2=nel2(Carr123)
  n3=nel3(Carr123)                 ;los direction
  dv=dvar[1]-dvar[0]               ; wavelength grid size in velocity [cm/s]
  SI_12_dv=fltarr(n1,n2,nl)
;  print,dv/dl
;window,0,xs=500,ys=300
;wset,0
  for ii=0,n1-1 do begin
;  for ii=0,0 do begin
    pp,ii
     for jj=0,n2-1 do begin; & pp,jj
;     for jj=0,0 do begin; & pp,jj
        for kk=0,n3-1 do begin
;print,jj,kk
;pp,kk
           ; Maxewllian in velocity unit
           dv_sig=dv_dw_ar[ii,jj,kk]/sqrt(2.) ;[cm/s] =sqrt(k_b*te/m_fe)
           ; Coefficient of Gaussian
           aa_dv=1./(sqrt(2*!pi)*dv_sig) ;[s/cm] =1./(sqrt(!pi)*dv_te)
           params_dv=[aa_dv,dv_los_ar[ii,jj,kk],dv_sig,0.]
           f_dv=gaussian(dvar,params_dv) ; [s/cm]
           ; SI_12_dv
           SI_12_dv[ii,jj,*]=SI_12_dv[ii,jj,*]+f_dv*Carr123[ii,jj,kk]*dlos ; [erg/s/cm^3/sr s/cm * cm]
;wset,0
;plot,SI_12_dv[ii,jj,*],psym=-1,symsi=2
;wait,0.01
        endfor
;wset,1
;plot,SI_12_dv[ii,jj,*],psym=-1,symsi=2
;print,max(SI_12_dv[ii,jj,*]),alog10(max(SI_12_dv[ii,jj,*])),max(alog10(SI_12_dv[ii,jj,*]))
;wset,2
;plot,SI_12_dv[ii,jj,*]*dv/dl,psym=-1,symsi=2
;print,max(SI_12_dv[ii,jj,*]*dv/dl),alog10(max(SI_12_dv[ii,jj,*]*dv/dl)),max(alog10(SI_12_dv[ii,jj,*])*dv/dl)
;        if alog10(max(SI_12_dv[ii,jj,*]*dv/dl)) lt 10. then print,ii,jj ;line for debug
;wait,0.01
     endfor
  endfor
  SI_12_dl=SI_12_dv*dv/dl       ; [erg/s/cm^2/sr /cm]
;  mima,SI_12_dv
;  mima,SI_12_dl
;  wset,2
;  plot,dvar,SI_12_dv
;  plot,SI_12_dv
;  wset,3
;  plot,dvar,SI_12_dl
end

;================================================
pro get_SI_PSF,SI_12_dl,$       ;input
               dl,$             ;input: grid size in wavelength direction [cm]
               dvar,$           ;input
               SI_PSF_12_dl     ;output
  dv=dvar[1]-dvar[0]            ; wavelength grid size in velocity [cm/s]
  SI_12_dv=SI_12_dl*dl/dv
  n1=nel1(SI_12_dl)
  n2=nel2(SI_12_dl)
  n3=nel3(SI_12_dl)
  SI_PSF_12_dv=fltarr(n1,n2,n3)
  for kk=0,n3-1 do begin
     SI_PSF_12_dv[*,*,kk]=convol_euvst(SI_12_dv[*,*,kk])
  endfor
  SI_PSF_12_dl=SI_PSF_12_dv*dv/dl       ; [erg/s/cm^2/sr /cm]
end

;================================================
pro single_gauss_fit,dlar,$         ;input: wavelength arr [cm]
                     SI_12_dl,$     ;input: specific intensity in 1-2 plane [erg/s/cm^2/sr/cm]
                     min_SI,$       ;input: threshold of specific intensity for fitting ... Log_10([erg/s/cm^2/sr/cm])
                     la_0,$         ;input: rest wavelength [cm]
                     SI_12_dl_fit,$ ;output: fitting result of specific intensity in 1-2 plane [erg/s/cm^2/sr/cm]
                     DV_12,$        ;output: Doppler velocity [cm/s]
                     LW_12          ;output: line width [cm/s]
  ccc=!const.c*1e2                  ;=2.99792458e10 [cm/s], !const.c=2.99792458e8 [m/s]
  nterms=3                          ;number of free parameters for the fitting [1]
  SI_12_dl2=SI_12_dl/1e8
  min_SI2=min_SI-8
;print,min_SI,min_SI2
  n1=nel1(SI_12_dl)
  n2=nel2(SI_12_dl)
  SI_12_dl_fit=SI_12_dl*0.
  SI_12_dl2_fit=SI_12_dl2*0.
  DV_12=dblarr(n1,n2)
  LW_12=dblarr(n1,n2)
;window,0,xs=500,ys=300
;window,1,xs=500,ys=300
;window,2,xs=500,ys=300
;window,3,xs=500,ys=300
;set_line_color
  for ii=0,n1-1 do begin
;  for ii=n1-1,n1-1 do begin
     pp,ii
     for jj=0,n2-1 do begin
;     for jj=n2-1-200,n2-1 do begin
;print,alog10(max(SI_12_dl[ii,jj,*])),alog10(max(SI_12_dl2[ii,jj,*]))
;        if max(alog10(SI_12_dl2[ii,jj,*])) gt min_SI2 then begin
        if alog10(max(SI_12_dl2[ii,jj,*])) gt min_SI2 then begin
           SI_12_dl2_fit[ii,jj,*]=gaussfit(dlar,reform(SI_12_dl2[ii,jj,*]),aa,nterms=nterms,sigma=sigma)
                                ; [erg/s/cm^2/sr /cm]
           DV_12[ii,jj]=aa[1]/la_0*ccc ; Doppler Velocity [cm/s]
           LW_12[ii,jj]=aa[2]/la_0*ccc ; Line Width [cm/s] -> sigma
           SI_12_dl_fit[ii,jj,*]=SI_12_dl2_fit[ii,jj,*]*1e8
;print,DV_12[ii,jj]/1e5,LW_12[ii,jj]/1e5,aa
;wset,0
;plot,dlar,SI_12_dl2[ii,jj,*],thi=2,lin=2
;oplot,dlar,SI_12_dl2_fit[ii,jj,*],thi=1,lin=0,col=3
;wset,1
;plot,dlar,SI_12_dl_fit[ii,jj,*],thi=1,lin=0,col=5
;print,max(SI_12_dl_fit[ii,jj,*])
;wait,0.1
;stop
        endif else begin
;           print,'NG',ii,jj,alog10(max(SI_12_dl2[ii,jj,*]))
;           stop
        endelse
if 1 eq 0 then begin
        if alog10(max(SI_12_dl[ii,jj,*])) gt min_SI then begin
           SI_12_dl_fit[ii,jj,*]=gaussfit(dlar,reform(SI_12_dl[ii,jj,*]),aa,nterms=nterms,sigma=sigma)
                                ; [erg/s/cm^2/sr /cm]
           DV_12[ii,jj]=aa[1]/la_0*ccc ; Doppler Velocity [cm/s]
           LW_12[ii,jj]=aa[2]/la_0*ccc ; Line Width [cm/s] -> sigma
;print,DV_12[ii,jj]/1e5,LW_12[ii,jj]/1e5,aa
;wset,1
;oplot,dlar,SI_12_dl[ii,jj,*],thi=2,lin=2
;oplot,dlar,SI_12_dl_fit[ii,jj,*],thi=1,lin=0,col=3
        endif
endif
     endfor
  endfor
end

;================================================
pro get_WN,sig_obs,$                  ; input  2Darr of observed line width [cm/s]
           W_Z,$                      ; input  atomic weight of the atom [g/mol]
           logT,$                     ; input  formation temperature in log10 scale [K]
           wid_nth                    ; output 2Darr of non-thermal width [cm/s]
  k_b=!const.k*1e7                    ; Boltzman constant [J/K]->[erg/K]
  N_A=!const.Na*1e0                   ; Avogadro constant [/mol]
  m_Z=W_Z/N_A                         ; mass of the atom [g]
  wid_the=sqrt(2.*k_b*(10.^logT)/m_Z) ; thermal width of formation temp in velocity [cm/s]
  wid_obs=sqrt(2)*sig_obs             ; "thermal width" of observed line width [cm/s]
  wid_nth=sqrt(wid_obs^2-wid_the^2)   ; "thermal width" of non-thermal width [cm/s]
end

;================================================
pro plot_image_log_lin,image,log=log,min_log=min_log,max_log=max_log,min_lin=min_lin,max_lin=max_lin,_extra=_extra
  if log then begin
     plot_image,alog10(image>0.),min=min_log,max=max_log,_extra=_extra
  endif else begin
     plot_image,image,min=min_lin,max=max_lin,_extra=_extra
  endelse
end

;================================================
pro oplot_psym,slp1,slp2,psym,_extra=_extra
  for ii=0,nel(slp1)-1 do oplot,[slp1[ii]],[slp2[ii]],psym=psym[ii],symsize=1.,_extra=_extra
end
;================================================
pro oplot_psym2,slp1,slp2,psym,_extra=_extra
  for ii=0,nel(slp1)-1 do oplot,[slp1[ii]],[slp2[ii]],psym=psym[ii],symsize=1.,_extra=_extra
  oplot,slp1,slp2,symsize=1.,_extra=_extra
end


