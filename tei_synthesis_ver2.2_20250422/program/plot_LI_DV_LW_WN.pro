;Last Updated: <2022/01/06 13:59:06 from tambp.local by teiakiko>

;================================================
; restore
;================================================
; Read LI, DV, DW for MURaM 3D Atmosphere
restore,dir_LI_ll+'LI_'+lstr+'_d'+sttri(domain)+'_'+tstep+'.sav';,/verbose
restore,dir_DV_LW_WN_ll+'DV_LW_WN_'+lstr+'_d'+sttri(domain)+'_xy'+'_'+tstep+'.sav';,/verbose
restore,dir_DV_LW_WN_ll+'DV_LW_WN_'+lstr+'_d'+sttri(domain)+'_yz'+'_'+tstep+'.sav';,/verbose
restore,dir_DV_LW_WN_ll+'DV_LW_WN_'+lstr+'_d'+sttri(domain)+'_xz'+'_'+tstep+'.sav';,/verbose
delvarx,SI_xy_dl_fit,SI_xz_dl_fit,SI_yz_dl_fit

;================================================
; Color bar
;================================================
cbarr=[[indgen(256)],[indgen(256)]] ;array for color bar
cbarr2=transpose(cbarr)             ;array for color bar

;================================================
; Common parameters
;================================================
ori_xy=[sxar_d[0],syar_d[0]]/1e8   ;[cm]->[Mm]
ori_xz=[sxar_d[0],szar_d[0]]/1e8   ;[cm]->[Mm]
ori_yz=[syar_d[0],szar_d[0]]/1e8   ;[cm]->[Mm]
sca_xy=[dx,dy]/1e8              ;[cm]->[Mm]
sca_yz=[dy,dz]/1e8              ;[cm]->[Mm]
sca_xz=[dx,dz]/1e8              ;[cm]->[Mm]
xtit_xy='X [Mm]';'Solar-X [Mm]'
xtit_xz='X [Mm]';'Solar-X [Mm]'
xtit_yz='Y [Mm]';'Solar-Y [Mm]'
ytit_xy='Y [Mm]';'Solar-Y [Mm]'
ytit_xz='Z [Mm]';'Solar-Z [Mm]'
ytit_yz='Z [Mm]';'Solar-Z [Mm]'
tit_xy='Disk'
tit_xz='Face-on'
tit_yz='Edge-on'

ngrid_x=fix(sxar_d[-1]/1e8/10)-fix(sxar_d[0]/1e8/10) ;8
ngrid_y=fix(syar_d[-1]/1e8/10)-fix(syar_d[0]/1e8/10) ;4
ngrid_z=fix(szar_d[-1]/1e8/10)-fix(szar_d[0]/1e8/10) ;4
;print,ngrid_x,ngrid_y,ngrid_z

;================================================
; Window
;================================================
wscale=0.8
wxs=1500.*(sx_d+sx_d+sy_d)/(sx+sx+sy)*wscale ;300*5
wys=300*4.*(sz_d)/(sz)*wscale ;300*3
window,1,xs=wxs,ys=wys
wset,1

;================================================
; Position
;================================================
; main plot
scale=0.75
pos=fltarr(4,3,4)
pos[0,0,*]=0.05
pos[2,0,*]=pos[0,0,0]+(sx_d)/(sx_d+sx_d+sy_d)*scale
pos[0,1,*]=pos[2,0,0]+0.05
pos[2,1,*]=pos[0,1,*]+(sx_d)/(sx_d+sx_d+sy_d)*scale
pos[0,2,*]=pos[2,1,0]+0.05
pos[2,2,*]=pos[0,2,*]+(sy_d)/(sx_d+sx_d+sy_d)*scale
;for yy=0,4-1 do pos[1,*,yy]=0.07+0.24*yy
;for yy=0,4-1 do pos[3,0,yy]=pos[1,0,yy]+(sy_d)/(sx_d+sx_d+sy_d)*(1.*wxs/wys)*scale
;for yy=0,4-1 do pos[3,1:2,yy]=pos[1,0,yy]+(sz_d)/(sx_d+sx_d+sy_d)*(1.*wxs/wys)*scale
pos[1,*,0]=0.05
pos[3,*,0]=pos[1,0,0]+(sz_d)/(sx_d+sx_d+sy_d)*(1.*wxs/wys)*scale
pos[1,*,1]=pos[3,0,0]+0.05
pos[3,*,1]=pos[1,0,1]+(sz_d)/(sx_d+sx_d+sy_d)*(1.*wxs/wys)*scale
pos[1,*,2]=pos[3,0,1]+0.05
pos[3,*,2]=pos[1,0,2]+(sz_d)/(sx_d+sx_d+sy_d)*(1.*wxs/wys)*scale
pos[1,*,3]=pos[3,0,2]+0.05
pos[3,*,3]=pos[1,0,3]+(sz_d)/(sx_d+sx_d+sy_d)*(1.*wxs/wys)*scale
; color bar
poscb=fltarr(4,4)
poscb[0,*]=pos[2,2,0]+0.06
poscb[2,*]=pos[2,2,0]+0.08
poscb[1,*]=pos[1,1,*]
poscb[3,*]=pos[3,1,*]

;================================================
; Settings
;================================================
!p.multi=[0,3+1,4]
!p.charsize=3*wscale
!x.tickinterval=10
!y.tickinterval=10
!x.style=1
!y.style=1

;=============================================================================================
; Plot Map of LI, DV, LW, WN
;=============================================================================================

;plot_image,dist(10,10)
;plot,indgen(10)
;write_png,'test.png',tvrd(/true)
;stop

;-----------------------------
; Line Intensity
;-----------------------------
loadct,0,/sil
; xy view
plot_image_log_lin,LI_xy,log=log,min_log=min_LIlog,max_log=max_LIlog,min_lin=min_LIlin,max_lin=max_LIlin,$
                   ori=ori_xy,sca=sca_xy,xtit=xtit_xy,ytit=ytit_xy,tit=tit_xy,pos=pos[*,0,3]
for xx=0,ngrid_x do xb,fix(sxar_d[0]/1e8/10)*10+xx*10,[syar_d[0],syar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_y do yb,fix(syar_d[0]/1e8/10)*10+yy*10,[sxar_d[0],sxar_d[-1]]/1e8,line=1,col=255
; xz view
plot_image_log_lin,LI_xz,log=log,min_log=min_LIlog,max_log=max_LIlog,min_lin=min_LIlin,max_lin=max_LIlin,$
                   ori=ori_xz,sca=sca_xz,xtit=xtit_xz,ytit=ytit_xz,tit=tit_xz,pos=pos[*,1,3]
for xx=0,ngrid_x do xb,fix(sxar_d[0]/1e8/10)*10+xx*10,[szar_d[0],szar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_z do yb,fix(szar_d[0]/1e8/10)*10+yy*10,[sxar_d[0],sxar_d[-1]]/1e8,line=1,col=255
; yz view
plot_image_log_lin,LI_yz,log=log,min_log=min_LIlog,max_log=max_LIlog,min_lin=min_LIlin,max_lin=max_LIlin,$
                   ori=ori_yz,sca=sca_yz,xtit=xtit_yz,ytit=ytit_yz,tit=tit_yz,pos=pos[*,2,3]
for xx=0,ngrid_y do xb,fix(syar_d[0]/1e8/10)*10+xx*10,[szar_d[0],szar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_z do yb,fix(szar_d[0]/1e8/10)*10+yy*10,[syar_d[0],syar_d[-1]]/1e8,line=1,col=255
; color bar
if log then begin
   plot_image,cbarr2,ori=[0,min_LIlog],sca=[1,(max_LIlog-min_LIlog)/255],ytickin=0.5,yticklen=0.1,$
              xtickin=2,xtickna=replicate(' ',50),xticklen=1e-5,pos=poscb[*,3],$
;                    ytit='log!D10!N (Line Intensity [erg s!U-1!N cm!U-2!N sr!U-1!N])'
              ytit='[erg s!U-1!N cm!U-2!N sr!U-1!N]'
   xyouts,poscb[0,3]-0.04,(poscb[1,3]+poscb[3,3])/2.,ali=0.5,'log!D10!N (Line Intensity)',/nor,ori=90,charsi=1.5
endif else begin
   plot_image,cbarr2,ori=[0,min_LIlin/1e3],sca=[1,(max_LIlin-min_LIlin)/1e3/255],ytickin=2,yticklen=0.1,$
              xtickin=2,xtickna=replicate(' ',50),xticklen=1e-5,pos=poscb[*,3],$
;                 ytickin=long((max_LIlin-min_LIlin)/1e3/255/500)*100,$
;                 ytit='Line Intensity [10!U3!N erg s!U-1!N cm!U-2!N sr!U-1!N]'
                 ytit='[10!U3!N erg s!U-1!N cm!U-2!N sr!U-1!N]'
   xyouts,poscb[0,3]-0.04,(poscb[1,3]+poscb[3,3])/2.,align=0.5,'Line Intensity',/nor,ori=90,charsi=1.5
endelse
xyouts,pos[0,0,3]-0.04,pos[3,0,3]+0.03,align=0,numbering(tt,3)+', '+tstep+', #'+sttri(ll)+', '+lstr+', logT='+sttri(logT),/nor,charsi=1.5*wscale

;-----------------------------
; Doppler velocity
;-----------------------------
;lbr
loadct,70,/sil
; xy view
plot_image,DV_xy/1e5,min=min_DV,max=max_DV,ori=ori_xy,sca=sca_xy,xtit=xtit_xy,ytit=ytit_xy,pos=pos[*,0,2]
                                ;mima,DV_xy/1e5
                                ;=>  -294.00000       0.0000000 [km/s] ... flare atmosphere
                                ;=>  -87.378305       1186.6396 [km/s] ... this atmosphere
for xx=0,ngrid_x do xb,fix(sxar_d[0]/1e8/10)*10+xx*10,[syar_d[0],syar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_y do yb,fix(syar_d[0]/1e8/10)*10+yy*10,[sxar_d[0],sxar_d[-1]]/1e8,line=1,col=255
; xz view
plot_image,DV_xz/1e5,min=min_DV,max=max_DV,ori=ori_xz,sca=sca_xz,xtit=xtit_xz,ytit=ytit_xz,pos=pos[*,1,2]
                                ;mima,DV_xz/1e5
                                ;=>  -96.671191       183.41449 [km/s] ... flare atmosphere
                                ;=>  -103.84755       194.07426 [km/s] ... this atmosphere
for xx=0,ngrid_x do xb,fix(sxar_d[0]/1e8/10)*10+xx*10,[szar_d[0],szar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_z do yb,fix(szar_d[0]/1e8/10)*10+yy*10,[sxar_d[0],sxar_d[-1]]/1e8,line=1,col=255
; yz view
plot_image,DV_yz/1e5,min=min_DV,max=max_DV,ori=ori_yz,sca=sca_yz,xtit=xtit_yz,ytit=ytit_yz,pos=pos[*,2,2]
                                ;mima,DV_yz/1e5
                                ;=>  -102.41405       158.70688 [km/s] ... flare atmosphere
                                ;=>  -217.45933       116.48674 [km/s] ... this atmosphere
for xx=0,ngrid_y do xb,fix(syar_d[0]/1e8/10)*10+xx*10,[szar_d[0],szar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_z do yb,fix(szar_d[0]/1e8/10)*10+yy*10,[syar_d[0],syar_d[-1]]/1e8,line=1,col=255
; color bar
;plot_image,cbarr2,ori=[0,min_DV],sca=[1,(max_DV-min_DV)/255.],ytickin=5,yticklen=0.1,$
plot_image,cbarr2,ori=[0,min_DV],sca=[1,(max_DV-min_DV)/255.],ytickin=10,yticklen=0.1,$
           xtickin=1,xtickna=replicate(' ',50),xticklen=1e-5,pos=poscb[*,2],$
;              ytit='Doppler Velocity [km s!U-1!N]'
           ytit='V!LDop!N [km s!U-1!N]'
;-----------------------------
; Line width
;-----------------------------
loadct,4,/sil
; xy view
plot_image,LW_xy/1e5,min=min_LW,max=max_LW,ori=ori_xy,sca=sca_xy,xtit=xtit_xy,ytit=ytit_xy,pos=pos[*,0,1]
                                ;mima,LW_xy/1e5
                                ;=>  0.0000000       0.0000000 [km/s] ... flare atmosphere
                                ;=>  0.0000000       398.44093 [km/s] ... this atmosphere
for xx=0,ngrid_x do xb,fix(sxar_d[0]/1e8/10)*10+xx*10,[syar_d[0],syar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_y do yb,fix(syar_d[0]/1e8/10)*10+yy*10,[sxar_d[0],sxar_d[-1]]/1e8,line=1,col=255
; xz view
plot_image,LW_xz/1e5,min=min_LW,max=max_LW,ori=ori_xz,sca=sca_xz,xtit=xtit_xz,ytit=ytit_xz,pos=pos[*,1,1]
                                ;mima,LW_xz/1e5
                                ;=>  0.0000000       87.634464 [km/s] ... flare atmosphere
                                ;=>  0.0000000       143.83077 [km/s] ... this atmosphere
for xx=0,ngrid_x do xb,fix(sxar_d[0]/1e8/10)*10+xx*10,[szar_d[0],szar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_z do yb,fix(szar_d[0]/1e8/10)*10+yy*10,[sxar_d[0],sxar_d[-1]]/1e8,line=1,col=255
; yz view
plot_image,LW_yz/1e5,min=min_LW,max=max_LW,ori=ori_yz,sca=sca_yz,xtit=xtit_yz,ytit=ytit_yz,pos=pos[*,2,1]
                                ;mima,LW_yz/1e5
                                ;=>  0.0000000       85.277680 [km/s] ... flare atmosphere
                                ;=>  0.0000000       139.36834 [km/s] ... this atmosphere
for xx=0,ngrid_y do xb,fix(syar_d[0]/1e8/10)*10+xx*10,[szar_d[0],szar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_z do yb,fix(szar_d[0]/1e8/10)*10+yy*10,[syar_d[0],syar_d[-1]]/1e8,line=1,col=255
; color bar
plot_image,cbarr2,ori=[0,min_LW],sca=[1,(max_LW-min_LW)/255.],ytickin=5,yticklen=0.1,$
           xtickin=1,xtickna=replicate(' ',50),xticklen=1e-5,pos=poscb[*,1],$
           ytit='W!Ltot!N [km s!U-1!N]'
;-----------------------------
; Non-thermal width
;-----------------------------
loadct,5,/sil
; xy view
plot_image,WN_xy/1e5,min=min_WN,max=max_WN,ori=ori_xy,sca=sca_xy,xtit=xtit_xy,ytit=ytit_xy,pos=pos[*,0,0]
                                ;mima,WN_xy/1e5
                                ;=>  0.0000000       0.0000000 [km/s] ... flare atmosphere
                                ;=>  0.0000000       562.95311 [km/s] ... this atmosphere
for xx=0,ngrid_x do xb,fix(sxar_d[0]/1e8/10)*10+xx*10,[syar_d[0],syar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_y do yb,fix(syar_d[0]/1e8/10)*10+yy*10,[sxar_d[0],sxar_d[-1]]/1e8,line=1,col=255
; xz view
plot_image,WN_xz/1e5,min=min_WN,max=max_WN,ori=ori_xz,sca=sca_xz,xtit=xtit_xz,ytit=ytit_xz,pos=pos[*,1,0]
                                ;mima,WN_xz/1e5
                                ;=>  0.0000000       87.634464 [km/s] ... flare atmosphere
                                ;=>  0.0000000       201.94170 [km/s] ... this atmosphere
for xx=0,ngrid_x do xb,fix(sxar_d[0]/1e8/10)*10+xx*10,[szar_d[0],szar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_z do yb,fix(szar_d[0]/1e8/10)*10+yy*10,[sxar_d[0],sxar_d[-1]]/1e8,line=1,col=255
; yz view
plot_image,WN_yz/1e5,min=min_WN,max=max_WN,ori=ori_yz,sca=sca_yz,xtit=xtit_yz,ytit=ytit_yz,pos=pos[*,2,0]
                                ;mima,WN_yz/1e5
                                ;=>  0.0000000       85.277680 [km/s] ... flare atmosphere
                                ;=>  0.0000000       195.58359 [km/s] ... this atmosphere
for xx=0,ngrid_y do xb,fix(syar_d[0]/1e8/10)*10+xx*10,[szar_d[0],szar_d[-1]]/1e8,line=1,col=255
for yy=0,ngrid_z do yb,fix(szar_d[0]/1e8/10)*10+yy*10,[syar_d[0],syar_d[-1]]/1e8,line=1,col=255
; color bar
plot_image,cbarr2,ori=[0,min_WN],sca=[1,(max_WN-min_WN)/255.],ytickin=10,yticklen=0.1,$
           xtickin=1,xtickna=replicate(' ',50),xticklen=1e-5,pos=poscb[*,0],$
           ytit='W!LNT!N [km s!U-1!N]'
;=============================================================================================
; Save image
;=============================================================================================
if log eq 1 then write_png,dir_plot_LI_DV_LW_WN_ll+'LIlog_DV_LW_WN_'+'#'+sttri(ll)+'_'+lstr+'_d'+sttri(domain)+'_'+numbering(tt,3)+'.png',tvrd(/true)
if log eq 0 then write_png,dir_plot_LI_DV_LW_WN_ll+'LIlin_DV_LW_WN_'+'#'+sttri(ll)+'_'+lstr+'_d'+sttri(domain)+'_'+numbering(tt,3)+'.png',tvrd(/true)



;================================================
; Settings
;================================================
loadct,0,/sil
!p.multi=0
!p.charsize=0
!x.tickinterval=0
!y.tickinterval=0
!x.style=0
!y.style=0
