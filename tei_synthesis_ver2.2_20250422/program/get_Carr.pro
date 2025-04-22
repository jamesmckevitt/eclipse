;Last Updated: <2021/12/29 12:51:21 from tambp.local by teiakiko>

print,' '
print,'****************************************'
print,'Getting Carr...'

;================================================
; Read MURaM atmosphere
; -> depending on logN, t_p
;================================================
@read_atm_N_T

;================================================
; Restore G(T,N)
;================================================
restore,dir_G_of_T_N+'G_of_T_N_'+lstr+'.sav',/verbose

;================================================
; Get Carr
;================================================
Garr=fltarr(nx,ny,nz)
Carr=fltarr(nx,ny,nz)

;;; ORIGINAL VERSION ;;;
;; nd=nel(logNarr)
;; for xx=0,nx-1 do begin & pp, xx
;;    for yy=0,ny-1 do for zz=0,nz-1 do begin
;;       if t_p[xx,yy,zz] ge 1e4 && t_p[xx,yy,zz] le 1e9 then begin
;;          tmp=G_of_T_N[0,*]*0.
;;          for dd=0,nd-1 do tmp[dd]=interpol(G_of_T_N[*,dd],logTarr,alog10(t_p[xx,yy,zz])) ; [erg/s cm^3]
;;          Garr[xx,yy,zz]=interpol(tmp[*],logNarr,logN[xx,yy,zz])                          ; [erg/s cm^3]
;;          Carr[xx,yy,zz]=Garr[xx,yy,zz]/(4*!pi)*(10.^logN[xx,yy,zz])^2                    ; [erg/s /cm^3/sr]
;;       endif
;;    endfor
;; endfor
;; print,' '
;;;;;;
;;; HIGH-SPEED VERSION: Updated on 2022/06/20 by S. Toriumi ;;;
alog10tp1d=reform(alog10(t_p),ulong(nx)*ulong(ny)*ulong(nz))
logN1d=reform(logN,ulong(nx)*ulong(ny)*ulong(nz))
Garr[*,*,*]=reform(interp2d(G_of_T_N,logTarr,logNarr,alog10tp1d,logN1d),nx,ny,nz)        ; [erg/s cm^3]
Garr[where((t_p lt 1e4) or (t_p gt 1e9))]=0.0
Carr=Garr/(4*!pi)*(10.^logN)^2                                                           ; [erg/s /cm^3/sr]
print,' '
;;;;;;

print,'   Removing logN,t_p, ...'
delvarx,logN,t_p

print,'   Removing, Garr, ...'
delvarx,Garr

indices=where(not float(finite(Carr))) ; Get NAN positions
Carr2=Carr
wheretomulti,Carr,indices,col,row,frame
for jj=0,nel(indices)-1 do Carr2[col[jj],row[jj],frame[jj]]=0.

print,'   Saving Carr,Carr2, ...'
save,f=dir_Carr_ll+'Carr_'+lstr+'_'+tstep+'.sav',Carr,Carr2

print,'   Removing, Carr,Carr2, ...'
delvarx,Carr,Carr2

print,'****************************************'

