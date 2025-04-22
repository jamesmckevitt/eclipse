;Last Updated: <2021/12/29 11:53:31 from tambp.local by teiakiko>

;===============================================================
; Prepare a Box of Atmosphere from the Periodic Atmosphere
; Define a Domain to Synthesize
;===============================================================
case domain of
   0: begin
      xst=0   & xen=nx-1
      yst=0   & yen=ny-1
      zst=0   & zen=nz-1
   end
endcase

; Get various cubes for the domain
;get_cubes_domain2,$
get_var_domain,xst,xen,yst,yen,zst,zen,dx,dy,dz,$
               nx,  ny,  nz,  sxar,  syar,  szar,$
               nx_d,ny_d,nz_d,sxar_d,syar_d,szar_d,sx_d,sy_d,sz_d


print,'A domain to synthesize is set.'
