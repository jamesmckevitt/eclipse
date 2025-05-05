PRO make_goft

compile_opt idl2

  ; Density & temperature grids
  nN = 2                           ; number of density points
  nT = 101                         ; number of temperature points (fixed by CHIANTI g_of_t output)
  logN_min = 7.0                   ; make sure density range covers the simulation range
  logN_max = 20.5
  logT_min = 4.0                   ; range fixed by CHIANTI g_of_t output
  logT_max = 9.0

  logNarr = logN_min + (findgen(nN)/(nN-1)) * (logN_max - logN_min)
  logTarr = logT_min + (findgen(nT)/(nT-1)) * (logT_max - logT_min)

  print,"Density and temperature grids set."

  ; CHIANTI files
  ioneq_file = '/home/jm/solar/ssw/packages/chianti/dbase/ioneq/chianti.ioneq'
  abund_file = '/home/jm/solar/ssw/packages/chianti/dbase/abundance/sun_coronal_2021_chianti.abund'

  ; Emission lines
  emission_lines = ['Fe12_195.1190']
  atoms          = [26             ] ; proton number
  ions           = [12             ] ; ionisation stage
  indices        = [30051          ] ; CHIANTI index for each

  nLines = n_elements(emission_lines)

  print,"Setting up data structure..."

  ; Build the data structure
  template = { $
    name: '', $
    atom: 0L, $
    ion:  0L, $
    index:0L, $
    goft: dblarr(nT, nN) $
  }
  goftArr = replicate(template, nLines)

  ; Loop over lines and densities, fill the structure
  for i = 0, nLines-1 do begin
    goftArr[i].name  = emission_lines[i]
    goftArr[i].atom  = atoms[i]
    goftArr[i].ion   = ions[i]
    goftArr[i].index = indices[i]

    for dd = 0, nN-1 do begin
      goftArr[i].goft[*, dd] = g_of_t( $
        goftArr[i].atom, $
        goftArr[i].ion, $
        index      = goftArr[i].index, $
        ioneq_file = ioneq_file, $
        abund_file = abund_file, $
        dens       = logNarr[dd], $
        LogT       = LogT $
      )

      ; verify all the values of LogT perfectly matches logTarr
      if (min(LogT - logTarr) gt 0.0001) then print, 'ERROR!!: LogT does not match logTarr'

    endfor
  endfor

  print,"G(T,N) calculated. Saving..."

  fname = '/home/jm/solar/solc/solc_euvst_sw_response/G_of_T.sav'
  save, goftArr, logTarr, logNarr, filename = '/home/jm/solar/solc/solc_euvst_sw_response/G_of_T.sav'

  print,"G(T,N) saved to ", fname

END
