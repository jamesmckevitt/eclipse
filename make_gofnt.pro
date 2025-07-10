PRO make_goft

  ; Density & temperature grids
  nN = 51                          ; number of density points
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
  abund_file = '/home/jm/solar/ssw/packages/chianti/dbase/abundance/archive/sun_coronal_2012_schmelz_ext.abund'

  ; Emission lines
  emission_lines = ['Fe13_194.9800', 'Fe14_194.9910', 'Fe13_194.9970', 'Fe12_195.0040', 'Fe11_195.0250', 'Fe09_195.0290', 'Mn10_195.0360', 'Fe11_195.0540', 'Fe12_195.0860', 'Fe12_195.1190', 'Fe11_195.1470', 'Fe10_195.1510', 'Fe13_195.1600', 'Ni11_195.1600', 'Fe12_195.1790', 'Fe09_195.2300', 'Fe14_195.2450', 'Fe11_195.2450', 'Fe10_195.2600', 'Fe10_195.2610', 'Fe12_195.2660']
  atoms          = [             26,              26,              26,              26,              26,              26,              25,              26,              26,              26,              26,              26,              26,              28,              26,              26,              26,              26,              26,              26,              26]
  ions           = [             13,              14,              13,              12,              11,               9,              10,              11,              12,              12,              11,              10,              13,              11,              12,               9,              14,              11,              10,              10,              12]
  wavelength     = [       194.9800,        194.9910,        194.9970,        195.0040,        195.0250,        195.0290,        195.0360,        195.0540,        195.0860,        195.1190,        195.1470,        195.1510,        195.1600,        195.1600,        195.1790,        195.2300,        195.2450,        195.2450,        195.2600,        195.2610,        195.2660]

  indices = intarr(n_elements(wavelength))
  for i = 0, n_elements(wavelength)-1 do begin
    e = emiss_calc(atoms[i], ions[i], /quiet)
    t = min(abs(e.lambda - wavelength[i]), index)
    indices[i] = index
  endfor

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
    print, 'Starting line ', i, ' of ', nLines
    goftArr[i].name  = emission_lines[i]
    goftArr[i].atom  = atoms[i]
    goftArr[i].ion   = ions[i]
    goftArr[i].index = indices[i]

    for dd = 0, nN-1 do begin

      print, 'Starting density ', dd, ' of ', nN, 'for line ', i, ' of ', nLines

      tmp = g_of_t( $
        goftArr[i].atom, $
        goftArr[i].ion, $
        index      = goftArr[i].index, $
        ioneq_file = ioneq_file, $
        abund_file = abund_file, $
        dens       = logNarr[dd], $
        LogT       = LogT $
      )

      goftArr[i].goft[*, dd] = tmp

      ; verify all the values of LogT perfectly matches logTarr
      if (min(LogT - logTarr) gt 0.0001) then print, 'ERROR!!: LogT does not match logTarr'

    endfor

    print, 'Completed line ', i, ' of ', nLines
  endfor

  print,"G(T,N) calculated. Saving..."

  fname = './data/gofnt.sav'
  save, goftArr, logTarr, logNarr, filename = fname

  print,"G(T,N) saved to ", fname

END