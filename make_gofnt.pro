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

  ; Wavelength (A) Intensity  Ion        Tmax  Transition
  ; 194.9800     2.89e-01 Fe XIII *     6.25   3s 3p3 3D1 - 3s 3p2 3d 3P0
  ; 194.9910     1.94e+00 Fe XIV *      6.30   3s2.4f 2F5/2 - 3s2.5g 2G7/2
  ; 194.9970     6.46e-01 Fe XIII *     6.25   3s 3p3 1P1 - 3s2 3d2 3F2
  ; 195.0040     5.08e+00 Fe XII *      6.20   3s 3p4 4P5/2 - 3s 3p3 3d 2F7/2
  ; 195.0250     9.76e+00 Fe XI *       6.15   3s2 3p3 3d 1F3 - 3s2 3p2 3d2 1G4
  ; 195.0290     7.56e+00 Fe IX *       6.00   3s2 3p5 3d 1F3 - 3s2 3p4 3d2 1D2
  ; 195.0360     3.86e+00 Mn X          6.10   3s2.3p4 3P0 - 3s2.3p3(4S).3d 3D1
  ; 195.0540     1.60e+00 Fe XI *       6.15   3s2 3p3 3d 3P1 - 3s2 3p2 3d2 3D2
  ; 195.0860     3.97e+00 Fe XII *      6.20   3s2 3p2 3d 4D7/2 - 3s2 3p 3d2 4F9/2
  ; 195.1190     2.24e+03 Fe XII        6.20   3s2 3p3 4S3/2 - 3s2 3p2 3d 4P5/2
  ; 195.1470     6.29e-01 Fe XI *       6.15   3s2 3p3 3d 1P1 - 3s2 3p2 3d2 3P1
  ; 195.1510     2.84e-01 Fe X *        6.05   3s2 3p4 3d 4D1/2 - 3s2 3p3 3d2 4D3/2
  ; 195.1600     3.57e-01 Fe XIII *     6.25   3s 3p3 3P1 - 3s 3p2 3d 1D2
  ; 195.1600     9.41e-01 Ni XI *       6.15   3s2 3p5 3d 1D2 - 3s2 3p4 3d2 5D3
  ; 195.1790     2.93e+02 Fe XII        6.20   3s2 3p3 2D3/2 - 3s2 3p2 3d 2D3/2
  ; 195.2300     3.29e-01 Fe IX *       6.00   3s2 3p4 3d2 3G5 - 3s2 3p3 3d3 1I6
  ; 195.2450     8.46e-01 Fe XIV        6.30   3s.3p2 4P3/2 - 3s.3p(3P).3d 2F5/2
  ; 195.2450     3.19e-01 Fe XI *       6.15   3s2 3p3 3d 3G4 - 3s2 3p2 3d2 3G4
  ; 195.2600     8.06e+00 Fe X *        6.05   3s2 3p4 3d 2F7/2 - 3s2 3p3 3d2 2G9/2
  ; 195.2610     2.66e+01 Fe X *        6.05   3s2 3p4 3d 2F7/2 - 3s2 3p3 3d2 2G9/2
  ; 195.2660     2.91e-01 Fe XII *      6.20   3s2 3p2 3d 2D5/2 - 3s2 3p 3d2 2D3/2

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

  fname = '/home/jm/solar/solc/solc_euvst_sw_response/gofnt.sav'
  save, goftArr, logTarr, logNarr, filename = fname

  print,"G(T,N) saved to ", fname

END