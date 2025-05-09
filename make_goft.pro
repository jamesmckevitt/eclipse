; TODO: Improve this by using emiss_calc and calculate goft ourselves, to as not to need a catalogue of transition indexes

PRO make_goft

compile_opt idl2

  ; Density & temperature grids
  nN = 51                           ; number of density points
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

  ; Lines (and their transition index) from CHIANIT surrounding Fe XII 195.119
  ; 24318    195.0250     9.76e+00 Fe XI *       6.15   3s2 3p3 3d 1F3 - 3s2 3p2 3d2 1G4
  ; 15963    195.0290     7.56e+00 Fe IX *       6.00   3s2 3p5 3d 1F3 - 3s2 3p4 3d2 1D2
  ; 30038    195.0860     3.97e+00 Fe XII *      6.20   3s2 3p2 3d 4D7/2 - 3s2 3p 3d2 4F9/2
  ; 30051    195.1190     2.24e+03 Fe XII        6.20   3s2 3p3 4S3/2 - 3s2 3p2 3d 4P5/2
  ; 30073    195.1790     2.93e+02 Fe XII        6.20   3s2 3p3 2D3/2 - 3s2 3p2 3d 2D3/2
  ; 6988     195.2600     8.06e+00 Fe X *        6.05   3s2 3p4 3d 2F7/2 - 3s2 3p3 3d2 2G9/2
  ; 6989     195.2610     2.66e+01 Fe X *        6.05   3s2 3p4 3d 2F7/2 - 3s2 3p3 3d2 2G9/2
  ; 81       195.2710     2.31e+01 Ni XVI        6.45   3s2.3p 2P3/2 - 3s2.3d 2D3/2

  ; Emission lines
  emission_lines = ['Fe11_195.0250', 'Fe09_195.0290', 'Fe12_195.0860', 'Fe12_195.1190', 'Fe12_195.1790', 'Fe10_195.2600', 'Fe10_195.2610', 'Ni16_195.2710']
  atoms          = [             26,              26,              26,              26,              26,              26,              26,              28]
  ions           = [             11,               9,              12,              12,              12,              10,              10,              16]
  indices        = [          24318,           15963,           30038,           30051,           30073,             6988,           6989,              81]
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

      print, ' Max value for that density = ', max(goftArr[i].goft[*, dd])

      ; verify all the values of LogT perfectly matches logTarr
      if (min(LogT - logTarr) gt 0.0001) then print, 'ERROR!!: LogT does not match logTarr'

    endfor

    print, 'Completed line ', i, ' of ', nLines
    ; print the maximum value in the array
    print, 'Max value in goftArr[', i, '].goft = ', max(goftArr[i].goft)
  endfor

  print,"G(T,N) calculated. Saving..."

  fname = '/home/jm/solar/solc/solc_euvst_sw_response/gofnt.sav'
  save, goftArr, logTarr, logNarr, filename = fname

  print,"G(T,N) saved to ", fname

END