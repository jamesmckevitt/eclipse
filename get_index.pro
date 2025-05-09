; .compile get_index.pro
; get_index

pro get_index

    ioneq_file = '/home/jm/solar/ssw/packages/chianti/dbase/ioneq/chianti.ioneq'
    abund_file = '/home/jm/solar/ssw/packages/chianti/dbase/abundance/archive/sun_coronal_2012_schmelz_ext.abund'
    iz=25
    ion=10
    wrange=[194.95,195.3]

    read_ioneq,ioneq_file,temp_all,ioneq,ref

    ioneq=REFORM(ioneq(*,iz-1,ion-1))
    ind=WHERE(ioneq NE 0.)          ; for "temp" of emiss_calc

    emiss=emiss_calc(iz,ion,temp=temp_all(ind),dens=10,abund_file=abund_name,ioneq_file=ioneq_name)
    cal_emiss=emiss_select(emiss,wrange=wrange,sel_ind=sel_ind)

    help,sel_ind

end